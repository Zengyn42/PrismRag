"""Pass 3a: Compute embeddings for all text nodes.

Supports three backends selected via PRISM_EMBED_BACKEND env var:

  ollama  (default) — local Ollama, no API key needed
                      Recommended models (set via PRISM_OLLAMA_MODEL):
                        bge-m3               dim=1024  multilingual, 8K ctx (default)
                        qwen3-embedding:8b   dim=1024  #1 MTEB multilingual (GPU required)
                        nomic-embed-text     dim=768   CPU-friendly, 8K ctx
                        mxbai-embed-large    dim=1024  English-focused
                        all-minilm           dim=384   fast, lightweight

  gemini            — Google Gemini API, requires PRISM_GEMINI_API_KEY
                      Recommended models (set via PRISM_GEMINI_EMBED_MODEL):
                        gemini-embedding-001  dim up to 3072  text, #1 MTEB multilingual (default)
                        gemini-embedding-2    multimodal (text/image/video/audio/PDF)

  openai            — OpenAI-compatible /v1/embeddings endpoint
                      Works with vLLM, LM Studio, OpenRouter, etc.

All backends expose the same interface:
  compute_embeddings(graph, settings) → dict[node_id, list[float]]

Use get_embedder(settings) as the single factory for constructing embedders.

Usage:
    from prism_rag.ingest.embedder import get_embedder, compute_embeddings
    embedder = get_embedder(settings)
    qvec = embedder.embed_query("context explosion")
    vectors = compute_embeddings(graph, settings)
"""

from __future__ import annotations

import fcntl
import json
import logging
import time
from pathlib import Path
from typing import Any

from prism_rag.config import PrismRagSettings
from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Gemini limits
_GEMINI_MODEL_DEFAULT = "gemini-embedding-001"  # text, up to 3072 dims; GA May 2026
_MAX_INPUT_CHARS = 30_000
_BATCH_SIZE = 20
_RATE_LIMIT_DELAY = 0.5

# ── Ollama Embedder ───────────────────────────────────────────────────────────

_OLLAMA_DEFAULT_HOST = "http://localhost:11434"


def detect_model_device(model: str, host: str) -> str:
    """Query Ollama /api/ps to find which device the model is using.

    Returns "gpu" | "cpu" | "unknown".
    """
    import urllib.request
    import json as _json

    try:
        url = f"{host.rstrip('/')}/api/ps"
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = _json.loads(resp.read())
        for entry in body.get("models", []):
            if model in entry.get("name", ""):
                return "gpu" if entry.get("size_vram", 0) > 0 else "cpu"
    except Exception:
        pass
    return "unknown"


class OllamaEmbedder:
    """Embed text via a local Ollama model.

    The ``model`` parameter is **required** — there is no silent default.
    Use ``get_embedder(settings)`` factory to construct from config.

    Usage::

        embedder = OllamaEmbedder(model="bge-m3")
        vec = embedder.embed_query("what is context explosion?")
        vecs = embedder.embed_batch(["text a", "text b"])
    """

    def __init__(
        self,
        model: str,  # Required — no default to prevent silent mismatch bugs
        base_url: str = _OLLAMA_DEFAULT_HOST,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self._timeout = timeout

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Raises on failure."""
        return self._call([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns one vector per input."""
        if not texts:
            return []
        return self._call(texts)

    def _call(self, inputs: list[str]) -> list[list[float]]:
        import urllib.request
        import json as _json

        payload = _json.dumps({"model": self.model, "input": inputs}).encode()
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = _json.loads(resp.read())
        embeddings = body.get("embeddings", [])
        if len(embeddings) != len(inputs):
            raise ValueError(
                f"Ollama returned {len(embeddings)} embeddings for {len(inputs)} inputs"
            )
        return [list(e) for e in embeddings]


# ── OpenAI-Compatible Embedder ───────────────────────────────────────────────


class OpenAICompatEmbedder:
    """Embed text via an OpenAI-compatible /v1/embeddings endpoint.

    Works with vLLM, LM Studio, OpenRouter, and any server that implements
    the OpenAI embedding API contract.

    Usage::

        embedder = OpenAICompatEmbedder(
            model="text-embedding-3-small",
            base_url="http://localhost:1234",
            api_key="sk-...",
        )
        vec = embedder.embed_query("hello")
    """

    def __init__(
        self,
        model: str,  # Required — no default
        base_url: str = "http://localhost:1234",
        api_key: str = "",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self._url = f"{base_url.rstrip('/')}/v1/embeddings"
        self._api_key = api_key
        self._timeout = timeout

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        results = self._call([text])
        return results[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns one vector per input."""
        if not texts:
            return []
        return self._call(texts)

    def _call(self, inputs: list[str]) -> list[list[float]]:
        import urllib.request
        import json as _json

        payload = _json.dumps({"input": inputs, "model": self.model}).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = _json.loads(resp.read())

        # OpenAI response: {"data": [{"embedding": [...], "index": 0}, ...]}
        data = body.get("data", [])
        # Sort by index to guarantee order
        data.sort(key=lambda d: d.get("index", 0))
        return [item["embedding"] for item in data]


# ── Embedder Factory ─────────────────────────────────────────────────────────


def get_embedder(
    settings: PrismRagSettings,
    *,
    namespace: str = "",
) -> OllamaEmbedder | OpenAICompatEmbedder:
    """Single factory for constructing the correct embedder from config.

    All call sites must use this factory — never construct OllamaEmbedder()
    or OpenAICompatEmbedder() directly (except in tests).

    If ``namespace`` is provided and a matching GraphSource has per-namespace
    embed overrides (embed_backend, embed_model, embed_dim), those take
    priority over global settings.
    """
    # Resolve per-namespace overrides
    backend = settings.embed_backend
    model = settings.get_embed_model_name()
    host = settings.ollama_host

    if namespace and settings.graphs:
        for gs in settings.graphs:
            if gs.namespace == namespace:
                if gs.embed_backend:
                    backend = gs.embed_backend
                if gs.embed_model:
                    model = gs.embed_model
                break

    if backend == "openai":
        return OpenAICompatEmbedder(
            model=model if model != settings.ollama_model else settings.openai_embed_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
    elif backend == "gemini":
        # Gemini embedder uses the google SDK, not our simple HTTP class.
        # For query-time use, fall back to a thin wrapper or raise.
        # For now, return OllamaEmbedder as a fallback if gemini is set but
        # query-time needs a simple embedder. The compute path uses SDK directly.
        raise ValueError(
            "Gemini backend does not support get_embedder() for query-time use. "
            "Use compute_embeddings() for index-time embedding."
        )
    else:
        # ollama (default)
        return OllamaEmbedder(model=model, base_url=host)


# ── Embed Meta (index/query consistency guard) ──────────────────────────────

_EMBED_META_FILENAME = "embed_meta.json"


def write_embed_meta(
    data_dir: Path,
    backend: str,
    model: str,
    dim: int,
) -> None:
    """Write embed_meta.json to record which model was used at index time."""
    meta_path = data_dir / _EMBED_META_FILENAME
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"backend": backend, "model": model, "dim": dim}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"[embedder] wrote embed_meta.json: {meta}")


def check_embed_consistency(
    settings: PrismRagSettings,
    *,
    data_dir: Path | None = None,
) -> None:
    """Check if current embed config matches the index-time embed_meta.json.

    If embed_meta.json is missing (old data), silently skips.
    If there is a mismatch, logs a prominent WARNING (does not raise).
    """
    target_dir = data_dir or settings.data_dir
    meta_path = target_dir / _EMBED_META_FILENAME
    if not meta_path.exists():
        return  # Old data dir without meta — skip silently

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return  # Corrupt file — skip

    index_backend = meta.get("backend", "")
    index_model = meta.get("model", "")
    index_dim = meta.get("dim", 0)

    current_backend = settings.embed_backend
    current_model = settings.get_embed_model_name()
    current_dim = settings.embedding_dim

    mismatches: list[str] = []
    if index_backend and index_backend != current_backend:
        mismatches.append(f"backend: index={index_backend} vs query={current_backend}")
    if index_model and index_model != current_model:
        mismatches.append(f"model: index={index_model} vs query={current_model}")
    if index_dim and index_dim != current_dim:
        mismatches.append(f"dim: index={index_dim} vs query={current_dim}")

    if mismatches:
        logger.warning(
            "[embedder] EMBEDDING MISMATCH — index model differs from query model, "
            "results will be garbage! %s. "
            "Re-run 'prism-rag ingest' with the current model to fix.",
            "; ".join(mismatches),
        )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _truncate(text: str, max_chars: int = _MAX_INPUT_CHARS) -> str:
    """Truncate text to fit within Gemini's input limit."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


_EMBEDDABLE_KINDS = frozenset({
    # Vault kinds
    "note", "knowledge", "image", "pdf", "audio",
    # Code kinds (CodeParser output)
    "function", "class", "module",
})


def _get_embeddable_nodes(graph: KnowledgeGraph) -> list[tuple[str, str]]:
    """Extract (node_id, content) pairs for all nodes worth embedding.

    Rules:
      - Kind must be content-bearing (see _EMBEDDABLE_KINDS)
      - Content must be non-empty (after strip)
      - Frontmatter.embed == False opts out (Phase 2 tiered embedding)
    """
    pairs: list[tuple[str, str]] = []
    for node_id, data in graph.g.nodes(data=True):
        kind = data.get("kind", "")
        content = data.get("content", "")
        if kind not in _EMBEDDABLE_KINDS:
            continue
        if not content.strip():
            continue
        fm = data.get("frontmatter") or {}
        if fm.get("embed") is False:
            continue
        pairs.append((node_id, content))
    return pairs


def compute_embeddings(
    graph: KnowledgeGraph,
    settings: PrismRagSettings,
    *,
    cache_path: Path | None = None,
) -> dict[str, list[float]]:
    """Compute embeddings for all embeddable nodes using the configured backend.

    Backend is selected by settings.embed_backend ('ollama', 'gemini', or 'openai').
    After computing, writes embed_meta.json to data_dir for consistency checking.

    Args:
        cache_path: Optional path to embed_cache.jsonl for checkpoint/resume support.
                    If provided, already-computed nodes (matched by content_hash) are
                    skipped. New results are appended to the cache file.

    Returns:
        dict mapping node_id → embedding vector (list of floats).
    """
    if settings.embed_backend == "ollama":
        result = _compute_embeddings_ollama(graph, settings, cache_path=cache_path)
    elif settings.embed_backend == "openai":
        result = _compute_embeddings_openai(graph, settings, cache_path=cache_path)
    elif settings.embed_backend == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "PRISM_GEMINI_API_KEY is required when embed_backend='gemini'. "
                "Get one at https://aistudio.google.com/apikey, "
                "or set PRISM_EMBED_BACKEND=ollama for local embedding."
            )
        if settings.privacy_tier == "free":
            logger.warning(
                "[embedder] privacy_tier=free: Gemini free tier may use your data "
                "for model training. Set PRISM_PRIVACY_TIER=paid for production use."
            )
        result = _compute_embeddings_gemini(graph, settings, settings.embed_dimensionality)
    else:
        raise ValueError(f"Unknown embed_backend: {settings.embed_backend!r}")

    # Write embed_meta.json for index/query consistency checking
    if result:
        write_embed_meta(
            settings.data_dir,
            backend=settings.embed_backend,
            model=settings.get_embed_model_name(),
            dim=settings.embedding_dim,
        )

    return result


_OLLAMA_BATCH_SIZE = 16   # texts per HTTP request; tune to GPU VRAM
_OLLAMA_TIMEOUT = 60    # seconds per batch request


def _load_embed_cache(cache_path: Path) -> dict[str, tuple[str, list[float]]]:
    """Load embed_cache.jsonl → {node_id: (sha, vec)}. Last-wins on duplicate node_id."""
    if not cache_path.exists():
        return {}
    result: dict[str, tuple[str, list[float]]] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            result[entry["node_id"]] = (entry["sha"], entry["vec"])
        except Exception:
            continue
    return result


def _append_cache_entry(
    cache_path: Path, node_id: str, sha: str, vec: list[float]
) -> None:
    """Append one cache entry to embed_cache.jsonl with exclusive file lock.

    4096-dimensional float32 vectors are ~65KB per line, well above PIPE_BUF
    (4096 bytes), so we cannot rely on atomic write semantics.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"node_id": node_id, "sha": sha, "vec": vec}, ensure_ascii=False)
    with cache_path.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def gc_embed_cache(
    cache_path: Path,
    live_sha_set: set[tuple[str, str]],
) -> int:
    """Remove stale entries from embed_cache.jsonl.

    An entry is stale if its (node_id, sha) pair is not in live_sha_set.
    Rewrites the file in place. Returns number of entries removed.
    """
    if not cache_path.exists():
        return 0

    cache = _load_embed_cache(cache_path)
    live = set(live_sha_set)
    kept: list[dict] = []
    removed = 0
    for node_id, (sha, vec) in cache.items():
        if (node_id, sha) in live:
            kept.append({"node_id": node_id, "sha": sha, "vec": vec})
        else:
            removed += 1

    if removed > 0:
        with cache_path.open("w", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                for entry in kept:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        logger.info(f"[embedder/gc] removed {removed} stale entries from {cache_path}")

    return removed


def _compute_embeddings_ollama(
    graph: KnowledgeGraph,
    settings: PrismRagSettings,
    cache_path: Path | None = None,
) -> dict[str, list[float]]:
    """Compute embeddings using local Ollama (batched)."""
    nodes_to_embed = _get_embeddable_nodes(graph)
    if not nodes_to_embed:
        logger.info("[embedder/ollama] no embeddable nodes found")
        return {}

    cache: dict[str, tuple[str, list[float]]] = {}
    if cache_path is not None:
        cache = _load_embed_cache(cache_path)

    vectors: dict[str, list[float]] = {}
    pending: list[tuple[str, str]] = []
    for node_id, content in nodes_to_embed:
        node_data = graph.g.nodes[node_id]
        sha = node_data.get("content_hash", "")
        if node_id in cache and cache[node_id][0] == sha:
            vectors[node_id] = cache[node_id][1]
        else:
            pending.append((node_id, content))

    if not pending:
        logger.info("[embedder/ollama] all nodes hit cache, skipping embed call")
        return vectors

    model = settings.ollama_model
    host = settings.ollama_host

    device = detect_model_device(model, host)
    if device == "cpu":
        logger.warning(
            f"[embedder/ollama] model={model} is running on CPU. "
            "Using CPU-safe timeout (300s) and batch_size=1."
        )
        effective_timeout = 300
        effective_batch_size = 1
    else:
        if device == "unknown":
            logger.info(f"[embedder/ollama] could not detect device for model={model}")
        effective_timeout = _OLLAMA_TIMEOUT
        effective_batch_size = _OLLAMA_BATCH_SIZE

    embedder = OllamaEmbedder(model=model, base_url=host, timeout=effective_timeout)
    total = len(pending)
    logger.info(
        f"[embedder/ollama] computing {total} embeddings "
        f"(model={embedder.model}, batch={effective_batch_size}, cache_hits={len(vectors)})"
    )

    for batch_start in range(0, total, effective_batch_size):
        batch = pending[batch_start: batch_start + effective_batch_size]
        node_ids = [nid for nid, _ in batch]
        texts = [_truncate(content) for _, content in batch]
        try:
            vecs = embedder.embed_batch(texts)
            for nid, vec in zip(node_ids, vecs):
                vectors[nid] = vec
                if cache_path is not None:
                    sha = graph.g.nodes[nid].get("content_hash", "")
                    _append_cache_entry(cache_path, nid, sha, vec)
        except Exception as exc:
            logger.error(
                f"[embedder/ollama] batch {batch_start}–{batch_start + len(batch) - 1} failed: {exc}"
                f" — retrying one-by-one"
            )
            # Fall back to single-item calls so one bad node doesn't lose the batch
            for nid, text in zip(node_ids, texts):
                try:
                    vec = embedder.embed_query(text)
                    vectors[nid] = vec
                    if cache_path is not None:
                        sha = graph.g.nodes[nid].get("content_hash", "")
                        _append_cache_entry(cache_path, nid, sha, vec)
                except Exception as exc2:
                    logger.error(f"[embedder/ollama] node {nid} failed: {exc2}")

        done = min(batch_start + _OLLAMA_BATCH_SIZE, total)
        if done % 160 == 0 or done == total:
            logger.info(f"[embedder/ollama] progress: {done}/{total}")

    logger.info(f"[embedder/ollama] done: {len(vectors)}/{total + len(cache)}")
    return vectors


def _compute_embeddings_openai(
    graph: KnowledgeGraph,
    settings: PrismRagSettings,
    cache_path: Path | None = None,
) -> dict[str, list[float]]:
    """Compute embeddings using an OpenAI-compatible /v1/embeddings endpoint."""
    nodes_to_embed = _get_embeddable_nodes(graph)
    if not nodes_to_embed:
        logger.info("[embedder/openai] no embeddable nodes found")
        return {}

    cache: dict[str, tuple[str, list[float]]] = {}
    if cache_path is not None:
        cache = _load_embed_cache(cache_path)

    vectors: dict[str, list[float]] = {}
    pending: list[tuple[str, str]] = []
    for node_id, content in nodes_to_embed:
        node_data = graph.g.nodes[node_id]
        sha = node_data.get("content_hash", "")
        if node_id in cache and cache[node_id][0] == sha:
            vectors[node_id] = cache[node_id][1]
        else:
            pending.append((node_id, content))

    if not pending:
        logger.info("[embedder/openai] all nodes hit cache, skipping embed call")
        return vectors

    embedder = OpenAICompatEmbedder(
        model=settings.openai_embed_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
    )
    total = len(pending)
    batch_size = _OLLAMA_BATCH_SIZE  # reuse same batch size

    logger.info(
        f"[embedder/openai] computing {total} embeddings "
        f"(model={embedder.model}, batch={batch_size}, cache_hits={len(vectors)})"
    )

    for batch_start in range(0, total, batch_size):
        batch = pending[batch_start: batch_start + batch_size]
        node_ids = [nid for nid, _ in batch]
        texts = [_truncate(content) for _, content in batch]
        try:
            vecs = embedder.embed_batch(texts)
            for nid, vec in zip(node_ids, vecs):
                vectors[nid] = vec
                if cache_path is not None:
                    sha = graph.g.nodes[nid].get("content_hash", "")
                    _append_cache_entry(cache_path, nid, sha, vec)
        except Exception as exc:
            logger.error(
                f"[embedder/openai] batch {batch_start}–{batch_start + len(batch) - 1} failed: {exc}"
            )
            for nid, text in zip(node_ids, texts):
                try:
                    vec = embedder.embed_query(text)
                    vectors[nid] = vec
                    if cache_path is not None:
                        sha = graph.g.nodes[nid].get("content_hash", "")
                        _append_cache_entry(cache_path, nid, sha, vec)
                except Exception as exc2:
                    logger.error(f"[embedder/openai] node {nid} failed: {exc2}")

        done = min(batch_start + batch_size, total)
        if done % 160 == 0 or done == total:
            logger.info(f"[embedder/openai] progress: {done}/{total}")

    logger.info(f"[embedder/openai] done: {len(vectors)}/{total + len(cache)}")
    return vectors


def _compute_embeddings_gemini(
    graph: KnowledgeGraph,
    settings: PrismRagSettings,
    dimensionality: int = 768,
) -> dict[str, list[float]]:
    """Compute embeddings using Gemini Embedding API."""
    from google import genai
    from google.genai.types import EmbedContentConfig

    if not settings.gemini_api_key:
        raise ValueError(
            "PRISM_GEMINI_API_KEY is required for Gemini backend. "
            "Set PRISM_EMBED_BACKEND=ollama for local embedding."
        )

    if settings.privacy_tier == "free":
        logger.warning(
            "[embedder/gemini] privacy_tier=free: Gemini free tier may use your data "
            "for model training. Set PRISM_PRIVACY_TIER=paid for production use."
        )

    model_name = getattr(settings, "gemini_embed_model", _GEMINI_MODEL_DEFAULT)
    client = genai.Client(api_key=settings.gemini_api_key)
    config = EmbedContentConfig(output_dimensionality=dimensionality)

    nodes_to_embed = _get_embeddable_nodes(graph)
    if not nodes_to_embed:
        logger.info("[embedder/gemini] no embeddable nodes found")
        return {}

    logger.info(
        f"[embedder/gemini] computing {len(nodes_to_embed)} embeddings "
        f"(model={model_name}, dim={dimensionality})"
    )
    vectors: dict[str, list[float]] = {}
    total = len(nodes_to_embed)

    for i, (node_id, content) in enumerate(nodes_to_embed):
        truncated = _truncate(content)
        try:
            result = client.models.embed_content(
                model=model_name,
                contents=truncated,
                config=config,
            )
            vectors[node_id] = result.embeddings[0].values
        except Exception as e:
            logger.error(f"[embedder/gemini] node {node_id} failed: {e}")
            continue
        if (i + 1) % 20 == 0 or i + 1 == total:
            logger.info(f"[embedder/gemini] progress: {i + 1}/{total}")
        if i + 1 < total:
            time.sleep(_RATE_LIMIT_DELAY)

    logger.info(f"[embedder/gemini] done: {len(vectors)}/{total}")
    return vectors


def persist_embeddings(
    vectors: dict[str, list[float]],
    lance_path: Path,
    dim: int = 768,
) -> int:
    """Persist computed embeddings to LanceDB.

    Args:
        vectors: dict mapping node_id → embedding vector.
        lance_path: Path to LanceDB directory.
        dim: Embedding dimension; used to detect schema mismatches and
             drop+recreate the table if needed (e.g. switching 768→1024).

    Returns:
        Number of embeddings persisted.
    """
    if not vectors:
        return 0

    from prism_rag.store.embedding_store import EmbeddingStore
    store = EmbeddingStore(lance_path, dim=dim)
    for node_id, vec in vectors.items():
        store.upsert(node_id, vec)
    logger.info(f"[embedder] persisted {len(vectors)} embeddings to {lance_path}")
    return len(vectors)
