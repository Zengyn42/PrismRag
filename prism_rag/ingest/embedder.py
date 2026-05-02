"""Pass 3a: Compute embeddings for all text nodes.

Supports two backends selected via PRISM_EMBED_BACKEND env var:

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

Both backends expose the same interface:
  compute_embeddings(graph, settings) → dict[node_id, list[float]]

OllamaEmbedder also provides embed_query(text) for query-time use in hybrid search.

Usage:
    from prism_rag.ingest.embedder import compute_embeddings, OllamaEmbedder
    vectors = compute_embeddings(graph, settings)
    embedder = OllamaEmbedder()
    qvec = embedder.embed_query("context explosion")
"""

from __future__ import annotations

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

_OLLAMA_DEFAULT_MODEL = "bge-m3"
_OLLAMA_DEFAULT_HOST = "http://localhost:11434"


class OllamaEmbedder:
    """Embed text via a local Ollama model (default: bge-m3, dim=1024).

    Works for both index-time (batch) and query-time (single) embedding,
    so the same model is used in both directions — a requirement for
    meaningful similarity comparisons.

    Usage::

        embedder = OllamaEmbedder()
        vec = embedder.embed_query("what is context explosion?")
        vecs = embedder.embed_batch(["text a", "text b"])
    """

    def __init__(
        self,
        model: str = _OLLAMA_DEFAULT_MODEL,
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
) -> dict[str, list[float]]:
    """Compute embeddings for all embeddable nodes using the configured backend.

    Backend is selected by settings.embed_backend ('ollama' or 'gemini').
    Ollama uses settings.ollama_host / settings.ollama_model (local, no key needed).
    Gemini uses settings.gemini_api_key / settings.embed_dimensionality.

    Returns:
        dict mapping node_id → embedding vector (list of floats).
    """
    if settings.embed_backend == "ollama":
        return _compute_embeddings_ollama(graph, settings)
    else:
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
        return _compute_embeddings_gemini(graph, settings, settings.embed_dimensionality)


_OLLAMA_BATCH_SIZE = 16   # texts per HTTP request; tune to GPU VRAM
_OLLAMA_TIMEOUT = 300    # seconds per batch request


def _compute_embeddings_ollama(
    graph: KnowledgeGraph,
    settings: PrismRagSettings | None = None,
) -> dict[str, list[float]]:
    """Compute embeddings using local Ollama (batched)."""
    nodes_to_embed = _get_embeddable_nodes(graph)
    if not nodes_to_embed:
        logger.info("[embedder/ollama] no embeddable nodes found")
        return {}

    model = settings.ollama_model if settings else _OLLAMA_DEFAULT_MODEL
    host = settings.ollama_host if settings else _OLLAMA_DEFAULT_HOST
    embedder = OllamaEmbedder(model=model, base_url=host, timeout=_OLLAMA_TIMEOUT)
    total = len(nodes_to_embed)
    logger.info(
        f"[embedder/ollama] computing {total} embeddings "
        f"(model={embedder.model}, batch={_OLLAMA_BATCH_SIZE})"
    )
    vectors: dict[str, list[float]] = {}

    for batch_start in range(0, total, _OLLAMA_BATCH_SIZE):
        batch = nodes_to_embed[batch_start: batch_start + _OLLAMA_BATCH_SIZE]
        node_ids = [nid for nid, _ in batch]
        texts = [_truncate(content) for _, content in batch]
        try:
            vecs = embedder.embed_batch(texts)
            for nid, vec in zip(node_ids, vecs):
                vectors[nid] = vec
        except Exception as exc:
            logger.error(
                f"[embedder/ollama] batch {batch_start}–{batch_start + len(batch) - 1} failed: {exc}"
                f" — retrying one-by-one"
            )
            # Fall back to single-item calls so one bad node doesn't lose the batch
            for nid, text in zip(node_ids, texts):
                try:
                    vectors[nid] = embedder.embed_query(text)
                except Exception as exc2:
                    logger.error(f"[embedder/ollama] node {nid} failed: {exc2}")

        done = min(batch_start + _OLLAMA_BATCH_SIZE, total)
        if done % 160 == 0 or done == total:
            logger.info(f"[embedder/ollama] progress: {done}/{total}")

    logger.info(f"[embedder/ollama] done: {len(vectors)}/{total}")
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
