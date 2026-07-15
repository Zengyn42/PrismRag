"""PrismRag runtime settings.

All fields overridable via environment variables with PRISM_ prefix,
or via a .env file in the current working directory.

Example:
    export PRISM_VAULT_PATH=~/Foundation/Vault
    export PRISM_GEMINI_API_KEY=...
    export PRISM_PRIVACY_TIER=paid
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DedupConfig(BaseModel):
    """Semantic deduplication settings for atomize_propose."""

    threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold to flag a claim as potentially duplicate "
                    "(0.90 = 90% similar to an existing KNOW node triggers review).",
    )
    calibrated_model: str = Field(
        default="",
        description="Embedding model used during threshold calibration (informational only). "
                    "Empty string means using the default from PrismRagSettings.ollama_model.",
    )
    min_nodes_for_calibration: int = Field(
        default=100,
        ge=1,
        description="Minimum number of KNOW nodes required before dedup is activated "
                    "(cold-start protection — skips dedup when the graph is too small).",
    )


PrivacyTier = Literal["paid", "free"]
EmbedBackend = Literal["ollama", "gemini"]

# Embedding dimensions for known Ollama models (key = model name without :tag or /path prefix).
# Used by PrismRagSettings.embedding_dim when ollama_embed_dim is 0 (auto-detect).
OLLAMA_MODEL_DIMS: dict[str, int] = {
    "qwen3-embedding": 4096,   # #1 MTEB multilingual (Dec 2025), GPU required; default dim=4096
    "bge-m3": 1024,            # dense + sparse + ColBERT, 100+ languages
    "mxbai-embed-large": 1024, # MixedBread, English-focused
    "snowflake-arctic-embed2": 1024,
    "nomic-embed-text": 768,   # CPU-friendly, 8K context, most downloaded
    "jina-embeddings-v2-base-en": 768,
    "paraphrase-multilingual": 768,
    "granite-embedding": 768,  # IBM multilingual
    "all-minilm": 384,         # 23M params, fast
}


@dataclass(frozen=True)
class ClassifierProfile:
    """Per-embedding-model thresholds for EdgeClassifier tier judgment."""

    tier1_min_conf: float
    tier1_top_k: int
    tier1_min_consecutive: int
    tier2_min_conf: float
    tier2_margin: float
    tier2_hard_cap: int
    tier2_min_consecutive: int


_DEFAULT_PROFILES: dict[str, ClassifierProfile] = {
    "bge-m3": ClassifierProfile(
        tier1_min_conf=0.75, tier1_top_k=1, tier1_min_consecutive=2,
        tier2_min_conf=0.70, tier2_margin=0.25, tier2_hard_cap=5, tier2_min_consecutive=2,
    ),
    "qwen3-embedding-8b": ClassifierProfile(
        tier1_min_conf=0.85, tier1_top_k=1, tier1_min_consecutive=2,
        tier2_min_conf=0.78, tier2_margin=0.20, tier2_hard_cap=5, tier2_min_consecutive=2,
    ),
    "default": ClassifierProfile(
        tier1_min_conf=0.85, tier1_top_k=1, tier1_min_consecutive=2,
        tier2_min_conf=0.75, tier2_margin=0.25, tier2_hard_cap=5, tier2_min_consecutive=2,
    ),
}


class GraphSource(BaseModel):
    """Configuration for a single graph source (vault + data directory pair).

    The ``sources`` field controls which source extractors run during ingest:
      - "docs"   — Obsidian vault markdown (default, excludes knowledge/ files)
      - "code"   — Tree-sitter Python AST
      - "memory" — Agent memory files (requires atomize)
      - "knot"   — KNOT files (knowledge/*.md with knowledge_id frontmatter)

    KNOT (Knowledge Ontology Token), file prefix KNOW- kept for data compat.
    """

    namespace: str
    vault_path: Path
    data_dir: Path
    writable: bool = False
    sources: list[str] = Field(
        default=["docs"],
        description="Source extractors to run: 'docs', 'code', 'memory', 'knot'. "
                    "Default ['docs'] preserves backward compatibility.",
    )
    atomize_docs: bool = Field(
        default=False,
        description="Whether to run atomize on docs source (optional, not required).",
    )
    memory_paths: list[Path] = Field(
        default_factory=list,
        description="File or directory paths for memory source (markdown). "
                    "Only used when 'memory' is in sources.",
    )

    @field_validator("sources", mode="before")
    @classmethod
    def _parse_sources(cls, v):
        if isinstance(v, str):
            v = [s.strip() for s in v.split(",") if s.strip()]
        from prism_rag.sources.base import VALID_SOURCES
        for s in v:
            if s not in VALID_SOURCES:
                raise ValueError(
                    f"Invalid source {s!r}. Must be one of: {', '.join(sorted(VALID_SOURCES))}"
                )
        return v

    @property
    def graph_path(self) -> Path:
        return self.data_dir / "graph.json"


class PrismRagSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRISM_",
        # Resolve .env relative to the package root so the server loads
        # correctly regardless of the working directory it is started from.
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Paths ────────────────────────────────────────────────────────
    vault_path: Path = Field(
        default=Path.home() / "Foundation" / "Vault",
        description="Obsidian vault root directory (NimbusVault)",
    )
    data_dir: Path = Field(
        default=Path.cwd() / "data",
        description="Output directory for graph.json, GRAPH_REPORT.md, cache/",
    )

    # ── Embedding backend ────────────────────────────────────────────
    embed_backend: EmbedBackend = Field(
        default="ollama",
        description="Embedding backend: 'ollama' (local, no API key) or 'gemini' (cloud API)",
    )

    # Ollama settings (used when embed_backend='ollama')
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL (PRISM_OLLAMA_HOST; falls back to OLLAMA_HOST env var)",
    )
    ollama_model: str = Field(
        default="bge-m3",
        description="Ollama embedding model — must be the same at index time and query time. "
                    "Supported: bge-m3 (1024), nomic-embed-text (768), qwen3-embedding (1024), "
                    "mxbai-embed-large (1024), all-minilm (384).",
    )
    ollama_embed_dim: int = Field(
        default=0,
        ge=0,
        description="Ollama embedding dimension override. 0 = auto-detect from model name via OLLAMA_MODEL_DIMS table. "
                    "Set explicitly when using a model not in the table.",
    )

    # Gemini settings (used when embed_backend='gemini')
    gemini_api_key: str = Field(default="", description="Gemini API key")
    gemini_embed_model: str = Field(
        default="gemini-embedding-001",
        description="Gemini embedding model. 'gemini-embedding-001' (text, up to 3072 dims, #1 MTEB) "
                    "or 'gemini-embedding-2' (multimodal: text/image/video/audio).",
    )
    privacy_tier: PrivacyTier = Field(
        default="paid",
        description="'paid' requires paid-tier API key (default); 'free' allows free tier (data may be used for training)",
    )
    embed_dimensionality: int = Field(
        default=768,
        ge=64,
        le=4096,
        description="Gemini embedding output dimension via MRL truncation (max 3072 for gemini-embedding-001). "
                    "Ignored for Ollama — use ollama_embed_dim or auto-detection instead.",
    )

    # ── Similarity edges (Pass 3, not used in MVP) ───────────────────
    similarity_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity to create a semantically_similar_to edge",
    )
    top_k_similarity: int = Field(
        default=10,
        ge=1,
        description="Top-K nearest neighbors per node for similarity edge generation",
    )

    # ── Cross-graph bridge settings ─────────────────────────────────
    bridge_similarity_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for cross-graph embedding bridges (higher than within-graph to reduce noise)",
    )
    bridge_top_k: int = Field(
        default=5,
        ge=1,
        description="Top-K cross-graph neighbors per node for embedding bridge generation",
    )

    # ── Leiden clustering ────────────────────────────────────────────
    leiden_resolution: float = Field(
        default=1.0,
        gt=0.0,
        description="Leiden resolution parameter (higher = more, smaller communities)",
    )
    leiden_seed: int = Field(
        default=42,
        description="Random seed for reproducible community detection",
    )
    god_nodes_per_community: int = Field(
        default=5,
        ge=1,
        description="Number of highest-degree nodes to mark as god nodes per community",
    )

    # ── Query-time budget (not used in MVP) ──────────────────────────
    default_query_budget: int = Field(
        default=4000,
        ge=100,
        description="Default token budget for query-time graph traversal",
    )

    # ── Multi-graph / federation ──────────────────────────────────────
    graphs: list[GraphSource] | None = Field(
        default=None,
        description="Multi-graph sources (overrides vault_path/data_dir when set). "
                    "Set via PRISM_GRAPHS as a JSON array.",
    )
    multi_graph_mode: str = Field(
        default="federated",
        description="Federation strategy. Only 'federated' is implemented.",
    )

    # ── Semantic deduplication (atomize_propose) ─────────────────────
    dedup: DedupConfig = Field(
        default_factory=DedupConfig,
        description="Dedup config for atomize_propose semantic similarity check. "
                    "Set via PRISM_DEDUP='{\"threshold\": 0.85}' (JSON) or .env.",
    )

    # ── EdgeClassifier per-model thresholds ──────────────────────────
    classifier_profiles: dict[str, ClassifierProfile] = Field(
        default_factory=lambda: dict(_DEFAULT_PROFILES),
        description="Per-embedding-model classifier thresholds. Look up by model_id.",
    )

    @field_validator("graphs", mode="before")
    @classmethod
    def _parse_graphs_json(cls, v):
        if isinstance(v, str):
            parsed = json.loads(v)
            return [GraphSource(**item) for item in parsed]
        return v

    @field_validator("vault_path", "data_dir", mode="before")
    @classmethod
    def _expand_user(cls, v):
        if isinstance(v, str):
            v = Path(v).expanduser()
        elif isinstance(v, Path):
            v = v.expanduser()
        return v

    @model_validator(mode="after")
    def _ollama_host_fallback(self) -> "PrismRagSettings":
        # If PRISM_OLLAMA_HOST wasn't set but OLLAMA_HOST is, use the latter.
        if self.ollama_host == "http://localhost:11434":
            env_host = os.environ.get("OLLAMA_HOST", "")
            if env_host:
                self.ollama_host = env_host
        return self

    # ── Derived paths ────────────────────────────────────────────────
    @property
    def graph_path(self) -> Path:
        return self.data_dir / "graph.json"

    @property
    def report_path(self) -> Path:
        return self.data_dir / "GRAPH_REPORT.md"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def embedding_cache_path(self) -> Path:
        return self.data_dir / "lance"

    @property
    def embedding_dim(self) -> int:
        """Expected embedding dimension for the configured backend.

        gemini → embed_dimensionality (default 768, max 3072)
        ollama → ollama_embed_dim if non-zero, else OLLAMA_MODEL_DIMS lookup by model name,
                 else 1024 fallback for unknown models.
        """
        if self.embed_backend == "gemini":
            return self.embed_dimensionality
        if self.ollama_embed_dim > 0:
            return self.ollama_embed_dim
        # Strip :tag and path prefix for lookup (e.g. "nomic-embed-text:v1.5" → "nomic-embed-text")
        base = self.ollama_model.split(":")[0].split("/")[-1]
        return OLLAMA_MODEL_DIMS.get(base, 1024)

    @property
    def resolved_graphs(self) -> list[GraphSource]:
        """Return the list of graph sources to load.

        If ``graphs`` is explicitly configured, return it directly.
        Otherwise synthesize a single GraphSource from the legacy
        ``vault_path`` / ``data_dir`` pair with namespace ``"default"``.
        """
        if self.graphs:  # None or empty list → fall back to legacy vault_path/data_dir
            return self.graphs
        return [
            GraphSource(
                namespace="default",
                vault_path=self.vault_path,
                data_dir=self.data_dir,
            )
        ]


def get_classifier_profile(settings: PrismRagSettings, model_id: str) -> ClassifierProfile:
    """Look up classifier profile by model_id; fall back to 'default'."""
    return settings.classifier_profiles.get(model_id) or settings.classifier_profiles["default"]
