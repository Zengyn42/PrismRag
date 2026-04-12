"""PrismRag runtime settings.

All fields overridable via environment variables with PRISM_ prefix,
or via a .env file in the current working directory.

Example:
    export PRISM_VAULT_PATH=~/Foundation/Vault
    export PRISM_GEMINI_API_KEY=...
    export PRISM_PRIVACY_TIER=paid
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PrivacyTier = Literal["paid", "free"]


class PrismRagSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRISM_",
        env_file=".env",
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

    # ── Gemini (for embedding and vision in Pass 2/3) ────────────────
    gemini_api_key: str = Field(default="", description="Gemini API key")
    privacy_tier: PrivacyTier = Field(
        default="paid",
        description="'paid' requires paid-tier API key (default); 'free' allows free tier (data may be used for training)",
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

    @field_validator("vault_path", "data_dir", mode="before")
    @classmethod
    def _expand_user(cls, v):
        if isinstance(v, str):
            v = Path(v).expanduser()
        elif isinstance(v, Path):
            v = v.expanduser()
        return v

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
