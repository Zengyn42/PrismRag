"""LanceDB-backed embedding persistence for PrismRag.

Each namespace gets its own EmbeddingStore (backed by a LanceDB directory).
Embeddings are written during ingest and read at serve-time for cross-graph
bridge computation.

The store is a cache — it can be rebuilt by re-running ingest.
"""
from __future__ import annotations

import logging
from pathlib import Path

import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)

_TABLE_NAME = "embeddings"


def _make_schema(dim: int) -> pa.Schema:
    return pa.schema([
        ("node_id", pa.string()),
        ("embedding", pa.list_(pa.float32(), dim)),
    ])


def _detect_dim(table) -> int | None:
    """Read the embedding dimension from an existing LanceDB table schema."""
    try:
        field = table.schema.field("embedding")
        # pa.list_(pa.float32(), N) → field.type.list_size
        return field.type.list_size
    except Exception:
        return None


class EmbeddingStore:
    """Per-namespace embedding store backed by LanceDB.

    Args:
        lance_path: Directory for the LanceDB database.
        dim: Expected embedding dimension. If the existing table has a
             different dimension, it is dropped and recreated automatically.
             Default 768 (Gemini); use 1024 for bge-m3 / Ollama.
    """

    def __init__(self, lance_path: Path, dim: int = 768) -> None:
        self._path = Path(lance_path)
        self._dim = dim
        self._db = lancedb.connect(str(self._path))
        self._table = self._ensure_table()

    def _ensure_table(self):
        existing = self._db.list_tables()
        # LanceDB >=0.20 returns ListTablesResponse with .tables attribute
        table_names = getattr(existing, "tables", existing)
        if _TABLE_NAME in table_names:
            table = self._db.open_table(_TABLE_NAME)
            existing_dim = _detect_dim(table)
            if existing_dim is not None and existing_dim != self._dim:
                # Auto-adapt to the stored dimension rather than silently wiping data.
                logger.info(
                    f"[embedding_store] adapting dim {self._dim} → {existing_dim} "
                    f"to match existing table"
                )
                self._dim = existing_dim
            return table
        return self._db.create_table(_TABLE_NAME, schema=_make_schema(self._dim))

    def upsert(self, node_id: str, embedding: list[float]) -> None:
        """Insert or update an embedding for a node."""
        self.delete(node_id)
        self._table.add([{"node_id": node_id, "embedding": embedding}])

    def delete(self, node_id: str) -> None:
        """Remove a node's embedding."""
        try:
            self._table.delete(f'node_id = "{node_id}"')
        except Exception:
            pass  # node may not exist

    def get(self, node_id: str) -> list[float] | None:
        """Get a single node's embedding, or None if not found."""
        try:
            results = self._table.search().where(f'node_id = "{node_id}"').limit(1).to_list()
            if results:
                return list(results[0]["embedding"])
        except Exception:
            pass
        return None

    def all_embeddings(self) -> dict[str, list[float]]:
        """Load all embeddings as {node_id: vector} dict."""
        try:
            rows = self._table.to_arrow().to_pylist()
            return {r["node_id"]: list(r["embedding"]) for r in rows}
        except Exception:
            return {}

    def search(self, vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """ANN search for top-K similar vectors. Returns [(node_id, distance), ...]."""
        try:
            results = self._table.search(vector).limit(top_k).to_list()
            return [(r["node_id"], float(r.get("_distance", 0.0))) for r in results]
        except Exception:
            return []

    def nearest(self, vector: list[float], top_k: int = 10) -> list[str]:
        """Return top-K node IDs ordered by similarity (closest first).

        Convenience wrapper used by hybrid_search — returns only IDs, not distances.
        """
        return [nid for nid, _ in self.search(vector, top_k=top_k)]

    def count(self) -> int:
        """Number of stored embeddings."""
        try:
            return self._table.count_rows()
        except Exception:
            return 0
