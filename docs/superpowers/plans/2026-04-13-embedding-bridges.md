# Embedding Similarity Bridges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist node embeddings in LanceDB during ingest, then use them at serve-time to build cross-graph embedding similarity bridge edges.

**Architecture:** New `EmbeddingStore` wraps LanceDB for per-namespace embedding CRUD. Ingest writes embeddings after computing. `FederatedGraph.build_bridges()` loads stores and adds embedding similarity bridges alongside existing shared-tag bridges.

**Tech Stack:** Python 3.12, LanceDB, PyArrow, pytest

**Spec:** `docs/superpowers/specs/2026-04-13-embedding-bridges-design.md`

---

## File Structure

| File | Role | Action |
|---|---|---|
| `prism_rag/store/embedding_store.py` | LanceDB wrapper — upsert/delete/search/all | Create |
| `prism_rag/config.py` | Add bridge threshold/top-K settings | Modify |
| `prism_rag/ingest/embedder.py` | Persist to LanceDB after computing | Modify |
| `prism_rag/ingest/incremental.py` | Use LanceDB cache instead of re-embed-all | Modify |
| `prism_rag/store/federated.py` | `_build_embedding_bridges()` + load stores | Modify |
| `tests/test_embedding_store.py` | EmbeddingStore unit tests | Create |
| `tests/test_federated.py` | Embedding bridge integration tests | Modify |

---

### Task 1: Install LanceDB + EmbeddingStore

**Files:**
- Create: `prism_rag/store/embedding_store.py`
- Create: `tests/test_embedding_store.py`

- [ ] **Step 1: Install lancedb**

Run: `cd /home/kingy/Foundation/PrismRag && pip install lancedb`

- [ ] **Step 2: Write failing tests**

Create `tests/test_embedding_store.py`:

```python
"""Tests for EmbeddingStore (LanceDB wrapper)."""
from __future__ import annotations

import pytest
from pathlib import Path

from prism_rag.store.embedding_store import EmbeddingStore


class TestEmbeddingStore:
    def test_upsert_and_get(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        vec = [0.1] * 768
        store.upsert("node_a", vec)
        result = store.get("node_a")
        assert result is not None
        assert len(result) == 768
        assert abs(result[0] - 0.1) < 1e-6

    def test_get_missing_returns_none(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        assert store.get("nonexistent") is None

    def test_delete(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        store.upsert("node_a", [0.1] * 768)
        store.delete("node_a")
        assert store.get("node_a") is None

    def test_upsert_overwrites(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        store.upsert("node_a", [0.1] * 768)
        store.upsert("node_a", [0.9] * 768)
        result = store.get("node_a")
        assert abs(result[0] - 0.9) < 1e-6

    def test_all_embeddings(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        store.upsert("a", [0.1] * 768)
        store.upsert("b", [0.2] * 768)
        all_vecs = store.all_embeddings()
        assert len(all_vecs) == 2
        assert "a" in all_vecs
        assert "b" in all_vecs

    def test_search(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        # Insert 3 vectors: a and b are similar, c is different
        store.upsert("a", [1.0, 0.0] + [0.0] * 766)
        store.upsert("b", [0.9, 0.1] + [0.0] * 766)
        store.upsert("c", [0.0, 1.0] + [0.0] * 766)
        results = store.search([1.0, 0.0] + [0.0] * 766, top_k=2)
        # a should be first (exact match), b second
        assert len(results) == 2
        assert results[0][0] == "a"
        assert results[1][0] == "b"

    def test_count(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        assert store.count() == 0
        store.upsert("a", [0.1] * 768)
        assert store.count() == 1
        store.upsert("b", [0.2] * 768)
        assert store.count() == 2

    def test_empty_store_all_embeddings(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        assert store.all_embeddings() == {}

    def test_empty_store_search(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        results = store.search([0.1] * 768, top_k=5)
        assert results == []
```

- [ ] **Step 3: Implement EmbeddingStore**

Create `prism_rag/store/embedding_store.py`:

```python
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

_SCHEMA = pa.schema([
    ("node_id", pa.string()),
    ("embedding", pa.list_(pa.float32(), 768)),
])
_TABLE_NAME = "embeddings"


class EmbeddingStore:
    """Per-namespace embedding store backed by LanceDB."""

    def __init__(self, lance_path: Path) -> None:
        self._path = Path(lance_path)
        self._db = lancedb.connect(str(self._path))
        self._table = self._ensure_table()

    def _ensure_table(self):
        if _TABLE_NAME in self._db.table_names():
            return self._db.open_table(_TABLE_NAME)
        return self._db.create_table(_TABLE_NAME, schema=_SCHEMA)

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
            df = self._table.to_pandas()
            if df.empty:
                return {}
            return {row["node_id"]: list(row["embedding"]) for _, row in df.iterrows()}
        except Exception:
            return {}

    def search(self, vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """ANN search for top-K similar vectors. Returns [(node_id, distance), ...]."""
        try:
            results = self._table.search(vector).limit(top_k).to_list()
            return [(r["node_id"], float(r.get("_distance", 0.0))) for r in results]
        except Exception:
            return []

    def count(self) -> int:
        """Number of stored embeddings."""
        try:
            return self._table.count_rows()
        except Exception:
            return 0
```

- [ ] **Step 4: Run tests**

Run: `/usr/bin/python3 -m pytest tests/test_embedding_store.py -v`

- [ ] **Step 5: Commit**

```bash
git add prism_rag/store/embedding_store.py tests/test_embedding_store.py
git commit -m "feat: EmbeddingStore — LanceDB wrapper for embedding persistence"
```

---

### Task 2: Config additions + ingest persistence

**Files:**
- Modify: `prism_rag/config.py`
- Modify: `prism_rag/ingest/embedder.py`
- Modify: `prism_rag/cli.py`

- [ ] **Step 1: Add bridge config settings**

In `prism_rag/config.py`, after `top_k_similarity`, add:

```python
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
```

- [ ] **Step 2: Add persist_embeddings function to embedder.py**

Add at the bottom of `prism_rag/ingest/embedder.py`:

```python
def persist_embeddings(
    vectors: dict[str, list[float]],
    lance_path: Path,
) -> int:
    """Persist computed embeddings to LanceDB.

    Args:
        vectors: dict mapping node_id → embedding vector.
        lance_path: Path to LanceDB directory.

    Returns:
        Number of embeddings persisted.
    """
    if not vectors:
        return 0

    from prism_rag.store.embedding_store import EmbeddingStore
    store = EmbeddingStore(lance_path)
    for node_id, vec in vectors.items():
        store.upsert(node_id, vec)
    logger.info(f"[embedder] persisted {len(vectors)} embeddings to {lance_path}")
    return len(vectors)
```

Add `from pathlib import Path` to imports if not already present.

- [ ] **Step 3: Call persist in CLI full ingest**

In `prism_rag/cli.py`, after line 111 (`n_new = link_similar_nodes(graph, vectors, settings)`), add:

```python
        # Persist embeddings to LanceDB for serve-time bridge computation
        from prism_rag.ingest.embedder import persist_embeddings
        n_persisted = persist_embeddings(vectors, settings.embedding_cache_path)
        if n_persisted:
            typer.echo(f"   Persisted {n_persisted} embeddings to LanceDB")
```

- [ ] **Step 4: Run full test suite**

Run: `/usr/bin/python3 -m pytest tests/ -q`

- [ ] **Step 5: Commit**

```bash
git add prism_rag/config.py prism_rag/ingest/embedder.py prism_rag/cli.py
git commit -m "feat: persist embeddings to LanceDB during ingest + bridge config"
```

---

### Task 3: Fix incremental ingest to use LanceDB cache

**Files:**
- Modify: `prism_rag/ingest/incremental.py`

- [ ] **Step 1: Replace re-embed-all with LanceDB lookup**

In `prism_rag/ingest/incremental.py`, replace the block at lines 192-215 (the `if vectors:` block inside the embedding section). The new logic:

1. Compute embedding only for the new/changed file (already done — `vectors = compute_embeddings(temp_graph, settings)`)
2. Persist the new embedding to LanceDB
3. Load existing embeddings from LanceDB (instead of re-computing all via API)
4. Generate similarity edges using combined vectors

```python
        if vectors:
            new_vec = vectors.get(doc.id)
            if new_vec:
                # Persist new embedding to LanceDB
                from prism_rag.store.embedding_store import EmbeddingStore
                store = EmbeddingStore(settings.embedding_cache_path)
                store.upsert(doc.id, new_vec)

                # Load existing embeddings from LanceDB (cached, no API calls)
                all_vectors = store.all_embeddings()

                edges_before_sim = graph.edge_count
                link_similar_nodes(graph, all_vectors, settings)
                embed_edges = graph.edge_count - edges_before_sim
```

Also: when removing old node (line 170), add LanceDB cleanup:

```python
        # Remove old embedding from LanceDB
        try:
            from prism_rag.store.embedding_store import EmbeddingStore
            store = EmbeddingStore(settings.embedding_cache_path)
            store.delete(doc.id)
        except Exception:
            pass  # LanceDB may not exist yet
```

- [ ] **Step 2: Remove unused imports**

The old code imported `_cosine_similarity`, `_find_top_k` from `similarity_linker` (line 196), `load_vault` from `vault_loader` (line 203), and created `all_nodes_graph`. These are no longer needed — remove them.

- [ ] **Step 3: Run tests**

Run: `/usr/bin/python3 -m pytest tests/ -q`

- [ ] **Step 4: Commit**

```bash
git add prism_rag/ingest/incremental.py
git commit -m "fix: incremental ingest uses LanceDB cache instead of re-embed-all"
```

---

### Task 4: Embedding bridges in FederatedGraph

**Files:**
- Modify: `prism_rag/store/federated.py`
- Test: `tests/test_federated.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_federated.py`:

```python
class TestEmbeddingBridges:
    def test_embedding_bridges_created(self, tmp_path):
        """Two graphs with similar embeddings should get embedding_similar bridges."""
        from prism_rag.store.embedding_store import EmbeddingStore

        g1 = _make_graph([("a", "A")], [])
        g2 = _make_graph([("x", "X")], [])

        # Create stores with similar vectors
        s1 = EmbeddingStore(tmp_path / "lance1")
        s1.upsert("a", [1.0, 0.0] + [0.0] * 766)
        s2 = EmbeddingStore(tmp_path / "lance2")
        s2.upsert("x", [0.95, 0.05] + [0.0] * 766)  # very similar to a

        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        fg.build_bridges(stores={"ns1": s1, "ns2": s2}, bridge_threshold=0.5, bridge_top_k=5)

        embedding_bridges = [b for b in fg.bridges if b["relation"] == "embedding_similar"]
        assert len(embedding_bridges) >= 1
        bridge = embedding_bridges[0]
        assert {bridge["source_ns"], bridge["target_ns"]} == {"ns1", "ns2"}

    def test_no_embedding_bridges_below_threshold(self, tmp_path):
        """Dissimilar embeddings should NOT create bridges."""
        from prism_rag.store.embedding_store import EmbeddingStore

        g1 = _make_graph([("a", "A")], [])
        g2 = _make_graph([("x", "X")], [])

        s1 = EmbeddingStore(tmp_path / "lance1")
        s1.upsert("a", [1.0, 0.0] + [0.0] * 766)
        s2 = EmbeddingStore(tmp_path / "lance2")
        s2.upsert("x", [0.0, 1.0] + [0.0] * 766)  # orthogonal = similarity ~0

        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        fg.build_bridges(stores={"ns1": s1, "ns2": s2}, bridge_threshold=0.5, bridge_top_k=5)

        embedding_bridges = [b for b in fg.bridges if b["relation"] == "embedding_similar"]
        assert len(embedding_bridges) == 0

    def test_no_stores_graceful(self):
        """No stores provided → only shared-tag bridges, no error."""
        g1 = _make_graph([("tag:py", "python")], [])
        g2 = _make_graph([("tag:py", "python")], [])
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        fg.build_bridges()  # no stores arg
        # Should still have shared-tag bridges
        assert any(b["relation"] == "shared_tag" for b in fg.bridges)

    def test_embedding_bridges_in_unified_view(self, tmp_path):
        """Embedding bridges should appear in unified_view."""
        from prism_rag.store.embedding_store import EmbeddingStore

        g1 = _make_graph([("a", "A")], [])
        g2 = _make_graph([("x", "X")], [])
        s1 = EmbeddingStore(tmp_path / "lance1")
        s1.upsert("a", [1.0, 0.0] + [0.0] * 766)
        s2 = EmbeddingStore(tmp_path / "lance2")
        s2.upsert("x", [0.95, 0.05] + [0.0] * 766)

        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        fg.build_bridges(stores={"ns1": s1, "ns2": s2}, bridge_threshold=0.5, bridge_top_k=5)

        uv = fg.unified_view
        # Should have an edge between ns1::a and ns2::x (or vice versa)
        has_edge = uv.has_edge("ns1::a", "ns2::x") or uv.has_edge("ns2::x", "ns1::a")
        assert has_edge
```

- [ ] **Step 2: Update build_bridges signature and implementation**

In `prism_rag/store/federated.py`, update `build_bridges()`:

```python
def build_bridges(
    self,
    stores: dict[str, "EmbeddingStore"] | None = None,
    bridge_threshold: float = 0.70,
    bridge_top_k: int = 5,
) -> int:
    """Compute cross-graph bridge edges.

    Bridge types:
    1. Shared tags: same tag node ID exists in multiple graphs
    2. Embedding similarity: cross-graph ANN search via LanceDB stores

    Args:
        stores: namespace → EmbeddingStore mapping (optional; skip embedding bridges if None)
        bridge_threshold: minimum cosine similarity for embedding bridges
        bridge_top_k: max neighbors per node for embedding bridge search

    Returns: number of bridge edges created.
    """
    self._bridges.clear()
    self._unified = None

    if self._single:
        return 0

    # 1. Shared tag bridges (existing)
    tag_index: dict[str, list[str]] = {}
    for ns, g in self._graphs.items():
        for node_id, data in g.g.nodes(data=True):
            if data.get("kind") == "tag":
                tag_index.setdefault(node_id, []).append(ns)

    for tag_id, namespaces in tag_index.items():
        if len(namespaces) < 2:
            continue
        for i in range(len(namespaces)):
            for j in range(i + 1, len(namespaces)):
                self._bridges.append({
                    "source_ns": namespaces[i],
                    "source_id": tag_id,
                    "target_ns": namespaces[j],
                    "target_id": tag_id,
                    "relation": "shared_tag",
                    "confidence": "INFERRED",
                    "weight": 0.5,
                })

    # 2. Embedding similarity bridges
    if stores and len(stores) >= 2:
        self._build_embedding_bridges(stores, bridge_threshold, bridge_top_k)

    logger.info(f"[federated] built {len(self._bridges)} bridge edges across {len(self._graphs)} graphs")
    return len(self._bridges)
```

Add `_build_embedding_bridges`:

```python
def _build_embedding_bridges(
    self,
    stores: dict[str, "EmbeddingStore"],
    threshold: float,
    top_k: int,
) -> None:
    """Add embedding similarity bridges between namespace pairs."""
    import math

    ns_list = sorted(stores.keys())
    existing_bridges: set[tuple[str, str, str, str]] = set()

    for i in range(len(ns_list)):
        for j in range(i + 1, len(ns_list)):
            ns_a, ns_b = ns_list[i], ns_list[j]
            store_a, store_b = stores[ns_a], stores[ns_b]
            vecs_a = store_a.all_embeddings()

            for node_id_a, vec_a in vecs_a.items():
                # Verify node still exists in graph
                if node_id_a not in self._graphs[ns_a].g:
                    continue

                results = store_b.search(vec_a, top_k=top_k)
                for node_id_b, distance in results:
                    # Verify node still exists in graph
                    if node_id_b not in self._graphs[ns_b].g:
                        continue

                    # LanceDB returns L2 distance; convert to cosine similarity
                    # For normalized vectors: cosine_sim = 1 - (L2^2 / 2)
                    similarity = 1.0 - (distance / 2.0)

                    if similarity < threshold:
                        continue

                    # Avoid duplicate bridges
                    key = (ns_a, node_id_a, ns_b, node_id_b)
                    rev_key = (ns_b, node_id_b, ns_a, node_id_a)
                    if key in existing_bridges or rev_key in existing_bridges:
                        continue
                    existing_bridges.add(key)

                    self._bridges.append({
                        "source_ns": ns_a,
                        "source_id": node_id_a,
                        "target_ns": ns_b,
                        "target_id": node_id_b,
                        "relation": "embedding_similar",
                        "confidence": "INFERRED",
                        "weight": round(similarity, 4),
                        "source_pass": "embedding",
                    })

    emb_count = sum(1 for b in self._bridges if b["relation"] == "embedding_similar")
    logger.info(f"[federated] embedding bridges: {emb_count}")
```

Add `TYPE_CHECKING` import for `EmbeddingStore`:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from prism_rag.store.embedding_store import EmbeddingStore
```

- [ ] **Step 3: Update FederatedGraph.load() to load stores**

Update the `load()` classmethod:

```python
@classmethod
def load(cls, sources: list, settings=None) -> "FederatedGraph":
    """Load a FederatedGraph from a list of GraphSource configs.
    Skips sources whose graph.json doesn't exist (logs warning).
    Automatically computes bridge edges after loading.
    If settings provided, loads EmbeddingStores for embedding bridges.
    """
    graphs: dict[str, KnowledgeGraph] = {}
    stores: dict[str, "EmbeddingStore"] = {}

    for src in sources:
        gpath = src.graph_path
        if not gpath.exists():
            logger.warning(f"[federated] graph not found: {gpath} (namespace={src.namespace}), skipping")
            continue
        g = KnowledgeGraph.load(gpath)
        graphs[src.namespace] = g
        logger.info(f"[federated] loaded {src.namespace}: {g.node_count} nodes, {g.edge_count} edges")

        # Try loading EmbeddingStore
        lance_path = src.data_dir / "lance"
        if lance_path.exists():
            try:
                from prism_rag.store.embedding_store import EmbeddingStore as ES
                store = ES(lance_path)
                if store.count() > 0:
                    stores[src.namespace] = store
                    logger.info(f"[federated] loaded embeddings for {src.namespace}: {store.count()} vectors")
            except Exception as e:
                logger.warning(f"[federated] failed to load embeddings for {src.namespace}: {e}")

    fg = cls(graphs)

    bridge_threshold = 0.70
    bridge_top_k = 5
    if settings:
        bridge_threshold = getattr(settings, "bridge_similarity_threshold", 0.70)
        bridge_top_k = getattr(settings, "bridge_top_k", 5)

    fg.build_bridges(stores=stores or None, bridge_threshold=bridge_threshold, bridge_top_k=bridge_top_k)
    return fg
```

- [ ] **Step 4: Run tests**

Run: `/usr/bin/python3 -m pytest tests/test_federated.py tests/test_embedding_store.py -v`

- [ ] **Step 5: Run full suite**

Run: `/usr/bin/python3 -m pytest tests/ -q`

- [ ] **Step 6: Commit**

```bash
git add prism_rag/store/federated.py tests/test_federated.py
git commit -m "feat: embedding similarity bridges via LanceDB in FederatedGraph"
```

---

### Task 5: Update MCP server to pass settings to FederatedGraph.load()

**Files:**
- Modify: `prism_rag/mcp_server/server.py`

- [ ] **Step 1: Update _ensure_federated to pass settings**

In `server.py`, find `_ensure_federated()` and update the `FederatedGraph.load()` call to pass settings:

```python
fg = FederatedGraph.load(settings.resolved_graphs, settings=settings)
```

- [ ] **Step 2: Run full test suite**

Run: `/usr/bin/python3 -m pytest tests/ -q`

- [ ] **Step 3: Commit**

```bash
git add prism_rag/mcp_server/server.py
git commit -m "feat: MCP server passes settings to FederatedGraph.load for embedding bridges"
```

---

### Task 6: Full E2E test

**Files:**
- Test: `tests/test_federated.py`

- [ ] **Step 1: Add E2E test**

Add to `tests/test_federated.py`:

```python
class TestEmbeddingBridgesE2E:
    def test_embedding_bridge_enables_cross_namespace_bfs(self, tmp_path):
        """Two graphs with NO shared tags but similar embeddings → BFS crosses via embedding bridge."""
        from prism_rag.store.embedding_store import EmbeddingStore

        # Two graphs with NO shared tags (shared-tag bridges won't help)
        g1 = _make_graph([("ml-intro", "ML Introduction")], [])
        g2 = _make_graph([("dl-guide", "Deep Learning Guide")], [])

        # But their embeddings are similar (both about ML)
        s1 = EmbeddingStore(tmp_path / "lance1")
        s1.upsert("ml-intro", [0.9, 0.1] + [0.0] * 766)
        s2 = EmbeddingStore(tmp_path / "lance2")
        s2.upsert("dl-guide", [0.85, 0.15] + [0.0] * 766)

        fg = FederatedGraph({"research": g1, "tutorials": g2})
        fg.build_bridges(stores={"research": s1, "tutorials": s2}, bridge_threshold=0.5, bridge_top_k=5)

        # Should have embedding bridge (no shared tags)
        assert any(b["relation"] == "embedding_similar" for b in fg.bridges)
        assert not any(b["relation"] == "shared_tag" for b in fg.bridges)

        # BFS should cross the embedding bridge
        results = federated_bfs(fg, "research", "ml-intro", budget=5000)
        namespaces = {r["namespace"] for r in results}
        assert "research" in namespaces
        assert "tutorials" in namespaces, "BFS should cross embedding bridge to tutorials"

    def test_embedding_bridge_trace_path(self, tmp_path):
        """trace_path should find cross-namespace path via embedding bridge."""
        import json
        from prism_rag.mcp_server import server as mcp_mod
        from prism_rag.store.embedding_store import EmbeddingStore

        g1 = _make_graph([("a", "Alpha")], [])
        g2 = _make_graph([("b", "Beta")], [])

        s1 = EmbeddingStore(tmp_path / "lance1")
        s1.upsert("a", [0.9, 0.1] + [0.0] * 766)
        s2 = EmbeddingStore(tmp_path / "lance2")
        s2.upsert("b", [0.85, 0.15] + [0.0] * 766)

        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        fg.build_bridges(stores={"ns1": s1, "ns2": s2}, bridge_threshold=0.5, bridge_top_k=5)

        mcp_mod._federated = fg
        result = json.loads(mcp_mod.trace_path("ns1::a", "ns2::b", max_length=10))
        assert "error" not in result, f"trace_path failed: {result}"
        assert result["path_length"] >= 1
```

- [ ] **Step 2: Run all tests**

Run: `/usr/bin/python3 -m pytest tests/ -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_federated.py
git commit -m "test: embedding bridges E2E — BFS + trace_path cross namespace via embedding similarity"
```
