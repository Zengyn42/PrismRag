# Embedding Similarity Bridges Design

> Date: 2026-04-13
> Status: Approved

## Goal

Persist node embeddings in LanceDB during ingest, then use them at serve-time to build cross-graph embedding similarity bridge edges in the federated multi-graph.

## Current State

- Embeddings computed at ingest-time (Gemini Embedding 2, 768-dim) but **discarded** after generating within-graph `semantically_similar_to` edges
- No persistence layer — incremental ingest re-computes ALL embeddings every time (expensive, ~0.5s per node API call)
- Config has `embedding_cache_path = data_dir / "lance"` (unused)
- `pyproject.toml` declares `lancedb>=0.15.0` (not installed)
- Cross-graph bridges currently only use shared tags

## Architecture

### 1. EmbeddingStore — new module `prism_rag/store/embedding_store.py`

Lightweight LanceDB wrapper for per-namespace embedding persistence.

```python
class EmbeddingStore:
    def __init__(self, lance_path: Path)
    def upsert(self, node_id: str, embedding: list[float]) -> None
    def delete(self, node_id: str) -> None
    def get(self, node_id: str) -> list[float] | None
    def all_embeddings(self) -> dict[str, list[float]]
    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]
    def count(self) -> int
```

- LanceDB table schema: `{node_id: str, embedding: vector[768]}`
- Storage: `data_dir/lance/` per namespace (config's `embedding_cache_path`)
- Table name: `"embeddings"`

### 2. Ingest pipeline changes

**Full ingest** (Pass 3a in `embedder.py`):
- After `compute_embeddings()` returns vectors, persist them to LanceDB via `EmbeddingStore.upsert()`

**Incremental ingest** (`incremental.py`):
- On node delete: `store.delete(node_id)`
- On node add/update: compute embedding for the single new node, `store.upsert()`
- For similarity edge generation: read existing embeddings from LanceDB (`store.all_embeddings()`) instead of re-computing all via API
- Eliminates the current "re-embed ALL existing notes" hack

### 3. FederatedGraph.build_bridges() extension

After existing shared-tag bridges, call `_build_embedding_bridges()`:

```python
def _build_embedding_bridges(self, stores: dict[str, EmbeddingStore], threshold: float, top_k: int) -> int
```

Algorithm:
- For each pair of namespaces (A, B):
  - For each embedding in A: search B's LanceDB for top-K similar vectors
  - For each result >= threshold: create bridge edge
- Bridge edge attributes: `relation="embedding_similar"`, `confidence="INFERRED"`, `weight=cosine_similarity`, `source_pass="embedding"`
- Uses LanceDB native ANN search (not Python cosine loop)

### 4. FederatedGraph.load() changes

After loading graphs, attempt to load EmbeddingStores:
- For each GraphSource: check if `data_dir/lance/` exists
- If exists: open EmbeddingStore, pass to `build_bridges()`
- If missing: skip embedding bridges, log warning ("embeddings not available, run ingest to enable embedding bridges")
- Shared-tag bridges always work regardless

### 5. Config additions

- `bridge_similarity_threshold: float = 0.70` — cross-graph threshold (higher than within-graph 0.65 to reduce noise)
- `bridge_top_k: int = 5` — cross-graph top-K per node (lower than within-graph 10)

### 6. Graceful degradation

| Scenario | Behavior |
|---|---|
| LanceDB not installed | Import error caught, skip embedding bridges, log warning |
| lance/ dir missing | Skip embedding bridges for that namespace |
| lance/ partially stale | Use what's available, stale entries won't match graph nodes (harmless) |
| Single-graph mode | No cross-graph bridges needed, skip entirely |

## File Changes

| File | Action | Description |
|---|---|---|
| `prism_rag/store/embedding_store.py` | Create | LanceDB wrapper |
| `prism_rag/ingest/embedder.py` | Modify | Persist to LanceDB after computing |
| `prism_rag/ingest/incremental.py` | Modify | Use LanceDB for cached embeddings |
| `prism_rag/store/federated.py` | Modify | `_build_embedding_bridges()` + load stores |
| `prism_rag/config.py` | Modify | Add bridge threshold/top-K settings |
| `tests/test_embedding_store.py` | Create | EmbeddingStore unit tests |
| `tests/test_federated.py` | Modify | Embedding bridge tests |

## Not Changing

- `similarity_linker.py` — within-graph similarity logic unchanged
- `KnowledgeGraph` / `Node` dataclass — no embedding field on nodes
- BFS/DFS/trace_path — already work with any bridge type via unified_view
- `unified_view` — bridge edges added the same way regardless of source
