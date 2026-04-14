# Cross-Namespace Traversal Design

> Date: 2026-04-13
> Status: Approved

## Goal

Enable trace_path, BFS, and DFS to traverse across namespace boundaries via bridge edges, completing the federated multi-graph query story.

## Current State

- `FederatedGraph` loads multiple `KnowledgeGraph` instances and computes shared-tag bridge edges at serve-time
- Bridge edges are stored in `fg.bridges` (list of dicts) but NOT added to any NetworkX graph
- `trace_path` hard-rejects cross-namespace queries: `"Cross-namespace trace not yet supported"`
- `federated_bfs/dfs` delegate to single-namespace `bfs_traverse/dfs_traverse` — no bridge crossing

## Architecture

### 1. `FederatedGraph.unified_view` — lazy-built unified NetworkX graph

A new `@property` on `FederatedGraph` that builds and caches a single `nx.DiGraph` containing:

- All nodes from all namespaces, with IDs prefixed: `namespace::node_id`
- All original edges, with prefixed source/target IDs
- All bridge edges from `self._bridges`
- Node attributes copied from originals, plus `namespace` field added

Caching:
- `_unified: nx.DiGraph | None` — built on first access
- `build_bridges()` sets `_unified = None` to invalidate cache
- Rebuilt lazily on next `unified_view` access

Single-graph optimization:
- When `self._single` is True, return the original graph's `g` directly (no prefixing, no copy)

### 2. `trace_path` — cross-namespace pathfinding

Remove the `src_ns != tgt_ns` error branch. Replace single-graph pathfinding with unified view:

```
unified = fg.unified_view.to_undirected()
src_qid = f"{src_ns}::{src_id}"
tgt_qid = f"{tgt_ns}::{tgt_id}"
path = nx.shortest_path(unified, source=src_qid, target=tgt_qid)
```

Path result changes:
- Each step includes `namespace` (parsed from qualified node_id)
- Edge metadata looked up from unified view (covers both original edges and bridge edges)
- `_node_summary` extended to accept `FederatedGraph` + qualified_id (helper extracts namespace, looks up correct `KnowledgeGraph`)
- Top-level `namespace` field in result replaced with per-step namespace

### 3. BFS/DFS — traverse on unified view

`federated_bfs()` and `federated_dfs()` rewritten to operate on `fg.unified_view`:

- Entry node addressed as `namespace::entry_id`
- BFS/DFS traverses naturally across bridge edges (they are normal edges in unified view)
- Each result node tagged with `namespace` (parsed from qualified node_id)
- Token budget is unified (shared across namespaces, first-come-first-served)
- `scope` parameter: when non-empty, only traverse within the specified namespace (filter out cross-namespace neighbors)

Implementation approach: new `_bfs_unified()` / `_dfs_unified()` functions that work on `nx.DiGraph` directly (the unified view), rather than `KnowledgeGraph`. The existing `bfs_traverse()` / `dfs_traverse()` stay unchanged for backward compatibility (called when `fg.is_single` for zero-overhead single-graph path).

### 4. Not Changing

- `KnowledgeGraph` class — untouched
- `build_bridges()` — still shared-tag only (embedding bridges is a separate follow-up)
- `search_knowledge` MCP tool interface — `scope` parameter already exists
- `explain_node`, `write_note`, `read_note` — unaffected
- Ingest pipeline — no changes

## File Changes

| File | Action | Description |
|---|---|---|
| `prism_rag/store/federated.py` | Modify | Add `unified_view` property, cache invalidation in `build_bridges()` |
| `prism_rag/retrieve/bfs.py` | Modify | Add `_bfs_unified()`, rewrite `federated_bfs()` to use unified view (with scope filtering) |
| `prism_rag/retrieve/dfs.py` | Modify | Add `_dfs_unified()`, rewrite `federated_dfs()` to use unified view (with scope filtering) |
| `prism_rag/mcp_server/server.py` | Modify | `trace_path`: remove cross-namespace error, use unified view; `_node_summary` extended for federated |
| `tests/test_federated.py` | Modify | Add tests: cross-namespace trace_path, BFS/DFS bridge crossing, scope filtering, single-graph backward compat |

## Edge Cases

- **No path across namespaces**: `nx.shortest_path` raises `NetworkXNoPath` — same error handling as single-namespace (returns `"error": "No path found"`)
- **Single-graph mode**: `unified_view` returns original graph directly, no prefixing overhead. All existing behavior preserved.
- **scope filter + bridge edge**: When `scope="nimbus"`, BFS/DFS skip neighbors whose namespace != "nimbus", even if a bridge edge leads there
- **Bridge edge weight**: Currently 0.5 for shared_tag bridges (lower than typical content edges at 0.7-1.0). BFS/DFS naturally deprioritize bridge crossings due to weight-sorted neighbor exploration.
