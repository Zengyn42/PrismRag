# PrismRag Federated Graph Design

> Date: 2026-04-13
> Status: Approved (architecture decision, not yet implemented)

## Summary

Multi-vault, multi-graph federation for PrismRag. Each vault produces an independent
graph; MCP server instances are configured to load one or more graphs at runtime.
Cross-graph relationships (bridge edges) are computed at serve-time, not ingest-time.

## Core Principle

**Ingest is per-vault. Federation is per-MCP-instance.**

Ingest layer knows nothing about federation. The serve layer handles multi-graph
loading, bridge edge construction, and cross-namespace query routing.

## Architecture

```
                    ingest (per-vault, independent)
                    ┌──────────┐     ┌──────────┐     ┌──────────┐
                    │ Vault-A  │     │ Vault-B  │     │ Vault-C  │
                    └────┬─────┘     └────┬─────┘     └────┬─────┘
                         │                │                │
                         ▼                ▼                ▼
                    graph-A.json     graph-B.json     graph-C.json
                    (independent     (independent     (independent
                     Leiden)          Leiden)           Leiden)

                    serve (configurable, per-MCP-instance)
                    ┌─────────────────────────────────────────────┐
MCP-1 (Agent X)     │ load: [A]           → searches A only       │
                    ├─────────────────────────────────────────────┤
MCP-2 (Agent Y)     │ load: [A, B, C]     → searches all          │
                    │                       + cross-graph bridges  │
                    ├─────────────────────────────────────────────┤
MCP-3 (Agent Z)     │ load: [B]           → searches B only       │
                    └─────────────────────────────────────────────┘
```

## Three-Layer Separation

| Layer | Responsibility | Granularity |
|---|---|---|
| **Ingest** | vault → AST → embed → Leiden → graph.json | Per-vault, independent, no knowledge of federation |
| **Store** | graph.json persistence | 1 vault = 1 graph.json |
| **Serve** | Load N graphs at startup, build bridges at runtime | Per-MCP-instance, configured via `graphs` list |

## Multi-Graph Load Modes

When loading multiple graphs, the MCP instance can operate in two modes:

| Mode | Behavior | Use Case |
|---|---|---|
| **`federated`** (default) | Each graph stays independent; bridge edges computed at runtime; Leiden communities per-graph; results prefixed with namespace | Vaults with different domains (tech vault + personal vault); need access control per-vault |
| **`merged`** | All graphs merged into one unified graph at startup; Leiden re-clustered on the merged graph; single community structure; no namespace prefixes | Vaults with highly related content (design docs vault + implementation vault); want cross-vault communities |

```
federated mode:                          merged mode:
┌─────────┐  bridge  ┌─────────┐         ┌──────────────────────┐
│ Graph-A  │◄──────►│ Graph-B  │         │   Unified Graph      │
│ (Leiden) │         │ (Leiden) │         │   (re-Leiden on all) │
└─────────┘         └─────────┘         └──────────────────────┘
  independent         independent          single community set
  communities         communities          nodes prefixed A::, B::
```

### Federated vs Merged trade-offs

| Dimension | Federated | Merged |
|---|---|---|
| Leiden communities | Per-graph (independent) | Unified (cross-vault communities emerge) |
| Startup cost | Cheap (load + bridge) | Expensive (load + merge + re-Leiden) |
| Incremental update | Re-ingest one vault, rebuild bridges only | Re-ingest one vault, re-Leiden entire merged graph |
| Access control | Natural (load different graphs per MCP) | All-or-nothing (merged graph contains everything) |
| Cross-vault discovery | Via bridge edges (explicit) | Native (same graph, same traversal) |
| Node ID conflicts | No conflict (namespaced) | Prefix-based dedup (A::note, B::note) |

## Configuration

```python
# config.py
class GraphSource(BaseModel):
    namespace: str          # "nimbus", "work", "personal"
    vault_path: Path        # vault root directory
    data_dir: Path          # graph.json + cache location
    writable: bool = False  # only one MCP instance should write per vault

class PrismRagSettings(BaseSettings):
    graphs: list[GraphSource]   # replaces single vault_path
    multi_graph_mode: str = "federated"  # "federated" or "merged"
```

### Example: single graph (backward compatible)

```bash
PRISM_GRAPHS='[{"namespace":"nimbus","vault_path":"~/Vault","data_dir":"~/PrismRag/data/nimbus"}]'
```

### Example: federated (multiple graphs)

```bash
PRISM_GRAPHS='[
  {"namespace":"nimbus","vault_path":"~/Vault","data_dir":"~/PrismRag/data/nimbus","writable":true},
  {"namespace":"work","vault_path":"~/WorkVault","data_dir":"~/PrismRag/data/work","writable":true},
  {"namespace":"personal","vault_path":"~/Personal","data_dir":"~/PrismRag/data/personal"}
]'
```

## Node Addressing

### Single-graph mode (one graph loaded)

Node IDs have no prefix. Fully backward compatible with current behavior.

```
search_knowledge("session management")
explain_node("session_mode_design")
```

### Federated mode (multiple graphs loaded)

Node IDs are prefixed with namespace. Tools accept optional `scope` parameter.

```
search_knowledge("session management")
→ searches all loaded graphs, results prefixed: nimbus::session_mode_design, work::meeting_notes_0312

search_knowledge("session management", scope="nimbus")
→ searches only the specified namespace

explain_node("nimbus::session_mode_design")
→ cross-graph addressing

trace_path("nimbus::node_a", "work::node_b")
→ cross-graph path traversal via bridge edges
```

## Bridge Edges

Cross-graph edges are computed at **serve-time** when the MCP loads multiple graphs.
They are **not persisted** — they depend on which graphs are loaded.

```python
class FederatedGraph:
    """Runtime-only federation layer over multiple KnowledgeGraph instances."""

    graphs: dict[str, KnowledgeGraph]   # namespace → graph
    bridge: nx.DiGraph                   # cross-namespace edges only

    def build_bridges(self):
        # 1. Shared tags
        #    nimbus has #architecture, work has #architecture
        #    → bridge: (nimbus::#architecture) --same_tag-- (work::#architecture)
        #
        # 2. Embedding similarity (cross-graph)
        #    Compare node embeddings between graphs, top-K per graph pair
        #    → bridge edge with confidence=INFERRED, source_pass="embedding"
        #
        # 3. Explicit cross-references
        #    A note in vault-A mentions "[[work/some_note]]"
        #    → bridge edge with confidence=EXTRACTED, source_pass="ast"
```

### Why not persist bridges?

Bridge edges depend on which graphs are loaded. MCP-1 loads [A] — no bridges.
MCP-2 loads [A,B,C] — has A↔B, A↔C, B↔C bridges. Storing bridges in any
single graph.json would couple that graph to other vaults, violating per-vault
independence.

## Query Path (federated mode)

```
query("session management", scope=None)
    │
    ├── 1. Entry point resolution
    │      Search all loaded graphs for matching nodes
    │      → [nimbus::session_mode_design, work::standup_0312]
    │
    ├── 2. BFS/DFS traversal
    │      Start from each entry, traverse within own graph
    │      When hitting a bridge edge: cross into other graph (if budget allows)
    │
    ├── 3. Token budget
    │      Unified budget across all graphs
    │
    └── 4. Return results with namespace prefixes
```

## Write Operations

```
write_note("nimbus::path/to/doc.md", content, ...)
    │
    ├── Parse namespace → resolve to nimbus vault_path
    ├── Check writable == true (reject if false)
    ├── Write file to vault
    ├── Incremental ingest into nimbus graph
    └── Rebuild bridge edges (new node may create cross-graph relationships)
```

Write safety: `writable: bool` per GraphSource. Only one MCP instance should
have `writable: true` for a given vault to prevent concurrent write conflicts.
CAS (content-addressable storage) provides additional conflict detection.

## Single-Graph Backward Compatibility

When `graphs` contains exactly one entry, behavior is identical to current
single-vault mode:
- Node IDs have no namespace prefix
- No bridge edge computation
- All MCP tools work without `scope` parameter
- Zero breaking changes

## Data Directory Layout

```
~/Foundation/PrismRag/
├── data/
│   ├── nimbus/               ← per-vault, independent
│   │   ├── graph.json
│   │   ├── GRAPH_REPORT.md
│   │   ├── graph.html
│   │   └── cache/
│   │       └── file_hashes.json
│   ├── work/
│   │   ├── graph.json
│   │   └── ...
│   └── personal/
│       ├── graph.json
│       └── ...
└── prism_rag/                ← source code
```

## Code Changes Required

| Module | Change | Scope |
|---|---|---|
| `config.py` | `vault_path: Path` → `graphs: list[GraphSource]` | Medium |
| `store/graph.py` | New `FederatedGraph` class (~100 lines) | New file or extension |
| `retrieve/entry.py` | Support `namespace::node_id` addressing | Small |
| `retrieve/bfs.py` / `dfs.py` | Recognize bridge edges, cross graph boundaries | Small |
| `mcp_server/server.py` | Add `scope` param to all tools; load multiple graphs + build_bridges at startup | Medium |
| `cli.py` | `ingest` accepts `--namespace`; `query` accepts `--scope` | Small |
| **`ingest/`** | **No changes** — ingest remains per-vault | None |

## Design Decisions

### Why default to federated?

Federated is the default because it preserves per-vault independence:
- Changing vault A does not trigger re-clustering of vault B
- Different agents can load different vault subsets (access control)
- Startup is cheap (no re-Leiden)

Merged mode is available for the case where vaults are tightly coupled and
cross-vault communities are valuable (e.g., a design-docs vault and an
implementation vault that should cluster together). The user chooses via
`multi_graph_mode` config.

### Why bridge at serve-time (not ingest-time)?

- Bridge depends on which graphs are co-loaded — not a property of any single vault
- Avoids coupling between vault data directories
- Bridges can be rebuilt cheaply (tag matching is O(n), embedding similarity is
  already cached from ingest)

### Why namespace prefix (not separate tool endpoints)?

Alternative: `search_nimbus(query)`, `search_work(query)`, `search_all(query)`.
Rejected because:
- Tool count explodes with N vaults
- Agent needs to know vault names at prompt-design time
- Adding a vault requires updating all agent configs

With `scope` parameter: one `search_knowledge(query, scope=...)` tool handles all cases.
Agent discovers available namespaces via `list_namespaces()` tool.

## Future: Additional MCP Tools for Federation

| Tool | Description |
|---|---|
| `list_namespaces()` | Returns loaded graph namespaces with stats (node count, edge count, communities) |
| `search_knowledge(query, scope?, budget?)` | Existing tool + optional scope filter |
| `cross_graph_bridges(namespace_a, namespace_b)` | Show bridge edges between two specific graphs |

## See Also

- `docs/ARCHITECTURE.md` — single-graph architecture (current implementation)
- `prism_rag/store/graph.py` — KnowledgeGraph schema (Node, Edge, Community)
- `prism_rag/config.py` — current single-vault settings
