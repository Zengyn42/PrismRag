# Cross-Namespace Traversal Implementation Plan

> **[STALE — 2026-04-30]** 跨 namespace BFS/DFS/trace_path 已在 retrieve/bfs.py, dfs.py 中实现（federated_bfs, federated_dfs）。本计划 34 个任务均未勾选但代码已完成，请勿作为执行依据。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable trace_path, BFS, and DFS to traverse across namespace boundaries via bridge edges in the federated multi-graph.

**Architecture:** `FederatedGraph` gets a lazy-built `unified_view` property (a single `nx.DiGraph` merging all namespaces with prefixed node IDs + bridge edges). trace_path, BFS, and DFS all operate on this unified graph for cross-namespace traversal.

**Tech Stack:** Python 3.12, NetworkX, pytest

**Spec:** `docs/superpowers/specs/2026-04-13-cross-namespace-traversal-design.md`

---

## File Structure

| File | Role | Action |
|---|---|---|
| `prism_rag/store/federated.py` | `FederatedGraph.unified_view` property + cache invalidation | Modify |
| `prism_rag/retrieve/bfs.py` | `federated_bfs()` uses unified view, crosses bridges | Modify |
| `prism_rag/retrieve/dfs.py` | `federated_dfs()` uses unified view, crosses bridges | Modify |
| `prism_rag/mcp_server/server.py` | `trace_path` cross-namespace support + `_node_summary` federated | Modify |
| `tests/test_federated.py` | New test classes for all cross-namespace behavior | Modify |

---

### Task 1: FederatedGraph.unified_view property

**Files:**
- Modify: `prism_rag/store/federated.py`
- Test: `tests/test_federated.py`

- [ ] **Step 1: Write failing tests for unified_view**

Add to `tests/test_federated.py`:

```python
class TestUnifiedView:
    def test_single_graph_returns_original(self):
        """Single-graph mode returns the original nx.DiGraph directly (no prefixing)."""
        g = _make_graph([("a", "A"), ("b", "B")], [("a", "b", "links_to")])
        fg = FederatedGraph({"ns1": g})
        uv = fg.unified_view
        assert "a" in uv
        assert "b" in uv
        assert uv is g.g  # same object, zero-copy

    def test_multi_graph_prefixed_nodes(self):
        """Multi-graph mode prefixes all node IDs with namespace::."""
        g1 = _make_graph([("a", "A")])
        g2 = _make_graph([("x", "X")])
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        uv = fg.unified_view
        assert "ns1::a" in uv
        assert "ns2::x" in uv
        assert "a" not in uv  # bare IDs should not exist

    def test_multi_graph_edges_preserved(self):
        """Original edges are carried over with prefixed IDs."""
        g1 = _make_graph([("a", "A"), ("b", "B")], [("a", "b", "links_to")])
        fg = FederatedGraph({"ns1": g1})
        fg._single = False  # force multi-graph path for testing
        uv = fg.unified_view
        assert uv.has_edge("ns1::a", "ns1::b")
        edge = uv.edges["ns1::a", "ns1::b"]
        assert edge["relation"] == "links_to"

    def test_multi_graph_bridge_edges_included(self):
        """Bridge edges appear as real edges in unified view."""
        g1 = _make_graph([("tag:py", "python")], [])
        g2 = _make_graph([("tag:py", "python")], [])
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        fg.build_bridges()
        uv = fg.unified_view
        assert uv.has_edge("ns1::tag:py", "ns2::tag:py") or uv.has_edge("ns2::tag:py", "ns1::tag:py")

    def test_node_namespace_attribute(self):
        """Nodes in unified view carry a 'namespace' attribute."""
        g1 = _make_graph([("a", "A")])
        g2 = _make_graph([("x", "X")])
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        uv = fg.unified_view
        assert uv.nodes["ns1::a"]["namespace"] == "ns1"
        assert uv.nodes["ns2::x"]["namespace"] == "ns2"

    def test_cache_invalidation_on_rebuild_bridges(self):
        """build_bridges() invalidates the unified_view cache."""
        g1 = _make_graph([("tag:py", "python")])
        g2 = _make_graph([("tag:py", "python")])
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        fg.build_bridges()
        uv1 = fg.unified_view
        fg.build_bridges()  # should invalidate
        uv2 = fg.unified_view
        assert uv1 is not uv2  # rebuilt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/kingy/Foundation/PrismRag && python -m pytest tests/test_federated.py::TestUnifiedView -v`
Expected: FAIL — `FederatedGraph` has no `unified_view`

- [ ] **Step 3: Implement unified_view**

In `prism_rag/store/federated.py`, add to `__init__`:

```python
self._unified: nx.DiGraph | None = None
```

Add the property after `build_bridges()`:

```python
@property
def unified_view(self) -> nx.DiGraph:
    """Lazy-built unified graph with namespace-prefixed node IDs + bridge edges.
    Single-graph mode returns the original graph directly (zero-copy).
    """
    if self._single:
        return next(iter(self._graphs.values())).g

    if self._unified is not None:
        return self._unified

    import networkx as nx
    unified = nx.DiGraph()

    for ns, kg in self._graphs.items():
        for node_id, data in kg.g.nodes(data=True):
            qid = f"{ns}::{node_id}"
            unified.add_node(qid, **data, namespace=ns)
        for src, tgt, data in kg.g.edges(data=True):
            unified.add_edge(f"{ns}::{src}", f"{ns}::{tgt}", **data)

    for bridge in self._bridges:
        src_qid = f"{bridge['source_ns']}::{bridge['source_id']}"
        tgt_qid = f"{bridge['target_ns']}::{bridge['target_id']}"
        unified.add_edge(src_qid, tgt_qid,
                         relation=bridge["relation"],
                         confidence=bridge["confidence"],
                         weight=bridge.get("weight", 0.5))
        unified.add_edge(tgt_qid, src_qid,
                         relation=bridge["relation"],
                         confidence=bridge["confidence"],
                         weight=bridge.get("weight", 0.5))

    self._unified = unified
    return self._unified
```

In `build_bridges()`, add cache invalidation after `self._bridges.clear()`:

```python
self._unified = None
```

Also add `import networkx as nx` at module top level (next to existing imports).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kingy/Foundation/PrismRag && python -m pytest tests/test_federated.py::TestUnifiedView -v`
Expected: 7 PASSED

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -q`
Expected: All pass (no regressions)

- [ ] **Step 6: Commit**

```bash
git add prism_rag/store/federated.py tests/test_federated.py
git commit -m "feat: FederatedGraph.unified_view — merged nx.DiGraph with bridge edges"
```

---

### Task 2: Cross-namespace BFS

**Files:**
- Modify: `prism_rag/retrieve/bfs.py`
- Test: `tests/test_federated.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_federated.py`:

```python
class TestCrossNamespaceBFS:
    def _make_bridged_fg(self):
        """Two graphs connected via shared tag:python."""
        g1 = _make_graph(
            [("a", "A"), ("tag:python", "python")],
            [("a", "tag:python", "tagged_as")],
        )
        g2 = _make_graph(
            [("x", "X"), ("tag:python", "python")],
            [("x", "tag:python", "tagged_as")],
        )
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        fg.build_bridges()
        return fg

    def test_bfs_crosses_bridge(self):
        """BFS from ns1::a should reach ns2::x via shared tag:python bridge."""
        fg = self._make_bridged_fg()
        results = federated_bfs(fg, "ns1", "a", budget=5000)
        namespaces = {r["namespace"] for r in results}
        assert "ns1" in namespaces
        assert "ns2" in namespaces

    def test_bfs_scope_prevents_crossing(self):
        """With scope=ns1, BFS should NOT cross bridge edges."""
        fg = self._make_bridged_fg()
        results = federated_bfs(fg, "ns1", "a", budget=5000, scope="ns1")
        namespaces = {r["namespace"] for r in results}
        assert namespaces == {"ns1"}

    def test_bfs_single_graph_unchanged(self):
        """Single-graph BFS still works identically (bare IDs, no prefix)."""
        g = _make_graph([("a", "A"), ("b", "B")], [("a", "b", "links_to")])
        fg = FederatedGraph({"ns1": g})
        results = federated_bfs(fg, "ns1", "a", budget=1000)
        ids = [r["id"] for r in results]
        assert "a" in ids
        assert "b" in ids
        assert all(r["namespace"] == "ns1" for r in results)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_federated.py::TestCrossNamespaceBFS -v`
Expected: `test_bfs_crosses_bridge` FAILS (current federated_bfs doesn't cross bridges)

- [ ] **Step 3: Rewrite federated_bfs**

Replace `federated_bfs` in `prism_rag/retrieve/bfs.py`:

```python
def federated_bfs(
    federated: "FederatedGraph",
    namespace: str,
    entry_id: str,
    budget: int = 4000,
    max_depth: int = 10,
    scope: str = "",
) -> list[dict]:
    """BFS traversal starting from a node, crossing bridge edges.

    In single-graph mode, delegates to bfs_traverse() for zero overhead.
    In multi-graph mode, operates on the unified_view.

    Args:
        scope: If non-empty, restrict traversal to this namespace only
               (no bridge crossing).
    """
    if federated.is_single:
        graph = federated.get_graph(namespace)
        if graph is None:
            return []
        nodes = bfs_traverse(graph, entry_id, budget=budget, max_depth=max_depth)
        for n in nodes:
            n["namespace"] = namespace
        return nodes

    # Multi-graph: use unified_view
    uv = federated.unified_view
    entry_qid = f"{namespace}::{entry_id}"
    if entry_qid not in uv:
        return []

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(entry_qid, 0)])
    result: list[dict] = []
    accumulated_tokens = 0

    while queue:
        qid, depth = queue.popleft()
        if qid in visited or depth > max_depth:
            continue

        node_data = uv.nodes[qid]
        node_ns = node_data.get("namespace", namespace)

        # Scope filter: skip nodes outside the requested namespace
        if scope and node_ns != scope:
            continue

        node_tokens = node_data.get("tokens", 0)
        if result and accumulated_tokens + node_tokens > budget:
            continue

        visited.add(qid)
        accumulated_tokens += node_tokens

        # Parse bare ID from qualified ID for the result
        bare_id = qid.split("::", 1)[1] if "::" in qid else qid
        result.append({"id": bare_id, "namespace": node_ns, **{
            k: v for k, v in node_data.items() if k != "namespace"
        }})

        if accumulated_tokens >= budget:
            break

        # Neighbors: outgoing + incoming, sorted by weight
        neighbors: list[tuple[str, float]] = []
        for nbr in uv.neighbors(qid):
            if nbr not in visited:
                w = float(uv.edges[qid, nbr].get("weight", 1.0))
                neighbors.append((nbr, w))
        for pred in uv.predecessors(qid):
            if pred not in visited:
                w = float(uv.edges[pred, qid].get("weight", 1.0))
                neighbors.append((pred, w))

        neighbors.sort(key=lambda p: p[1], reverse=True)
        for nbr, _ in neighbors:
            queue.append((nbr, depth + 1))

    return result
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_federated.py::TestCrossNamespaceBFS tests/test_federated.py::TestFederatedTraversal -v`
Expected: All PASS (new cross-namespace tests + old single-graph tests)

- [ ] **Step 5: Commit**

```bash
git add prism_rag/retrieve/bfs.py tests/test_federated.py
git commit -m "feat: federated_bfs crosses bridge edges via unified_view"
```

---

### Task 3: Cross-namespace DFS

**Files:**
- Modify: `prism_rag/retrieve/dfs.py`
- Test: `tests/test_federated.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_federated.py`:

```python
class TestCrossNamespaceDFS:
    def _make_bridged_fg(self):
        g1 = _make_graph(
            [("a", "A"), ("tag:python", "python")],
            [("a", "tag:python", "tagged_as")],
        )
        g2 = _make_graph(
            [("x", "X"), ("tag:python", "python")],
            [("x", "tag:python", "tagged_as")],
        )
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        fg.build_bridges()
        return fg

    def test_dfs_crosses_bridge(self):
        fg = self._make_bridged_fg()
        results = federated_dfs(fg, "ns1", "a", budget=5000)
        namespaces = {r["namespace"] for r in results}
        assert "ns1" in namespaces
        assert "ns2" in namespaces

    def test_dfs_scope_prevents_crossing(self):
        fg = self._make_bridged_fg()
        results = federated_dfs(fg, "ns1", "a", budget=5000, scope="ns1")
        namespaces = {r["namespace"] for r in results}
        assert namespaces == {"ns1"}

    def test_dfs_single_graph_unchanged(self):
        g = _make_graph(
            [("a", "A"), ("b", "B"), ("c", "C")],
            [("a", "b", "links_to"), ("b", "c", "links_to")],
        )
        fg = FederatedGraph({"ns1": g})
        results = federated_dfs(fg, "ns1", "a", budget=1000)
        ids = [r["id"] for r in results]
        assert ids == ["a", "b", "c"]
        assert all(r["namespace"] == "ns1" for r in results)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_federated.py::TestCrossNamespaceDFS -v`
Expected: `test_dfs_crosses_bridge` FAILS

- [ ] **Step 3: Rewrite federated_dfs**

Replace `federated_dfs` in `prism_rag/retrieve/dfs.py`:

```python
def federated_dfs(
    federated: "FederatedGraph",
    namespace: str,
    entry_id: str,
    budget: int = 4000,
    max_depth: int = 10,
    scope: str = "",
) -> list[dict]:
    """DFS traversal starting from a node, crossing bridge edges.

    In single-graph mode, delegates to dfs_traverse() for zero overhead.
    In multi-graph mode, operates on the unified_view.

    Args:
        scope: If non-empty, restrict traversal to this namespace only.
    """
    if federated.is_single:
        graph = federated.get_graph(namespace)
        if graph is None:
            return []
        nodes = dfs_traverse(graph, entry_id, budget=budget, max_depth=max_depth)
        for n in nodes:
            n["namespace"] = namespace
        return nodes

    # Multi-graph: use unified_view
    uv = federated.unified_view
    entry_qid = f"{namespace}::{entry_id}"
    if entry_qid not in uv:
        return []

    visited: set[str] = set()
    result: list[dict] = []
    accumulated_tokens = 0

    def _dfs(qid: str, depth: int) -> None:
        nonlocal accumulated_tokens

        if qid in visited or depth > max_depth:
            return

        node_data = uv.nodes[qid]
        node_ns = node_data.get("namespace", namespace)

        if scope and node_ns != scope:
            return

        node_tokens = node_data.get("tokens", 0)
        if result and accumulated_tokens + node_tokens > budget:
            return

        visited.add(qid)
        accumulated_tokens += node_tokens

        bare_id = qid.split("::", 1)[1] if "::" in qid else qid
        result.append({"id": bare_id, "namespace": node_ns, **{
            k: v for k, v in node_data.items() if k != "namespace"
        }})

        if accumulated_tokens >= budget:
            return

        neighbors: list[tuple[str, float]] = []
        for nbr in uv.neighbors(qid):
            if nbr not in visited:
                w = float(uv.edges[qid, nbr].get("weight", 1.0))
                neighbors.append((nbr, w))
        for pred in uv.predecessors(qid):
            if pred not in visited:
                w = float(uv.edges[pred, qid].get("weight", 1.0))
                neighbors.append((pred, w))

        neighbors.sort(key=lambda p: p[1], reverse=True)
        for nbr, _ in neighbors:
            _dfs(nbr, depth + 1)

    _dfs(entry_qid, 0)
    return result
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_federated.py::TestCrossNamespaceDFS tests/test_federated.py::TestFederatedTraversal -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add prism_rag/retrieve/dfs.py tests/test_federated.py
git commit -m "feat: federated_dfs crosses bridge edges via unified_view"
```

---

### Task 4: Cross-namespace trace_path

**Files:**
- Modify: `prism_rag/mcp_server/server.py`
- Test: `tests/test_federated.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_federated.py`:

```python
class TestCrossNamespaceTracePath:
    def _make_bridged_fg(self):
        g1 = _make_graph(
            [("a", "A"), ("tag:python", "python")],
            [("a", "tag:python", "tagged_as")],
        )
        g2 = _make_graph(
            [("x", "X"), ("tag:python", "python")],
            [("x", "tag:python", "tagged_as")],
        )
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        fg.build_bridges()
        return fg

    def test_cross_namespace_path_found(self):
        """trace_path should find a path from ns1::a to ns2::x via shared tag bridge."""
        import json
        from prism_rag.mcp_server import server as mcp_mod
        fg = self._make_bridged_fg()
        mcp_mod._federated = fg
        result = json.loads(mcp_mod.trace_path("ns1::a", "ns2::x", max_length=10))
        assert "error" not in result
        assert result["path_length"] >= 1
        # Path should cross namespaces
        step_namespaces = {s["namespace"] for s in result["steps"]}
        assert "ns1" in step_namespaces
        assert "ns2" in step_namespaces

    def test_cross_namespace_no_path(self):
        """Two graphs with no shared tags → no path."""
        import json
        from prism_rag.mcp_server import server as mcp_mod
        g1 = _make_graph([("a", "A")], [])
        g2 = _make_graph([("x", "X")], [])
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        fg.build_bridges()
        mcp_mod._federated = fg
        result = json.loads(mcp_mod.trace_path("ns1::a", "ns2::x"))
        assert "error" in result
        assert result["error"] == "No path found"

    def test_same_namespace_still_works(self):
        """Same-namespace trace_path still works as before."""
        import json
        from prism_rag.mcp_server import server as mcp_mod
        g = _make_graph(
            [("a", "A"), ("b", "B")],
            [("a", "b", "links_to")],
        )
        fg = FederatedGraph({"ns1": g})
        mcp_mod._federated = fg
        result = json.loads(mcp_mod.trace_path("a", "b"))
        assert "error" not in result
        assert result["path_length"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_federated.py::TestCrossNamespaceTracePath -v`
Expected: `test_cross_namespace_path_found` FAILS with "Cross-namespace trace not yet supported"

- [ ] **Step 3: Add federated _node_summary helper**

Add to `prism_rag/mcp_server/server.py` next to existing `_node_summary`:

```python
def _federated_node_summary(fg: FederatedGraph, qualified_id: str) -> dict:
    """Build node summary from a qualified ID (namespace::node_id) in federated graph."""
    if "::" in qualified_id:
        ns, _, bare_id = qualified_id.partition("::")
    elif fg.is_single:
        ns = fg.namespaces[0]
        bare_id = qualified_id
    else:
        ns, bare_id = "", qualified_id

    graph = fg.get_graph(ns)
    if graph is None:
        return {"id": bare_id, "namespace": ns, "label": bare_id, "kind": "?"}

    summary = _node_summary(graph, bare_id)
    summary["namespace"] = ns
    return summary
```

- [ ] **Step 4: Rewrite trace_path**

Replace the `trace_path` function body in `server.py`:

```python
@mcp.tool()
def trace_path(from_node: str, to_node: str, max_length: int = 5) -> str:
    """Find the shortest path between two nodes in the knowledge graph.

    Supports cross-namespace paths via bridge edges.

    Args:
        from_node: Starting node (ID, label, partial name, or namespace::node_id)
        to_node: Ending node (ID, label, partial name, or namespace::node_id)
        max_length: Maximum path length to search (default 5)

    Returns:
        JSON with the shortest path as a sequence of nodes and edges.
    """
    fg = _ensure_federated()
    src_entries = resolve_entry_points(fg, from_node)
    tgt_entries = resolve_entry_points(fg, to_node)

    if not src_entries:
        return json.dumps({"error": f"Source node not found: {from_node!r}"}, ensure_ascii=False)
    if not tgt_entries:
        return json.dumps({"error": f"Target node not found: {to_node!r}"}, ensure_ascii=False)

    src_ns, src_id = src_entries[0]
    tgt_ns, tgt_id = tgt_entries[0]

    if fg.is_single:
        # Single-graph: use original graph directly (no prefixing)
        graph = fg.get_graph(src_ns)
        undirected = graph.g.to_undirected()
        src_qid, tgt_qid = src_id, tgt_id
    else:
        # Multi-graph: use unified view
        undirected = fg.unified_view.to_undirected()
        src_qid = f"{src_ns}::{src_id}"
        tgt_qid = f"{tgt_ns}::{tgt_id}"

    try:
        path = nx.shortest_path(undirected, source=src_qid, target=tgt_qid)
    except nx.NetworkXNoPath:
        return json.dumps({
            "error": "No path found",
            "from": _federated_node_summary(fg, f"{src_ns}::{src_id}"),
            "to": _federated_node_summary(fg, f"{tgt_ns}::{tgt_id}"),
        }, ensure_ascii=False)

    if len(path) - 1 > max_length:
        return json.dumps({
            "error": f"Shortest path has {len(path)-1} hops (max_length={max_length})",
            "from": _federated_node_summary(fg, f"{src_ns}::{src_id}"),
            "to": _federated_node_summary(fg, f"{tgt_ns}::{tgt_id}"),
        }, ensure_ascii=False)

    # Build path steps
    steps = []
    uv = undirected  # already undirected
    for i, qid in enumerate(path):
        step = _federated_node_summary(fg, qid if "::" in qid else f"{src_ns}::{qid}")
        if i < len(path) - 1:
            next_qid = path[i + 1]
            edge_data = uv.edges.get((qid, next_qid), {})
            step["edge_to_next"] = {
                "relation": edge_data.get("relation", "?"),
                "confidence": edge_data.get("confidence", "?"),
                "score": edge_data.get("confidence_score", 0),
            }
        steps.append(step)

    result = {
        "path_length": len(path) - 1,
        "steps": steps,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)
```

Also add the import at top of server.py if not present:

```python
from prism_rag.store.federated import FederatedGraph
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_federated.py::TestCrossNamespaceTracePath -v`
Expected: 3 PASSED

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -q`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add prism_rag/mcp_server/server.py tests/test_federated.py
git commit -m "feat: trace_path supports cross-namespace paths via bridge edges"
```

---

### Task 5: Update search_knowledge to pass scope through

**Files:**
- Modify: `prism_rag/mcp_server/server.py`
- Test: `tests/test_federated.py`

- [ ] **Step 1: Verify search_knowledge passes scope to BFS/DFS**

Check the current `search_knowledge` implementation — it already has a `scope` parameter and calls `federated_bfs`/`federated_dfs`. The new `scope` parameter we added to `federated_bfs`/`federated_dfs` needs to be plumbed through.

Read `server.py` lines 84-139 to confirm how `federated_bfs`/`federated_dfs` are called.

- [ ] **Step 2: Update search_knowledge call sites**

In `server.py`, find where `federated_bfs` and `federated_dfs` are called and add `scope=scope`:

From:
```python
results = federated_bfs(fg, ns, entry_id, budget=budget)
```

To:
```python
results = federated_bfs(fg, ns, entry_id, budget=budget, scope=scope)
```

Same for `federated_dfs` calls.

- [ ] **Step 3: Write test for scoped search**

Add to `tests/test_federated.py`:

```python
class TestSearchScopeIntegration:
    def test_search_without_scope_crosses_bridges(self):
        """search_knowledge without scope should find nodes across namespaces."""
        g1 = _make_graph(
            [("a", "Python Guide"), ("tag:python", "python")],
            [("a", "tag:python", "tagged_as")],
        )
        g2 = _make_graph(
            [("x", "Python Tutorial"), ("tag:python", "python")],
            [("x", "tag:python", "tagged_as")],
        )
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        fg.build_bridges()
        results = federated_bfs(fg, "ns1", "a", budget=5000)
        namespaces = {r["namespace"] for r in results}
        assert len(namespaces) == 2

    def test_search_with_scope_stays_local(self):
        g1 = _make_graph(
            [("a", "Python Guide"), ("tag:python", "python")],
            [("a", "tag:python", "tagged_as")],
        )
        g2 = _make_graph(
            [("x", "Python Tutorial"), ("tag:python", "python")],
            [("x", "tag:python", "tagged_as")],
        )
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        fg.build_bridges()
        results = federated_bfs(fg, "ns1", "a", budget=5000, scope="ns1")
        namespaces = {r["namespace"] for r in results}
        assert namespaces == {"ns1"}
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_federated.py::TestSearchScopeIntegration -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add prism_rag/mcp_server/server.py tests/test_federated.py
git commit -m "feat: search_knowledge passes scope to BFS/DFS for cross-namespace control"
```

---

### Task 6: Full E2E test + final verification

**Files:**
- Test: `tests/test_federated.py`

- [ ] **Step 1: Add comprehensive E2E test**

Add to `tests/test_federated.py`:

```python
class TestCrossNamespaceE2E:
    def test_full_cross_namespace_pipeline(self, tmp_path):
        """End-to-end: two graphs → federate → BFS crosses bridge → trace_path crosses bridge."""
        import json
        from prism_rag.mcp_server import server as mcp_mod

        g1 = _make_graph(
            [("session-mgmt", "Session Management"), ("auth", "Authentication"),
             ("tag:backend", "backend")],
            [("session-mgmt", "auth", "links_to"),
             ("session-mgmt", "tag:backend", "tagged_as"),
             ("auth", "tag:backend", "tagged_as")],
        )
        g2 = _make_graph(
            [("standup-0312", "Standup March 12"), ("retro-0315", "Retro March 15"),
             ("tag:backend", "backend")],
            [("standup-0312", "retro-0315", "links_to"),
             ("standup-0312", "tag:backend", "tagged_as")],
        )

        fg = FederatedGraph({"tech": g1, "meetings": g2})
        fg.build_bridges()
        assert len(fg.bridges) >= 1

        # BFS from tech::session-mgmt should reach meetings via tag:backend
        results = federated_bfs(fg, "tech", "session-mgmt", budget=5000)
        namespaces = {r["namespace"] for r in results}
        assert "tech" in namespaces
        assert "meetings" in namespaces, "BFS should cross bridge to meetings namespace"

        # trace_path from tech::session-mgmt to meetings::standup-0312
        mcp_mod._federated = fg
        path_result = json.loads(mcp_mod.trace_path(
            "tech::session-mgmt", "meetings::standup-0312", max_length=10
        ))
        assert "error" not in path_result, f"trace_path failed: {path_result}"
        assert path_result["path_length"] >= 1
        step_ns = {s["namespace"] for s in path_result["steps"]}
        assert "tech" in step_ns
        assert "meetings" in step_ns
```

- [ ] **Step 2: Run E2E test**

Run: `python -m pytest tests/test_federated.py::TestCrossNamespaceE2E -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_federated.py
git commit -m "test: cross-namespace E2E — BFS + trace_path across bridge edges"
```
