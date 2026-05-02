# PrismRag Federated Multi-Graph Implementation Plan

> **[STALE — 2026-04-30]** FederatedGraph 已实现（store/federated.py），跨 namespace 遍历和 bridge edges 已实现。本计划 49 个任务均未勾选但代码已完成，请勿作为执行依据。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement multi-vault federation so a single PrismRag MCP server can load multiple independently-ingested knowledge graphs and query across them via bridge edges.

**Architecture:** Each vault is ingested independently into its own `graph.json` (existing pipeline, unchanged). A new `FederatedGraph` class loads N graphs at serve-time, computes bridge edges (shared tags + embedding similarity), and exposes them through the existing MCP tools with an added `scope` parameter for namespace filtering. Backward-compatible: single-graph config works identically to today.

**Tech Stack:** Python 3.12, Pydantic v2, NetworkX, FastMCP, pytest

**Spec:** `docs/FEDERATED_GRAPH_DESIGN.md`

---

## File Structure

| File | Role | Action |
|---|---|---|
| `prism_rag/config.py` | Add `GraphSource` model + extend `PrismRagSettings` with `graphs` list | Modify |
| `prism_rag/store/federated.py` | `FederatedGraph` class — multi-graph loading + bridge edges + namespaced queries | Create |
| `prism_rag/retrieve/entry.py` | Support `namespace::node_id` addressing + multi-graph search | Modify |
| `prism_rag/retrieve/bfs.py` | Accept `FederatedGraph`, cross bridge edges | Modify |
| `prism_rag/retrieve/dfs.py` | Accept `FederatedGraph`, cross bridge edges | Modify |
| `prism_rag/mcp_server/server.py` | Load `FederatedGraph` at startup, add `scope` param to tools, add `list_namespaces` tool | Modify |
| `prism_rag/cli.py` | `ingest --namespace`, `query --scope`, `serve` uses federated loader | Modify |
| `tests/test_federated.py` | Unit + integration tests for FederatedGraph | Create |

**Not modified:** `prism_rag/ingest/` — ingest pipeline is per-vault and doesn't know about federation.

---

### Task 1: Extend config with GraphSource + multi-graph settings

**Files:**
- Modify: `prism_rag/config.py`
- Test: `tests/test_federated.py`

- [ ] **Step 1: Write failing tests for config**

```python
# tests/test_federated.py
"""Tests for federated multi-graph functionality."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_rag.config import GraphSource, PrismRagSettings


class TestGraphSource:
    def test_graph_source_basic(self, tmp_path):
        src = GraphSource(
            namespace="nimbus",
            vault_path=tmp_path / "vault",
            data_dir=tmp_path / "data" / "nimbus",
        )
        assert src.namespace == "nimbus"
        assert src.writable is False  # default

    def test_graph_source_graph_path(self, tmp_path):
        src = GraphSource(
            namespace="work",
            vault_path=tmp_path / "vault",
            data_dir=tmp_path / "data" / "work",
        )
        assert src.graph_path == tmp_path / "data" / "work" / "graph.json"


class TestSettingsBackwardCompat:
    def test_single_vault_path_still_works(self, tmp_path, monkeypatch):
        """Old-style PRISM_VAULT_PATH + PRISM_DATA_DIR still works."""
        monkeypatch.setenv("PRISM_VAULT_PATH", str(tmp_path / "vault"))
        monkeypatch.setenv("PRISM_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.delenv("PRISM_GRAPHS", raising=False)
        s = PrismRagSettings()
        graphs = s.resolved_graphs
        assert len(graphs) == 1
        assert graphs[0].namespace == "default"
        assert graphs[0].vault_path == tmp_path / "vault"
        assert graphs[0].data_dir == tmp_path / "data"

    def test_explicit_graphs_env(self, tmp_path, monkeypatch):
        """PRISM_GRAPHS JSON overrides vault_path."""
        graphs_json = json.dumps([
            {"namespace": "a", "vault_path": str(tmp_path / "va"), "data_dir": str(tmp_path / "da")},
            {"namespace": "b", "vault_path": str(tmp_path / "vb"), "data_dir": str(tmp_path / "db"), "writable": True},
        ])
        monkeypatch.setenv("PRISM_GRAPHS", graphs_json)
        s = PrismRagSettings()
        graphs = s.resolved_graphs
        assert len(graphs) == 2
        assert graphs[0].namespace == "a"
        assert graphs[1].writable is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/kingy/Foundation/PrismRag && python -m pytest tests/test_federated.py::TestGraphSource -x -v`
Expected: FAIL (GraphSource not defined)

- [ ] **Step 3: Implement GraphSource and extend PrismRagSettings**

In `prism_rag/config.py`, add `GraphSource` model and `resolved_graphs` property:

```python
from pydantic import BaseModel

class GraphSource(BaseModel):
    """One vault's graph source configuration."""
    namespace: str                      # "nimbus", "work", "personal"
    vault_path: Path                    # vault root directory
    data_dir: Path                      # graph.json + cache location
    writable: bool = False              # only one MCP instance should write per vault

    @property
    def graph_path(self) -> Path:
        return self.data_dir / "graph.json"
```

In `PrismRagSettings`, add:

```python
    # ── Multi-graph federation ──────────────────────────────────────
    graphs: list[GraphSource] | None = Field(
        default=None,
        description="Multi-graph sources. Overrides vault_path/data_dir when set. "
                    "Set via PRISM_GRAPHS env var as JSON array.",
    )
    multi_graph_mode: str = Field(
        default="federated",
        description="'federated' (independent graphs + bridges) or 'merged' (unified re-Leiden). "
                    "Only 'federated' is implemented.",
    )

    @property
    def resolved_graphs(self) -> list[GraphSource]:
        """Return the effective list of graph sources.

        If `graphs` is explicitly set (via PRISM_GRAPHS env var), use it.
        Otherwise, synthesize a single-graph source from legacy vault_path + data_dir
        for backward compatibility.
        """
        if self.graphs:
            return list(self.graphs)
        return [GraphSource(
            namespace="default",
            vault_path=self.vault_path,
            data_dir=self.data_dir,
        )]
```

Also add a `@field_validator` for `graphs` to handle JSON string from env var:

```python
    @field_validator("graphs", mode="before")
    @classmethod
    def _parse_graphs_json(cls, v):
        if isinstance(v, str):
            import json as _json
            return [GraphSource(**g) for g in _json.loads(v)]
        return v
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_federated.py::TestGraphSource tests/test_federated.py::TestSettingsBackwardCompat -x -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add prism_rag/config.py tests/test_federated.py
git commit -m "feat: add GraphSource config + resolved_graphs backward compat"
```

---

### Task 2: FederatedGraph class — multi-graph loading

**Files:**
- Create: `prism_rag/store/federated.py`
- Test: `tests/test_federated.py`

- [ ] **Step 1: Write failing tests for FederatedGraph loading**

Append to `tests/test_federated.py`:

```python
from prism_rag.store.graph import Edge, KnowledgeGraph, Node
from prism_rag.store.federated import FederatedGraph


def _make_graph(nodes: list[tuple[str, str]], edges: list[tuple[str, str, str]] = ()) -> KnowledgeGraph:
    """Helper: create a small graph from (id, label) tuples and (src, tgt, relation) tuples."""
    g = KnowledgeGraph()
    for nid, label in nodes:
        g.add_node(Node(id=nid, label=label, kind="note", tokens=50, content=f"Content of {label}"))
    for src, tgt, rel in edges:
        g.add_edge(Edge(source=src, target=tgt, relation=rel, confidence="EXTRACTED"))
    return g


class TestFederatedGraphLoad:
    def test_single_graph(self):
        g = _make_graph([("a", "A"), ("b", "B")], [("a", "b", "links_to")])
        fg = FederatedGraph({"nimbus": g})
        assert fg.node_count == 2
        assert fg.edge_count == 1
        assert fg.namespaces == ["nimbus"]

    def test_multi_graph_node_count(self):
        g1 = _make_graph([("a", "A"), ("b", "B")])
        g2 = _make_graph([("x", "X"), ("y", "Y"), ("z", "Z")])
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        assert fg.node_count == 5
        assert fg.namespaces == ["ns1", "ns2"]

    def test_namespaced_node_access(self):
        g1 = _make_graph([("a", "A")])
        g2 = _make_graph([("a", "Alpha")])  # same ID, different graph
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        assert fg.get_node("ns1::a") is not None
        assert fg.get_node("ns2::a") is not None
        assert fg.get_node("ns1::a")["label"] == "A"
        assert fg.get_node("ns2::a")["label"] == "Alpha"

    def test_get_graph_by_namespace(self):
        g1 = _make_graph([("a", "A")])
        fg = FederatedGraph({"nimbus": g1})
        assert fg.get_graph("nimbus") is g1
        assert fg.get_graph("nonexistent") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_federated.py::TestFederatedGraphLoad -x -v`
Expected: FAIL (FederatedGraph not defined)

- [ ] **Step 3: Implement FederatedGraph core**

Create `prism_rag/store/federated.py`:

```python
"""Federated multi-graph layer.

Loads multiple independent KnowledgeGraph instances and provides unified
query access with namespace-prefixed node IDs.

Bridge edges (cross-graph) are computed at load time (serve-time),
not persisted — they depend on which graphs are loaded.
"""
from __future__ import annotations

import logging
from typing import Any

from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class FederatedGraph:
    """Runtime federation over multiple KnowledgeGraph instances.

    Each graph is identified by a namespace string.
    Nodes are addressed as "namespace::node_id" in multi-graph mode.
    In single-graph mode (one namespace), bare node IDs work without prefix.
    """

    def __init__(self, graphs: dict[str, KnowledgeGraph]) -> None:
        self._graphs: dict[str, KnowledgeGraph] = dict(graphs)
        self._single = len(self._graphs) == 1
        self._bridges: list[dict] = []  # computed by build_bridges()

    @property
    def namespaces(self) -> list[str]:
        return sorted(self._graphs.keys())

    @property
    def node_count(self) -> int:
        return sum(g.node_count for g in self._graphs.values())

    @property
    def edge_count(self) -> int:
        return sum(g.edge_count for g in self._graphs.values()) + len(self._bridges)

    @property
    def is_single(self) -> bool:
        return self._single

    @property
    def bridges(self) -> list[dict]:
        return self._bridges

    def get_graph(self, namespace: str) -> KnowledgeGraph | None:
        return self._graphs.get(namespace)

    def get_node(self, qualified_id: str) -> dict[str, Any] | None:
        """Get node data by qualified ID ("namespace::node_id").

        In single-graph mode, bare node_id (no prefix) is accepted.
        """
        ns, node_id = self._parse_id(qualified_id)
        graph = self._graphs.get(ns)
        if graph is None:
            return None
        if node_id not in graph.g:
            return None
        return dict(graph.g.nodes[node_id])

    def _parse_id(self, qualified_id: str) -> tuple[str, str]:
        """Parse "namespace::node_id" → (namespace, node_id).

        In single-graph mode, bare IDs map to the only namespace.
        """
        if "::" in qualified_id:
            ns, _, node_id = qualified_id.partition("::")
            return ns, node_id
        if self._single:
            return next(iter(self._graphs)), qualified_id
        # Multi-graph but no prefix — search all graphs for a match
        for ns, g in self._graphs.items():
            if qualified_id in g.g:
                return ns, qualified_id
        return "", qualified_id  # will fail on lookup
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_federated.py::TestFederatedGraphLoad -x -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add prism_rag/store/federated.py tests/test_federated.py
git commit -m "feat: FederatedGraph class — multi-graph loading + namespaced access"
```

---

### Task 3: Bridge edge computation

**Files:**
- Modify: `prism_rag/store/federated.py`
- Test: `tests/test_federated.py`

- [ ] **Step 1: Write failing tests for bridge edges**

Append to `tests/test_federated.py`:

```python
class TestBridgeEdges:
    def test_shared_tag_bridge(self):
        """Two graphs sharing tag:python → bridge edge created."""
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
        assert len(fg.bridges) >= 1
        bridge = fg.bridges[0]
        assert bridge["relation"] == "shared_tag"
        assert {bridge["source_ns"], bridge["target_ns"]} == {"ns1", "ns2"}

    def test_no_bridges_in_single_graph(self):
        g = _make_graph([("a", "A")])
        fg = FederatedGraph({"ns1": g})
        fg.build_bridges()
        assert len(fg.bridges) == 0

    def test_bridge_count_multiple_shared_tags(self):
        """Two shared tags → two bridge edges."""
        g1 = _make_graph(
            [("a", "A"), ("tag:python", "python"), ("tag:rust", "rust")],
            [("a", "tag:python", "tagged_as"), ("a", "tag:rust", "tagged_as")],
        )
        g2 = _make_graph(
            [("x", "X"), ("tag:python", "python"), ("tag:rust", "rust")],
            [("x", "tag:python", "tagged_as"), ("x", "tag:rust", "tagged_as")],
        )
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        fg.build_bridges()
        # Each shared tag creates bridges between all graph pairs
        assert len(fg.bridges) >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_federated.py::TestBridgeEdges -x -v`
Expected: FAIL (build_bridges not implemented)

- [ ] **Step 3: Implement build_bridges**

Add to `FederatedGraph` in `prism_rag/store/federated.py`:

```python
    def build_bridges(self) -> int:
        """Compute cross-graph bridge edges.

        Bridge types:
        1. Shared tags: same tag node ID exists in multiple graphs
           → bridge between the tag nodes

        Returns: number of bridge edges created.
        """
        self._bridges.clear()

        if self._single:
            return 0

        # --- 1. Shared tag bridges ---
        # Collect tag nodes per namespace
        tag_index: dict[str, list[str]] = {}  # tag_id → [namespace, ...]
        for ns, g in self._graphs.items():
            for node_id, data in g.g.nodes(data=True):
                if data.get("kind") == "tag":
                    tag_index.setdefault(node_id, []).append(ns)

        # Create bridge for each tag shared by 2+ namespaces
        for tag_id, namespaces in tag_index.items():
            if len(namespaces) < 2:
                continue
            # Create pairwise bridges
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

        logger.info(f"[federated] built {len(self._bridges)} bridge edges "
                    f"across {len(self._graphs)} graphs")
        return len(self._bridges)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_federated.py::TestBridgeEdges -x -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add prism_rag/store/federated.py tests/test_federated.py
git commit -m "feat: bridge edge computation — shared tags across graphs"
```

---

### Task 4: Multi-graph entry point resolution

**Files:**
- Modify: `prism_rag/retrieve/entry.py`
- Test: `tests/test_federated.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_federated.py`:

```python
from prism_rag.retrieve.entry import resolve_entry_points


class TestFederatedEntryResolution:
    def test_single_graph_bare_query(self):
        g = _make_graph([("session-mgmt", "Session Management")])
        fg = FederatedGraph({"nimbus": g})
        results = resolve_entry_points(fg, "session management")
        assert len(results) == 1
        assert results[0] == ("nimbus", "session-mgmt")

    def test_multi_graph_finds_in_all(self):
        g1 = _make_graph([("arch", "Architecture")])
        g2 = _make_graph([("arch-review", "Architecture Review")])
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        results = resolve_entry_points(fg, "architecture")
        assert len(results) == 2

    def test_scoped_query(self):
        g1 = _make_graph([("a", "Design")])
        g2 = _make_graph([("b", "Design Doc")])
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        results = resolve_entry_points(fg, "design", scope="ns1")
        assert len(results) == 1
        assert results[0][0] == "ns1"

    def test_qualified_id_query(self):
        g1 = _make_graph([("a", "A")])
        fg = FederatedGraph({"nimbus": g1})
        results = resolve_entry_points(fg, "nimbus::a")
        assert len(results) == 1
        assert results[0] == ("nimbus", "a")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_federated.py::TestFederatedEntryResolution -x -v`
Expected: FAIL (resolve_entry_points not defined)

- [ ] **Step 3: Implement resolve_entry_points**

Add to `prism_rag/retrieve/entry.py`:

```python
from prism_rag.store.federated import FederatedGraph


def resolve_entry_points(
    federated: FederatedGraph,
    query: str,
    scope: str | None = None,
) -> list[tuple[str, str]]:
    """Resolve entry points across a federated graph.

    Supports:
    - "namespace::node_id" qualified addressing
    - scope="namespace" to search only one graph
    - bare query searches all graphs

    Args:
        federated: The federated graph to search.
        query: User query string or qualified node ID.
        scope: Optional namespace to restrict search.

    Returns:
        List of (namespace, node_id) tuples, ordered by match quality.
    """
    # Handle qualified ID (e.g., "nimbus::session_mgmt")
    if "::" in query:
        ns, _, node_id = query.partition("::")
        graph = federated.get_graph(ns)
        if graph and node_id in graph.g:
            return [(ns, node_id)]
        # Try resolving the node_id part within the namespace
        if graph:
            match = resolve_entry_point(graph, node_id)
            if match:
                return [(ns, match)]
        return []

    # Determine which namespaces to search
    if scope:
        search_ns = [scope] if federated.get_graph(scope) else []
    else:
        search_ns = federated.namespaces

    # Search each namespace
    results: list[tuple[str, str]] = []
    for ns in search_ns:
        graph = federated.get_graph(ns)
        if graph is None:
            continue
        match = resolve_entry_point(graph, query)
        if match:
            results.append((ns, match))

    return results
```

Keep the existing `resolve_entry_point` function unchanged (single-graph, used internally).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_federated.py::TestFederatedEntryResolution -x -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add prism_rag/retrieve/entry.py tests/test_federated.py
git commit -m "feat: multi-graph entry point resolution with scope + namespace addressing"
```

---

### Task 5: Federated BFS/DFS traversal

**Files:**
- Modify: `prism_rag/retrieve/bfs.py`
- Modify: `prism_rag/retrieve/dfs.py`
- Test: `tests/test_federated.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_federated.py`:

```python
from prism_rag.retrieve.bfs import federated_bfs
from prism_rag.retrieve.dfs import federated_dfs


class TestFederatedTraversal:
    def _two_graph_fg(self):
        g1 = _make_graph(
            [("a", "A"), ("b", "B"), ("tag:shared", "shared")],
            [("a", "b", "links_to"), ("a", "tag:shared", "tagged_as")],
        )
        g2 = _make_graph(
            [("x", "X"), ("y", "Y"), ("tag:shared", "shared")],
            [("x", "y", "links_to"), ("x", "tag:shared", "tagged_as")],
        )
        fg = FederatedGraph({"ns1": g1, "ns2": g2})
        fg.build_bridges()
        return fg

    def test_bfs_single_graph(self):
        g = _make_graph([("a", "A"), ("b", "B")], [("a", "b", "links_to")])
        fg = FederatedGraph({"ns1": g})
        results = federated_bfs(fg, "ns1", "a", budget=1000)
        ids = [r["id"] for r in results]
        assert "a" in ids
        assert "b" in ids

    def test_bfs_stays_in_scope(self):
        fg = self._two_graph_fg()
        results = federated_bfs(fg, "ns1", "a", budget=1000)
        # Should traverse within ns1 only (no cross_bridges by default without follow_bridges)
        ns1_ids = {n["id"] for n in results}
        assert "a" in ns1_ids

    def test_dfs_single_graph(self):
        g = _make_graph(
            [("a", "A"), ("b", "B"), ("c", "C")],
            [("a", "b", "links_to"), ("b", "c", "links_to")],
        )
        fg = FederatedGraph({"ns1": g})
        results = federated_dfs(fg, "ns1", "a", budget=1000)
        ids = [r["id"] for r in results]
        assert ids == ["a", "b", "c"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_federated.py::TestFederatedTraversal -x -v`
Expected: FAIL (federated_bfs not defined)

- [ ] **Step 3: Implement federated_bfs**

Add to `prism_rag/retrieve/bfs.py`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism_rag.store.federated import FederatedGraph


def federated_bfs(
    federated: "FederatedGraph",
    namespace: str,
    entry_id: str,
    budget: int = 4000,
    max_depth: int = 10,
) -> list[dict]:
    """BFS traversal starting from a node in a specific namespace.

    Traverses within the namespace's graph. Bridge edges are not
    automatically followed (future: add follow_bridges param).

    Results include a "namespace" key on each node dict.
    """
    graph = federated.get_graph(namespace)
    if graph is None:
        return []
    nodes = bfs_traverse(graph, entry_id, budget=budget, max_depth=max_depth)
    # Tag each result with its namespace
    for n in nodes:
        n["namespace"] = namespace
    return nodes
```

- [ ] **Step 4: Implement federated_dfs**

Add to `prism_rag/retrieve/dfs.py`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism_rag.store.federated import FederatedGraph


def federated_dfs(
    federated: "FederatedGraph",
    namespace: str,
    entry_id: str,
    budget: int = 4000,
    max_depth: int = 10,
) -> list[dict]:
    """DFS traversal starting from a node in a specific namespace.

    Traverses within the namespace's graph. Bridge edges are not
    automatically followed (future: add follow_bridges param).

    Results include a "namespace" key on each node dict.
    """
    graph = federated.get_graph(namespace)
    if graph is None:
        return []
    nodes = dfs_traverse(graph, entry_id, budget=budget, max_depth=max_depth)
    for n in nodes:
        n["namespace"] = namespace
    return nodes
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_federated.py::TestFederatedTraversal -x -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add prism_rag/retrieve/bfs.py prism_rag/retrieve/dfs.py tests/test_federated.py
git commit -m "feat: federated BFS/DFS traversal with namespace tagging"
```

---

### Task 6: FederatedGraph loader from config

**Files:**
- Modify: `prism_rag/store/federated.py`
- Test: `tests/test_federated.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_federated.py`:

```python
from prism_rag.config import GraphSource


class TestFederatedLoader:
    def test_load_from_graph_sources(self, tmp_path):
        """Load FederatedGraph from GraphSource configs with real graph.json files."""
        # Create two graph.json files
        g1 = _make_graph([("a", "A")], [])
        g2 = _make_graph([("x", "X"), ("y", "Y")], [("x", "y", "links_to")])
        d1 = tmp_path / "data" / "ns1"
        d2 = tmp_path / "data" / "ns2"
        d1.mkdir(parents=True)
        d2.mkdir(parents=True)
        g1.save(d1 / "graph.json")
        g2.save(d2 / "graph.json")

        sources = [
            GraphSource(namespace="ns1", vault_path=tmp_path / "v1", data_dir=d1),
            GraphSource(namespace="ns2", vault_path=tmp_path / "v2", data_dir=d2),
        ]
        fg = FederatedGraph.load(sources)
        assert fg.node_count == 3
        assert fg.namespaces == ["ns1", "ns2"]

    def test_load_skips_missing_graph(self, tmp_path):
        """Missing graph.json is skipped with warning, not error."""
        g1 = _make_graph([("a", "A")])
        d1 = tmp_path / "data" / "ns1"
        d1.mkdir(parents=True)
        g1.save(d1 / "graph.json")

        sources = [
            GraphSource(namespace="ns1", vault_path=tmp_path / "v1", data_dir=d1),
            GraphSource(namespace="missing", vault_path=tmp_path / "v2", data_dir=tmp_path / "nope"),
        ]
        fg = FederatedGraph.load(sources)
        assert fg.node_count == 1
        assert fg.namespaces == ["ns1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_federated.py::TestFederatedLoader -x -v`
Expected: FAIL (FederatedGraph.load not defined)

- [ ] **Step 3: Implement FederatedGraph.load classmethod**

Add to `FederatedGraph` in `prism_rag/store/federated.py`:

```python
    @classmethod
    def load(cls, sources: list) -> "FederatedGraph":
        """Load a FederatedGraph from a list of GraphSource configs.

        Skips sources whose graph.json doesn't exist (logs a warning).
        Automatically computes bridge edges after loading.

        Args:
            sources: list of GraphSource (from config.resolved_graphs)

        Returns:
            FederatedGraph with all loadable graphs + bridges.
        """
        graphs: dict[str, KnowledgeGraph] = {}
        for src in sources:
            gpath = src.graph_path
            if not gpath.exists():
                logger.warning(f"[federated] graph not found: {gpath} (namespace={src.namespace}), skipping")
                continue
            g = KnowledgeGraph.load(gpath)
            graphs[src.namespace] = g
            logger.info(
                f"[federated] loaded {src.namespace}: "
                f"{g.node_count} nodes, {g.edge_count} edges"
            )

        fg = cls(graphs)
        fg.build_bridges()
        return fg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_federated.py::TestFederatedLoader -x -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add prism_rag/store/federated.py tests/test_federated.py
git commit -m "feat: FederatedGraph.load() from GraphSource configs + auto-bridge"
```

---

### Task 7: Update MCP server for federated loading

**Files:**
- Modify: `prism_rag/mcp_server/server.py`
- Test: `tests/test_federated.py`

- [ ] **Step 1: Write failing test for federated MCP tools**

Append to `tests/test_federated.py`:

```python
class TestMCPFederatedIntegration:
    def test_ensure_federated_returns_federated_graph(self, tmp_path, monkeypatch):
        """_ensure_federated() loads a FederatedGraph from settings."""
        from prism_rag.mcp_server import server as mcp_mod

        g = _make_graph([("a", "Session Management")])
        d = tmp_path / "data"
        d.mkdir()
        g.save(d / "graph.json")

        monkeypatch.setenv("PRISM_VAULT_PATH", str(tmp_path / "vault"))
        monkeypatch.setenv("PRISM_DATA_DIR", str(d))
        monkeypatch.delenv("PRISM_GRAPHS", raising=False)

        # Reset global state
        mcp_mod._federated = None
        fg = mcp_mod._ensure_federated()
        assert isinstance(fg, FederatedGraph)
        assert fg.node_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_federated.py::TestMCPFederatedIntegration -x -v`
Expected: FAIL (_ensure_federated not defined)

- [ ] **Step 3: Update server.py**

Replace the global `_graph` singleton with `_federated` and update all tools:

1. Replace `_graph: KnowledgeGraph | None = None` with `_federated: FederatedGraph | None = None`
2. Replace `_ensure_graph()` with `_ensure_federated()`:

```python
from prism_rag.store.federated import FederatedGraph

_federated: FederatedGraph | None = None

def _ensure_federated() -> FederatedGraph:
    global _federated
    if _federated is None:
        settings = PrismRagSettings()
        _federated = FederatedGraph.load(settings.resolved_graphs)
        logger.info(
            f"[mcp] federated loaded: {_federated.node_count} nodes, "
            f"{_federated.edge_count} edges across {len(_federated.namespaces)} namespaces"
        )
    return _federated
```

3. Update `search_knowledge` to accept `scope` parameter and use federated traversal:

```python
@mcp.tool()
def search_knowledge(
    query: str,
    budget: int = 4000,
    mode: str = "bfs",
    scope: str = "",
) -> str:
    """Search the knowledge graph for information about a topic.

    Args:
        query: Natural language query or node name
        budget: Maximum tokens to return (default 4000)
        mode: Traversal mode — "bfs" (broad context) or "dfs" (follow chains)
        scope: Namespace to search (e.g., "nimbus"). Empty = search all.
    """
    fg = _ensure_federated()
    entries = resolve_entry_points(fg, query, scope=scope or None)
    if not entries:
        return json.dumps({"error": f"No matching node for query: {query!r}"}, ensure_ascii=False)

    # Use the first (best) entry point
    ns, entry_id = entries[0]

    if mode == "dfs":
        nodes = federated_dfs(fg, ns, entry_id, budget=budget)
    else:
        nodes = federated_bfs(fg, ns, entry_id, budget=budget)

    graph = fg.get_graph(ns)
    result = {
        "entry_point": _node_summary(graph, entry_id),
        "namespace": ns,
        "total_nodes": len(nodes),
        "total_tokens": sum(n.get("tokens", 0) for n in nodes),
        "nodes": [
            {
                "id": f"{ns}::{n['id']}" if not fg.is_single else n["id"],
                "label": n.get("label", n["id"]),
                "kind": n.get("kind", "?"),
                "tokens": n.get("tokens", 0),
                "community": n.get("community_id", ""),
                "content": n.get("content", "")[:2000],
            }
            for n in nodes
            if n.get("kind") == "note"
        ],
    }
    return json.dumps(result, ensure_ascii=False, indent=2)
```

4. Similarly update `explain_node`, `trace_path`, `list_communities`, `explore_community` to use federated graph. For `explain_node` and `trace_path`, resolve via `resolve_entry_points`. For `list_communities` and `explore_community`, iterate all namespaces.

5. Add `list_namespaces` tool:

```python
@mcp.tool()
def list_namespaces() -> str:
    """List all loaded knowledge graph namespaces with statistics."""
    fg = _ensure_federated()
    namespaces = []
    for ns in fg.namespaces:
        g = fg.get_graph(ns)
        namespaces.append({
            "namespace": ns,
            "nodes": g.node_count,
            "edges": g.edge_count,
            "communities": len(g.communities),
        })
    return json.dumps({
        "namespaces": namespaces,
        "bridges": len(fg.bridges),
        "total_nodes": fg.node_count,
    }, ensure_ascii=False, indent=2)
```

6. Update `write_note` and `read_note` to accept `namespace` param and resolve the correct vault_path.

7. Update `run_server()` to call `_ensure_federated()` instead of `_ensure_graph()`.

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/ -x -v`
Expected: All tests pass (including old test_phase1_mvp.py which uses single-graph mode)

- [ ] **Step 5: Commit**

```bash
git add prism_rag/mcp_server/server.py tests/test_federated.py
git commit -m "feat: MCP server loads FederatedGraph + scope param + list_namespaces tool"
```

---

### Task 8: Update CLI for multi-graph

**Files:**
- Modify: `prism_rag/cli.py`

- [ ] **Step 1: Update `ingest` command**

Add `--namespace` option. When set, ingest writes to `data_dir/<namespace>/` instead of `data_dir/`:

```python
@app.command()
def ingest(
    vault: Path = ...,
    output: Path = ...,
    namespace: str = typer.Option("", help="Namespace for this vault's graph (for multi-graph federation)"),
    ...
):
    if namespace:
        output = output / namespace
    ...
```

- [ ] **Step 2: Update `query` command**

Add `--scope` option:

```python
@app.command()
def query(
    q: str = ...,
    scope: str = typer.Option("", help="Namespace to search (empty = all)"),
    ...
):
```

- [ ] **Step 3: Update `serve` command**

Use `FederatedGraph.load(settings.resolved_graphs)` instead of single `KnowledgeGraph.load`.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -x -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add prism_rag/cli.py
git commit -m "feat: CLI --namespace for ingest, --scope for query, federated serve"
```

---

### Task 9: Migrate existing data directory

**Files:**
- No code changes — data migration script

- [ ] **Step 1: Move existing flat data/ to data/default/ namespace**

```bash
cd /home/kingy/Foundation/PrismRag
mkdir -p data/default
mv data/graph.json data/GRAPH_REPORT.md data/graph.html data/default/
# Update PRISM_DATA_DIR if needed, or rely on backward-compat (single vault = data/)
```

Note: This step is optional — backward compatibility means existing flat `data/` still works via `resolved_graphs` returning a single `GraphSource(namespace="default", data_dir=data/)`.

- [ ] **Step 2: Verify serve still works with migrated data**

```bash
cd /home/kingy/Foundation/PrismRag
PRISM_GRAPHS='[{"namespace":"nimbus","vault_path":"~/Foundation/Vault","data_dir":"data/default"}]' python -m prism_rag.cli serve --transport stdio
```

- [ ] **Step 3: Commit data migration if done**

```bash
git add -A data/
git commit -m "chore: migrate flat data/ to data/default/ namespace layout"
```

---

### Task 10: End-to-end integration test

**Files:**
- Test: `tests/test_federated.py`

- [ ] **Step 1: Write E2E test**

Append to `tests/test_federated.py`:

```python
class TestFederatedE2E:
    def test_full_pipeline_two_graphs(self, tmp_path):
        """Create two graphs, federate them, search across both."""
        # Graph 1: tech notes
        g1 = _make_graph(
            [("session-mgmt", "Session Management"), ("auth", "Authentication"),
             ("tag:backend", "backend")],
            [("session-mgmt", "auth", "links_to"),
             ("session-mgmt", "tag:backend", "tagged_as"),
             ("auth", "tag:backend", "tagged_as")],
        )
        # Graph 2: meeting notes
        g2 = _make_graph(
            [("standup-0312", "Standup March 12"), ("retro-0315", "Retro March 15"),
             ("tag:backend", "backend")],
            [("standup-0312", "retro-0315", "links_to"),
             ("standup-0312", "tag:backend", "tagged_as")],
        )

        d1, d2 = tmp_path / "d1", tmp_path / "d2"
        d1.mkdir(); d2.mkdir()
        g1.save(d1 / "graph.json")
        g2.save(d2 / "graph.json")

        sources = [
            GraphSource(namespace="tech", vault_path=tmp_path / "v1", data_dir=d1),
            GraphSource(namespace="meetings", vault_path=tmp_path / "v2", data_dir=d2),
        ]
        fg = FederatedGraph.load(sources)

        # Bridge: shared tag:backend
        assert len(fg.bridges) >= 1

        # Search in tech namespace
        from prism_rag.retrieve.entry import resolve_entry_points
        results = resolve_entry_points(fg, "session management", scope="tech")
        assert len(results) == 1
        assert results[0] == ("tech", "session-mgmt")

        # Search across all
        results_all = resolve_entry_points(fg, "backend")
        # Found in both namespaces (tag:backend exists in both)
        assert len(results_all) == 2

        # BFS within tech
        from prism_rag.retrieve.bfs import federated_bfs
        traversed = federated_bfs(fg, "tech", "session-mgmt", budget=1000)
        assert len(traversed) >= 2  # session-mgmt + auth (+ tag:backend)
        assert all(n.get("namespace") == "tech" for n in traversed)

        # Namespaces
        assert fg.namespaces == ["meetings", "tech"]
```

- [ ] **Step 2: Run E2E test**

Run: `python -m pytest tests/test_federated.py::TestFederatedE2E -x -v`
Expected: PASS

- [ ] **Step 3: Run full test suite (existing + new)**

Run: `python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 4: Final commit**

```bash
git add tests/test_federated.py
git commit -m "test: federated multi-graph E2E integration test"
```
