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


# ── FederatedGraph tests ──────────────────────────────────────────────────────

from prism_rag.store.graph import Edge, KnowledgeGraph, Node
from prism_rag.store.federated import FederatedGraph


def _make_graph(nodes: list[tuple[str, str]], edges: list[tuple[str, str, str]] = ()) -> KnowledgeGraph:
    """Helper: create a small graph from (id, label) tuples and (src, tgt, relation) tuples.

    Nodes whose ID starts with "tag:" are automatically given kind="tag".
    """
    g = KnowledgeGraph()
    for nid, label in nodes:
        kind = "tag" if nid.startswith("tag:") else "note"
        g.add_node(Node(id=nid, label=label, kind=kind, tokens=50, content=f"Content of {label}"))
    for src, tgt, rel in edges:
        g.add_edge(Edge(source=src, target=tgt, relation=rel, confidence="EXTRACTED"))
    return g


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
        g2 = _make_graph([("a", "Alpha")])
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


class TestBridgeEdges:
    def test_shared_tag_bridge(self):
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
        assert len(fg.bridges) >= 2


# ── Embedding bridges ────────────────────────────────────────────────────────


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


# ── Task 4: Federated entry point resolution ──────────────────────────────────

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


# ── Task 5: Federated BFS/DFS traversal ──────────────────────────────────────

from prism_rag.retrieve.bfs import federated_bfs
from prism_rag.retrieve.dfs import federated_dfs


class TestFederatedTraversal:
    def test_bfs_single_graph(self):
        g = _make_graph([("a", "A"), ("b", "B")], [("a", "b", "links_to")])
        fg = FederatedGraph({"ns1": g})
        results = federated_bfs(fg, "ns1", "a", budget=1000)
        ids = [r["id"] for r in results]
        assert "a" in ids
        assert "b" in ids

    def test_bfs_tags_namespace(self):
        g = _make_graph([("a", "A"), ("b", "B")], [("a", "b", "links_to")])
        fg = FederatedGraph({"ns1": g})
        results = federated_bfs(fg, "ns1", "a", budget=1000)
        assert all(n.get("namespace") == "ns1" for n in results)

    def test_dfs_single_graph(self):
        g = _make_graph(
            [("a", "A"), ("b", "B"), ("c", "C")],
            [("a", "b", "links_to"), ("b", "c", "links_to")],
        )
        fg = FederatedGraph({"ns1": g})
        results = federated_dfs(fg, "ns1", "a", budget=1000)
        ids = [r["id"] for r in results]
        assert ids == ["a", "b", "c"]

    def test_nonexistent_namespace_returns_empty(self):
        g = _make_graph([("a", "A")])
        fg = FederatedGraph({"ns1": g})
        assert federated_bfs(fg, "nope", "a") == []
        assert federated_dfs(fg, "nope", "a") == []


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
        fg = self._make_bridged_fg()
        results = federated_bfs(fg, "ns1", "a", budget=5000)
        namespaces = {r["namespace"] for r in results}
        assert "ns1" in namespaces
        assert "ns2" in namespaces

    def test_bfs_scope_prevents_crossing(self):
        fg = self._make_bridged_fg()
        results = federated_bfs(fg, "ns1", "a", budget=5000, scope="ns1")
        namespaces = {r["namespace"] for r in results}
        assert namespaces == {"ns1"}

    def test_bfs_single_graph_unchanged(self):
        g = _make_graph([("a", "A"), ("b", "B")], [("a", "b", "links_to")])
        fg = FederatedGraph({"ns1": g})
        results = federated_bfs(fg, "ns1", "a", budget=1000)
        ids = [r["id"] for r in results]
        assert "a" in ids
        assert "b" in ids
        assert all(r["namespace"] == "ns1" for r in results)


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


# ── Task 4: Cross-namespace trace_path ──────────────────────────────────────


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


# ── Task 6: FederatedGraph.load() from config ─────────────────────────────────


class TestFederatedLoader:
    def test_load_from_graph_sources(self, tmp_path):
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


# -- Task 7: MCP server federated integration --------------------------------


class TestMCPFederatedIntegration:
    def test_ensure_federated_returns_federated_graph(self, tmp_path, monkeypatch):
        from prism_rag.mcp_server import server as mcp_mod

        g = _make_graph([("a", "Session Management")])
        d = tmp_path / "data"
        d.mkdir()
        g.save(d / "graph.json")

        monkeypatch.setenv("PRISM_VAULT_PATH", str(tmp_path / "vault"))
        monkeypatch.setenv("PRISM_DATA_DIR", str(d))
        monkeypatch.delenv("PRISM_GRAPHS", raising=False)

        mcp_mod._federated = None
        fg = mcp_mod._ensure_federated()
        assert isinstance(fg, FederatedGraph)
        assert fg.node_count == 1


# ── Task 10: End-to-end integration test ─────────────────────────────────────


class TestFederatedE2E:
    def test_full_pipeline_two_graphs(self, tmp_path):
        """Create two graphs, federate them, search across both."""
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
        results = resolve_entry_points(fg, "session management", scope="tech")
        assert len(results) == 1
        assert results[0] == ("tech", "session-mgmt")

        # Search across all — tag:backend in both
        results_all = resolve_entry_points(fg, "backend")
        assert len(results_all) == 2

        # BFS within tech (scoped to tech namespace only)
        traversed = federated_bfs(fg, "tech", "session-mgmt", budget=1000, scope="tech")
        assert len(traversed) >= 2
        assert all(n.get("namespace") == "tech" for n in traversed)

        # Namespaces
        assert fg.namespaces == ["meetings", "tech"]


class TestCrossNamespaceE2E:
    def test_full_cross_namespace_pipeline(self, tmp_path):
        """End-to-end: two graphs -> federate -> BFS crosses bridge -> trace_path crosses bridge."""
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
