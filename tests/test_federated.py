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
