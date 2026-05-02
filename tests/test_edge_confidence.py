"""T4.5 — tests for edge confidence tier filtering in BFS and DFS traversal."""

import pytest

from prism_rag.retrieve.bfs import bfs_traverse, federated_bfs
from prism_rag.retrieve.dfs import dfs_traverse, federated_dfs
from prism_rag.store.federated import FederatedGraph
from prism_rag.store.graph import Edge, KnowledgeGraph, Node


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _kg():
    """Build a small graph for confidence filtering tests.

    Graph:
        root --EXTRACTED(1.0)--> mid --INFERRED(0.70)--> leaf
                                      --AMBIGUOUS(0.10)--> ambiguous_leaf
    """
    kg = KnowledgeGraph()
    for nid, label in [("root", "Root"), ("mid", "Mid"), ("leaf", "Leaf"), ("amb", "Ambiguous")]:
        kg.add_node(Node(id=nid, label=label, kind="note"))

    kg.add_edge(Edge(source="root", target="mid", relation="links_to",
                     confidence="EXTRACTED", confidence_score=1.0))
    kg.add_edge(Edge(source="mid", target="leaf", relation="links_to",
                     confidence="INFERRED", confidence_score=0.70))
    kg.add_edge(Edge(source="mid", target="amb", relation="links_to",
                     confidence="AMBIGUOUS", confidence_score=0.10))
    return kg


def _fg(kg=None):
    if kg is None:
        kg = _kg()
    fg = FederatedGraph({"test": kg})
    fg.build_bridges()
    return fg


# ── bfs_traverse confidence filter ───────────────────────────────────────────


def test_bfs_no_filter_reaches_all():
    kg = _kg()
    result = bfs_traverse(kg, "root", max_depth=3, allowed_tiers=None)
    ids = {r["id"] for r in result}
    assert {"mid", "leaf", "amb"} <= ids


def test_bfs_excludes_ambiguous_with_explicit_tiers():
    kg = _kg()
    result = bfs_traverse(kg, "root", max_depth=3, allowed_tiers={"EXTRACTED", "INFERRED"})
    ids = {r["id"] for r in result}
    assert "amb" not in ids
    assert "leaf" in ids


def test_bfs_min_confidence_filter():
    kg = _kg()
    result = bfs_traverse(kg, "root", max_depth=3, min_confidence=0.80, allowed_tiers=None)
    ids = {r["id"] for r in result}
    # INFERRED(0.70) < 0.80 → leaf excluded
    assert "leaf" not in ids
    assert "mid" in ids


def test_bfs_extracted_only():
    kg = _kg()
    result = bfs_traverse(kg, "root", max_depth=3, allowed_tiers={"EXTRACTED"})
    ids = {r["id"] for r in result}
    assert "mid" in ids
    assert "leaf" not in ids
    assert "amb" not in ids


def test_bfs_all_tiers():
    kg = _kg()
    result = bfs_traverse(kg, "root", max_depth=3, allowed_tiers={"EXTRACTED", "INFERRED", "AMBIGUOUS"})
    ids = {r["id"] for r in result}
    assert "amb" in ids


# ── dfs_traverse confidence filter ───────────────────────────────────────────


def test_dfs_no_filter_reaches_all():
    kg = _kg()
    result = dfs_traverse(kg, "root", max_depth=3, allowed_tiers=None)
    ids = {r["id"] for r in result}
    assert {"mid", "leaf", "amb"} <= ids


def test_dfs_excludes_ambiguous_with_explicit_tiers():
    kg = _kg()
    result = dfs_traverse(kg, "root", max_depth=3, allowed_tiers={"EXTRACTED", "INFERRED"})
    ids = {r["id"] for r in result}
    assert "amb" not in ids
    assert "leaf" in ids


def test_dfs_min_confidence_filter():
    kg = _kg()
    result = dfs_traverse(kg, "root", max_depth=3, min_confidence=0.80, allowed_tiers=None)
    ids = {r["id"] for r in result}
    assert "leaf" not in ids


def test_dfs_extracted_only():
    kg = _kg()
    result = dfs_traverse(kg, "root", max_depth=3, allowed_tiers={"EXTRACTED"})
    ids = {r["id"] for r in result}
    assert "mid" in ids
    assert "leaf" not in ids
    assert "amb" not in ids


# ── federated_bfs / federated_dfs confidence filter ──────────────────────────


def test_federated_bfs_excludes_ambiguous():
    fg = _fg()
    # federated_bfs default is EXTRACTED+INFERRED only
    result = federated_bfs(fg, "test", "root", max_depth=3)
    ids = {r["id"] for r in result}
    assert "amb" not in ids
    assert "leaf" in ids


def test_federated_bfs_include_all_tiers():
    fg = _fg()
    result = federated_bfs(fg, "test", "root", max_depth=3,
                           allowed_tiers={"EXTRACTED", "INFERRED", "AMBIGUOUS"})
    ids = {r["id"] for r in result}
    assert "amb" in ids


def test_federated_dfs_excludes_ambiguous():
    fg = _fg()
    # federated_dfs default is EXTRACTED+INFERRED only
    result = federated_dfs(fg, "test", "root", max_depth=3)
    ids = {r["id"] for r in result}
    assert "amb" not in ids
    assert "leaf" in ids


def test_federated_dfs_include_all_tiers():
    fg = _fg()
    result = federated_dfs(fg, "test", "root", max_depth=3,
                           allowed_tiers={"EXTRACTED", "INFERRED", "AMBIGUOUS"})
    ids = {r["id"] for r in result}
    assert "amb" in ids
