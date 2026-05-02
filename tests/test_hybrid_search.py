"""T2.8 — tests for hybrid BM25 + embedding + exact search and RRF fusion."""

from unittest.mock import MagicMock

import pytest

from prism_rag.retrieve.hybrid import hybrid_search, reciprocal_rank_fusion
from prism_rag.store.graph import KnowledgeGraph, Node


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _kg_with_nodes(*ids_labels):
    """Build a KnowledgeGraph with the given (id, label) pairs."""
    kg = KnowledgeGraph()
    for node_id, label in ids_labels:
        kg.add_node(Node(id=node_id, label=label, kind="note"))
    return kg


def _bm25_stub(hits: list[str]):
    """Return a BM25Index stub that always returns `hits` with score 1.0."""
    stub = MagicMock()
    stub.is_ready = True
    stub.search.return_value = [(nid, 1.0) for nid in hits]
    return stub


def _embed_store_stub(hits: list[str]):
    """Return an EmbeddingStore stub that always returns `hits`."""
    stub = MagicMock()
    stub.nearest.return_value = hits
    return stub


# ── reciprocal_rank_fusion ────────────────────────────────────────────────────


def test_rrf_single_ranking():
    result = reciprocal_rank_fusion([["a", "b", "c"]])
    assert result == ["a", "b", "c"]


def test_rrf_two_rankings_merge():
    r1 = ["a", "b"]
    r2 = ["b", "c"]
    result = reciprocal_rank_fusion([r1, r2])
    assert result[0] == "b"          # b appears in both → highest score
    assert "a" in result
    assert "c" in result


def test_rrf_deduplicates():
    result = reciprocal_rank_fusion([["a", "a", "b"]])
    # Deduplicated by dict key; a should still beat b
    assert result[0] == "a"


def test_rrf_empty_rankings():
    assert reciprocal_rank_fusion([]) == []


def test_rrf_custom_k():
    result = reciprocal_rank_fusion([["x", "y"]], k=1)
    assert result[0] == "x"


# ── hybrid_search — signal composition ───────────────────────────────────────


def test_exact_match_only():
    kg = _kg_with_nodes(("alpha", "Alpha Node"), ("beta", "Beta Node"))
    result = hybrid_search("alpha", kg)
    assert "alpha" in result


def test_bm25_contributes():
    kg = _kg_with_nodes(("n1", "Node One"), ("n2", "Node Two"))
    bm25 = _bm25_stub(["n1", "n2"])
    result = hybrid_search("Node", kg, bm25_index=bm25)
    assert "n1" in result
    assert bm25.search.called


def test_embedding_contributes():
    kg = _kg_with_nodes(("n1", "Something"), ("n2", "Else"))
    embed_fn = MagicMock(return_value=[0.1] * 768)
    store = _embed_store_stub(["n2", "n1"])
    result = hybrid_search("query", kg, embed_fn=embed_fn, embedding_store=store)
    assert "n2" in result
    assert embed_fn.called
    assert store.nearest.called


def test_all_three_signals_fused():
    kg = _kg_with_nodes(("a", "MatchMe"), ("b", "Other"), ("c", "Third"))
    bm25 = _bm25_stub(["a", "b"])
    embed_fn = MagicMock(return_value=[0.0] * 768)
    store = _embed_store_stub(["b", "c"])
    result = hybrid_search("MatchMe", kg, bm25_index=bm25, embed_fn=embed_fn, embedding_store=store)
    assert "a" in result
    assert "b" in result


def test_top_k_respected():
    nodes = [(f"n{i}", f"Node {i}") for i in range(20)]
    kg = _kg_with_nodes(*nodes)
    bm25 = _bm25_stub([f"n{i}" for i in range(20)])
    result = hybrid_search("Node", kg, bm25_index=bm25, top_k=5)
    assert len(result) <= 5


def test_no_signals_returns_empty():
    kg = KnowledgeGraph()
    bm25 = MagicMock()
    bm25.is_ready = False
    result = hybrid_search("missing", kg, bm25_index=bm25)
    assert result == []


def test_embedding_failure_degrades_gracefully():
    kg = _kg_with_nodes(("x", "X Node"))
    embed_fn = MagicMock(side_effect=RuntimeError("embed error"))
    store = _embed_store_stub(["x"])
    result = hybrid_search("X", kg, embed_fn=embed_fn, embedding_store=store)
    # exact match still works; embedding failure is silently skipped
    assert "x" in result


# ── hybrid_search — namespace filter ─────────────────────────────────────────


def test_namespace_filter():
    kg = KnowledgeGraph()
    kg.add_node(Node(id="n1", label="Alpha", kind="note", namespace="nimbus"))
    kg.add_node(Node(id="n2", label="Alpha", kind="note", namespace="code"))
    result = hybrid_search("Alpha", kg, namespace="nimbus")
    assert "n1" in result
    assert "n2" not in result
