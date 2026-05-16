"""Tests for PrismRag v5.4 polish features (P2, P3, P4).

P2 — KNOW-ID tool routing: search_knowledge soft hint + explain_node smart error.
P3 — list_knowledge_nodes body_preview + max_results.
P4 — resolve_knowledge_label three-layer fallback.

Fixture design (per design doc D6):
  - Soft hint tests (P2): NO FederatedGraph needed — hint is returned before
    _ensure_federated() is called (pure input validation).
  - Mixed query / explain_node tests: require FederatedGraph with KNOW nodes.
  - P3 body_preview tests: require FederatedGraph with nodes carrying content.
  - P4 resolver tests: pure functions, NO fixture needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import prism_rag.mcp_server.server as mcp_server


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_federated():
    """Reset global server state between tests."""
    mcp_server._federated = None
    mcp_server._bm25_indices = {}
    mcp_server._embedding_stores = {}
    mcp_server._embedder = None
    yield
    mcp_server._federated = None


def _make_fake_federated(tmp_path: Path, extra_nodes=None):
    """Build a minimal FederatedGraph stub with knowledge nodes + content."""
    from prism_rag.store.graph import KnowledgeGraph, Node
    from prism_rag.store.federated import FederatedGraph
    from prism_rag.config import PrismRagSettings, GraphSource

    kg = KnowledgeGraph()
    # Standard KNOW nodes with content and knowledge_id
    kg.add_node(Node(
        id="KNOW-000008", label="Fresh Per Call Decision", kind="knowledge",
        knowledge_id="KNOW-000008",
        content="This node describes the fresh-per-call design decision in detail.",
    ))
    kg.add_node(Node(
        id="KNOW-000001", label="Atomic Concept", kind="knowledge",
        knowledge_id="KNOW-000001",
        content="A" * 51,  # 51 chars — for truncation tests
    ))
    kg.add_node(Node(
        id="KNOW-000002", label="Empty Content Node", kind="knowledge",
        knowledge_id="KNOW-000002",
        content="",  # empty content — for preview_empty_ok test
    ))
    kg.add_node(Node(
        id="regular-note", label="Regular Note", kind="note",
        content="A regular note",
    ))
    if extra_nodes:
        for n in extra_nodes:
            kg.add_node(n)

    graph_path = tmp_path / "graph.json"
    kg.save(graph_path)
    settings = PrismRagSettings(
        graphs=[GraphSource(namespace="nimbus", vault_path=tmp_path, data_dir=tmp_path)]
    )
    return FederatedGraph.load(settings.resolved_graphs)


# ===========================================================================
# P2 — search_knowledge soft hint (NO FederatedGraph needed)
# ===========================================================================

def test_search_knowledge_pure_id_6digit_hint():
    """Pure 6-digit KNOW-ID query must return soft hint, not search results."""
    result = json.loads(mcp_server.search_knowledge("KNOW-000008"))
    assert "hint" in result
    assert result["results"] == []
    assert "⚠️" in result["hint"]
    assert "explain_node" in result["hint"]
    assert "KNOW-000008" in result["hint"]


def test_search_knowledge_pure_id_case_insensitive():
    """Lowercase KNOW-ID must also trigger hint, and hint must uppercase the ID."""
    result = json.loads(mcp_server.search_knowledge("know-000008"))
    assert "hint" in result
    assert result["results"] == []
    # ID in hint must be uppercased (not 'know-000008')
    assert "KNOW-000008" in result["hint"]
    assert "know-000008" not in result["hint"]


def test_search_knowledge_pure_id_short_hint():
    """Under-6-digit KNOW-ID returns 'format incomplete' hint."""
    result = json.loads(mcp_server.search_knowledge("KNOW-12"))
    assert "hint" in result
    assert result["results"] == []
    assert "⚠️" in result["hint"]
    # Should mention the incomplete format, not direct lookup
    assert "6" in result["hint"] or "不完整" in result["hint"]


def test_search_knowledge_know_no_digits_no_hint(tmp_path):
    """'KNOW-' with no digits does not match regex — no soft hint triggered."""
    # This query will try to search, so we need a federated graph
    fg = _make_fake_federated(tmp_path)
    mcp_server._federated = fg
    result = json.loads(mcp_server.search_knowledge("KNOW-"))
    # Should NOT have a hint key — falls through to normal search
    assert "hint" not in result


def test_search_knowledge_mixed_query_no_hint(tmp_path):
    """Mixed query containing a KNOW-ID must NOT trigger soft hint."""
    fg = _make_fake_federated(tmp_path)
    mcp_server._federated = fg
    result = json.loads(mcp_server.search_knowledge("对比 KNOW-000008 与 database 设计"))
    assert "hint" not in result


# ===========================================================================
# P2 — explain_node smart error response (requires FederatedGraph)
# ===========================================================================

def test_explain_node_knowid_direct(tmp_path):
    """explain_node with a valid KNOW-ID returns the correct node."""
    fg = _make_fake_federated(tmp_path)
    mcp_server._federated = fg
    result = json.loads(mcp_server.explain_node("KNOW-000008"))
    assert "error" not in result
    assert "node" in result
    assert result["node"]["knowledge_id"] == "KNOW-000008"


def test_explain_node_knowid_case_insensitive(tmp_path):
    """explain_node is case-insensitive for KNOW-ID lookup."""
    fg = _make_fake_federated(tmp_path)
    mcp_server._federated = fg
    result = json.loads(mcp_server.explain_node("know-000008"))
    assert "error" not in result
    assert "node" in result


def test_explain_node_mixed_input_error(tmp_path):
    """explain_node with dirty input returns smart error pointing to clean ID."""
    fg = _make_fake_federated(tmp_path)
    mcp_server._federated = fg
    result = json.loads(mcp_server.explain_node("KNOW-000008 的架构"))
    assert "error" in result
    # Error must contain the clean ID call to break ping-pong loops
    assert "explain_node(node='KNOW-000008')" in result["error"]
    assert "额外文本" in result["error"] or "额外" in result["error"]


# ===========================================================================
# P3 — list_knowledge_nodes body_preview + max_results
# ===========================================================================

def test_list_knowledge_nodes_has_body_preview(tmp_path):
    """Every node in list_knowledge_nodes must have body_preview ≤ 50 chars."""
    fg = _make_fake_federated(tmp_path)
    mcp_server._federated = fg
    result = json.loads(mcp_server.list_knowledge_nodes())
    nodes = result["nodes"]
    assert len(nodes) > 0
    for n in nodes:
        assert "body_preview" in n
        assert len(n["body_preview"]) <= 50


def test_list_knowledge_nodes_preview_truncates_at_50(tmp_path):
    """Node with 51-char content must have body_preview of exactly 50 chars."""
    fg = _make_fake_federated(tmp_path)
    mcp_server._federated = fg
    result = json.loads(mcp_server.list_knowledge_nodes())
    # KNOW-000001 has content "A" * 51
    node = next(n for n in result["nodes"] if n["id"] == "KNOW-000001")
    assert len(node["body_preview"]) == 50


def test_list_knowledge_nodes_preview_empty_ok(tmp_path):
    """Node with empty content must have body_preview='' without error."""
    fg = _make_fake_federated(tmp_path)
    mcp_server._federated = fg
    result = json.loads(mcp_server.list_knowledge_nodes())
    node = next(n for n in result["nodes"] if n["id"] == "KNOW-000002")
    assert node["body_preview"] == ""


def test_list_knowledge_nodes_max_results_default(tmp_path):
    """Default max_results=20: returned ≤ 20, total reflects true count."""
    from prism_rag.store.graph import Node
    # Add 25 knowledge nodes
    extra = [
        Node(id=f"KNOW-{i:06d}", label=f"Node {i}", kind="knowledge",
             knowledge_id=f"KNOW-{i:06d}", content=f"content {i}")
        for i in range(100, 125)
    ]
    fg = _make_fake_federated(tmp_path, extra_nodes=extra)
    mcp_server._federated = fg
    result = json.loads(mcp_server.list_knowledge_nodes())
    assert result["returned"] <= 20
    assert result["total"] > 20  # total must be the real count
    assert result["returned"] == len(result["nodes"])


def test_list_knowledge_nodes_max_results_one(tmp_path):
    """max_results=1 returns exactly 1 node, total still reflects full count."""
    fg = _make_fake_federated(tmp_path)
    mcp_server._federated = fg
    result = json.loads(mcp_server.list_knowledge_nodes(max_results=1))
    assert result["returned"] == 1
    assert len(result["nodes"]) == 1
    assert result["total"] >= 1


# ===========================================================================
# P4 — resolve_knowledge_label pure function tests (NO fixture needed)
# ===========================================================================

def test_resolve_knowledge_label_title():
    """frontmatter title takes top priority."""
    from prism_rag.ingest.label_resolver import resolve_knowledge_label
    label = resolve_knowledge_label(
        {"title": "Fresh Per Call Decision"},
        "KNOW-000043-fresh-per-call-decision",
    )
    assert label == "Fresh Per Call Decision"


def test_resolve_knowledge_label_clean_slug():
    """No title → clean_slug extracts readable text from stem."""
    from prism_rag.ingest.label_resolver import resolve_knowledge_label
    label = resolve_knowledge_label(
        {},
        "KNOW-000043-fresh-per-call-decision",
    )
    assert label == "fresh per call decision"


def test_resolve_knowledge_label_fallback_stem():
    """No title and no slug segment → falls back to raw stem."""
    from prism_rag.ingest.label_resolver import resolve_knowledge_label
    label = resolve_knowledge_label(
        {},
        "KNOW-000001",  # only two hyphen-segments, no slug
    )
    assert label == "KNOW-000001"
