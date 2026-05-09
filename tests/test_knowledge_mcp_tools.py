"""Tests for alloc_knowledge_id and list_knowledge_nodes MCP tools."""
from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import prism_rag.mcp_server.server as mcp_server


KNOW_ID_PATTERN = re.compile(r"^KNOW-\d{6,}$")


@pytest.fixture(autouse=True)
def reset_federated():
    """Reset global server state between tests."""
    mcp_server._federated = None
    mcp_server._bm25_indices = {}
    mcp_server._embedding_stores = {}
    mcp_server._embedder = None
    yield
    mcp_server._federated = None


# ─── alloc_knowledge_id ───────────────────────────────────────────────────────

def test_alloc_knowledge_id_returns_valid_ids(tmp_path):
    from prism_rag.config import PrismRagSettings
    settings = MagicMock(spec=PrismRagSettings)
    settings.data_dir = tmp_path

    with patch("prism_rag.mcp_server.server.PrismRagSettings", return_value=settings):
        result = json.loads(mcp_server.alloc_knowledge_id(count=3))

    assert "ids" in result
    assert len(result["ids"]) == 3
    for kid in result["ids"]:
        assert KNOW_ID_PATTERN.match(kid), f"Bad format: {kid!r}"


def test_alloc_knowledge_id_default_count_is_one(tmp_path):
    from prism_rag.config import PrismRagSettings
    settings = MagicMock(spec=PrismRagSettings)
    settings.data_dir = tmp_path

    with patch("prism_rag.mcp_server.server.PrismRagSettings", return_value=settings):
        result = json.loads(mcp_server.alloc_knowledge_id())

    assert len(result["ids"]) == 1


def test_alloc_knowledge_id_sequential_across_calls(tmp_path):
    from prism_rag.config import PrismRagSettings
    settings = MagicMock(spec=PrismRagSettings)
    settings.data_dir = tmp_path

    with patch("prism_rag.mcp_server.server.PrismRagSettings", return_value=settings):
        r1 = json.loads(mcp_server.alloc_knowledge_id(count=2))
        r2 = json.loads(mcp_server.alloc_knowledge_id(count=2))

    all_ids = r1["ids"] + r2["ids"]
    assert len(set(all_ids)) == 4  # all unique


# ─── list_knowledge_nodes ─────────────────────────────────────────────────────

def _make_fake_federated(tmp_path: Path):
    """Build a minimal FederatedGraph stub with knowledge nodes."""
    from prism_rag.store.graph import KnowledgeGraph, Node
    from prism_rag.store.federated import FederatedGraph
    from prism_rag.config import PrismRagSettings, GraphSource

    kg = KnowledgeGraph()
    kg.add_node(Node(id="KNOW-000001", label="Atomic Concept", kind="knowledge",
                     knowledge_id="KNOW-000001", content="A concept"))
    kg.add_node(Node(id="KNOW-000002", label="Another Fact", kind="knowledge",
                     knowledge_id="KNOW-000002", content="A fact"))
    kg.add_node(Node(id="regular-note", label="Regular Note", kind="note",
                     content="A regular note"))
    graph_path = tmp_path / "graph.json"
    kg.save(graph_path)

    settings = PrismRagSettings(
        graphs=[GraphSource(namespace="nimbus", vault_path=tmp_path, data_dir=tmp_path)]
    )
    fg = FederatedGraph.load(settings.resolved_graphs)
    return fg


def test_list_knowledge_nodes_returns_only_knowledge_kind(tmp_path):
    fg = _make_fake_federated(tmp_path)
    mcp_server._federated = fg

    result = json.loads(mcp_server.list_knowledge_nodes())
    nodes = result["nodes"]
    assert len(nodes) == 2
    ids = {n["id"] for n in nodes}
    assert "KNOW-000001" in ids or any("KNOW-000001" in str(n) for n in nodes)
    assert "regular-note" not in ids


def test_list_knowledge_nodes_with_namespace_filter(tmp_path):
    fg = _make_fake_federated(tmp_path)
    mcp_server._federated = fg

    result = json.loads(mcp_server.list_knowledge_nodes(namespace="nimbus"))
    nodes = result["nodes"]
    assert len(nodes) == 2


def test_list_knowledge_nodes_empty_when_no_knowledge_nodes(tmp_path):
    from prism_rag.store.graph import KnowledgeGraph, Node
    from prism_rag.store.federated import FederatedGraph
    from prism_rag.config import PrismRagSettings, GraphSource

    kg = KnowledgeGraph()
    kg.add_node(Node(id="regular", label="Regular", kind="note"))
    graph_path = tmp_path / "graph.json"
    kg.save(graph_path)

    settings = PrismRagSettings(
        graphs=[GraphSource(namespace="nimbus", vault_path=tmp_path, data_dir=tmp_path)]
    )
    fg = FederatedGraph.load(settings.resolved_graphs)
    mcp_server._federated = fg

    result = json.loads(mcp_server.list_knowledge_nodes())
    assert result["nodes"] == []
