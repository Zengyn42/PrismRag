from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from prism_rag.config import GraphSource, PrismRagSettings
from prism_rag.inbox.store import InboxEntry, InboxStore
from prism_rag.mcp_server import server as mcp_server
from prism_rag.store.federated import FederatedGraph
from prism_rag.store.graph import KnowledgeGraph, Node


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    src = GraphSource(namespace="nimbus", vault_path=tmp_path, data_dir=tmp_path)
    g = KnowledgeGraph()
    g.add_node(Node(id="doc", label="doc"))
    g.save(src.graph_path)
    inbox = InboxStore(tmp_path / "inbox.jsonl")
    inbox.append(InboxEntry(
        id="e1", source="nimbus::doc", target="code::a.py::Foo",
        edge_kind="mentions_symbol", confidence=0.74, confidence_tier="INFERRED",
        model_id="bge-m3", probe_signals=[{"kind": "embedding_similar", "score": 0.74,
                                            "consecutive_seen": 2,
                                            "first_seen_at": "x", "last_seen_at": "x"}],
        top_k_rank=1, status="pending", created_at="2026-05-04T00:00:00Z",
    ))
    inbox.save_atomic()
    settings = PrismRagSettings(graphs=[src])
    monkeypatch.setattr(mcp_server, "_federated", None)
    monkeypatch.setattr("prism_rag.mcp_server.server.PrismRagSettings", lambda: settings)
    return tmp_path


def test_list_pending_edges_returns_pending(env):
    out = mcp_server.list_pending_edges(top_n=10)
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert any(e["id"] == "e1" for e in parsed)
    assert parsed[0]["status"] == "pending"


def test_get_pending_edge_context(env):
    out = mcp_server.get_pending_edge_context("e1")
    parsed = json.loads(out)
    assert "vault_context" in parsed
    assert "code_context" in parsed
    assert parsed["edge_id"] == "e1"


def test_get_pending_edge_context_unknown(env):
    out = mcp_server.get_pending_edge_context("does-not-exist")
    parsed = json.loads(out)
    assert parsed.get("status") == "error"


def test_review_approve(env):
    out = mcp_server.review_pending_edge("e1", "approve", "looks right")
    parsed = json.loads(out)
    assert parsed["status"] == "ok"
    assert parsed["decision"] == "approve"
    # InboxStore now has approved status
    inbox2 = InboxStore(env / "inbox.jsonl")
    assert inbox2.get("e1")["status"] == "approved"


def test_review_reject(env):
    out = mcp_server.review_pending_edge("e1", "reject", "no")
    parsed = json.loads(out)
    assert parsed["status"] == "ok"
    assert parsed["decision"] == "reject"


def test_review_invalid_decision(env):
    out = mcp_server.review_pending_edge("e1", "maybe", "")
    parsed = json.loads(out)
    assert parsed["status"] == "error"


def test_review_already_decided(env):
    mcp_server.review_pending_edge("e1", "approve", "first")
    out = mcp_server.review_pending_edge("e1", "reject", "second")
    parsed = json.loads(out)
    assert parsed["status"] == "error"


def test_review_unknown_id(env):
    out = mcp_server.review_pending_edge("nope", "approve", "")
    parsed = json.loads(out)
    assert parsed["status"] == "error"
