from __future__ import annotations

from pathlib import Path

import pytest

from prism_rag.inbox.approval import apply_decision
from prism_rag.inbox.store import InboxEntry, InboxStore
from prism_rag.store.federated import FederatedGraph
from prism_rag.config import GraphSource
from prism_rag.store.graph import Edge, KnowledgeGraph, LifecycleClass, Node


def _make_fg(tmp_path: Path) -> tuple[FederatedGraph, GraphSource]:
    src = GraphSource(namespace="nimbus", vault_path=tmp_path, data_dir=tmp_path)
    g = KnowledgeGraph()
    g.add_node(Node(id="doc", label="doc"))
    g.save(src.graph_path)
    fg = FederatedGraph.load([src])
    return fg, src


def _make_entry() -> InboxEntry:
    return InboxEntry(
        id="e1", source="nimbus::doc", target="code::a.py::Foo",
        edge_kind="mentions_symbol", confidence=0.74, confidence_tier="INFERRED",
        model_id="bge-m3", probe_signals=[], top_k_rank=1, status="pending",
        created_at="2026-05-04T00:00:00Z",
    )


def test_apply_approve_writes_anchored_edge(tmp_path: Path):
    fg, src = _make_fg(tmp_path)
    inbox = InboxStore(tmp_path / "inbox.jsonl")
    inbox.append(_make_entry())
    inbox.save_atomic()
    apply_decision("e1", "approve", "looks right", inbox=inbox, fg=fg, src=src,
                   decided_by="user_via_tui")
    inbox.save_atomic()
    nimbus = fg.get_graph("nimbus")
    assert nimbus.g.has_edge("doc", "code::a.py::Foo")
    edge_data = nimbus.g.edges["doc", "code::a.py::Foo"]
    assert edge_data["relation"] == "mentions_symbol"
    assert edge_data["lifecycle_class"] == LifecycleClass.ANCHORED
    assert edge_data["confidence_score"] == 0.74
    inbox2 = InboxStore(tmp_path / "inbox.jsonl")
    assert inbox2.get("e1")["status"] == "approved"


def test_apply_reject_no_edge_written(tmp_path: Path):
    fg, src = _make_fg(tmp_path)
    inbox = InboxStore(tmp_path / "inbox.jsonl")
    inbox.append(_make_entry())
    apply_decision("e1", "reject", "not relevant", inbox=inbox, fg=fg, src=src,
                   decided_by="user_via_cli")
    nimbus = fg.get_graph("nimbus")
    assert not nimbus.g.has_edge("doc", "code::a.py::Foo")
    assert inbox.get("e1")["status"] == "rejected"


def test_apply_decision_invalid_value(tmp_path: Path):
    fg, src = _make_fg(tmp_path)
    inbox = InboxStore(tmp_path / "inbox.jsonl")
    inbox.append(_make_entry())
    with pytest.raises(ValueError):
        apply_decision("e1", "maybe", "", inbox=inbox, fg=fg, src=src, decided_by="x")


def test_apply_decision_already_decided(tmp_path: Path):
    fg, src = _make_fg(tmp_path)
    inbox = InboxStore(tmp_path / "inbox.jsonl")
    inbox.append(_make_entry())
    apply_decision("e1", "approve", "", inbox=inbox, fg=fg, src=src, decided_by="x")
    with pytest.raises(Exception):
        apply_decision("e1", "reject", "", inbox=inbox, fg=fg, src=src, decided_by="y")
