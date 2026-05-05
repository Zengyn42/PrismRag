from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_rag.inbox.store import InboxEntry, InboxStore


def _entry(eid="e1", status="pending", confidence=0.74) -> InboxEntry:
    return InboxEntry(
        id=eid,
        source="nimbus::doc",
        target="code::a.py::Foo",
        edge_kind="mentions_symbol",
        confidence=confidence,
        confidence_tier="INFERRED",
        model_id="bge-m3",
        probe_signals=[{"kind": "embedding_similar", "score": 0.74,
                        "consecutive_seen": 2, "first_seen_at": "x", "last_seen_at": "x"}],
        top_k_rank=1,
        status=status,
        created_at="2026-05-04T00:00:00Z",
    )


def test_inbox_empty_load(tmp_path: Path):
    s = InboxStore(tmp_path / "inbox.jsonl")
    assert s.list_pending() == []


def test_inbox_append_and_get(tmp_path: Path):
    s = InboxStore(tmp_path / "inbox.jsonl")
    s.append(_entry("e1"))
    s.save_atomic()
    s2 = InboxStore(tmp_path / "inbox.jsonl")
    assert s2.get("e1") is not None
    assert s2.get("e1")["status"] == "pending"


def test_inbox_save_is_ndjson(tmp_path: Path):
    s = InboxStore(tmp_path / "inbox.jsonl")
    s.append(_entry("e1"))
    s.append(_entry("e2"))
    s.save_atomic()
    text = (tmp_path / "inbox.jsonl").read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 2
    for ln in lines:
        json.loads(ln)   # each line is valid JSON


def test_inbox_list_pending_filters(tmp_path: Path):
    s = InboxStore(tmp_path / "inbox.jsonl")
    s.append(_entry("e1", status="pending", confidence=0.8))
    s.append(_entry("e2", status="approved", confidence=0.9))
    s.append(_entry("e3", status="pending", confidence=0.7))
    pending = s.list_pending(top_n=10, sort_by="confidence")
    assert [p["id"] for p in pending] == ["e1", "e3"]
