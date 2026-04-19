"""Tests for vault_ops write path (Section 6)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_rag.vault_ops.cas import (
    CASConflict,
    atomic_write,
    compute_hash,
    write_with_cas,
)


def test_atomic_write_creates_file(tmp_path):
    target = tmp_path / "sub" / "note.md"
    target.parent.mkdir()
    atomic_write(target, "hello")
    assert target.read_text() == "hello"


def test_atomic_write_leaves_no_tmp(tmp_path):
    target = tmp_path / "note.md"
    atomic_write(target, "hello")
    tmps = list(tmp_path.glob("*.tmp"))
    assert tmps == []


def test_write_with_cas_fresh_file(tmp_path):
    target = tmp_path / "new.md"
    new_hash = write_with_cas(target, "hello", expected_hash=None)
    assert target.read_text() == "hello"
    assert new_hash == compute_hash("hello")


def test_write_with_cas_matching_hash(tmp_path):
    target = tmp_path / "existing.md"
    target.write_text("v1")
    v1_hash = compute_hash("v1")
    new_hash = write_with_cas(target, "v2", expected_hash=v1_hash)
    assert target.read_text() == "v2"
    assert new_hash == compute_hash("v2")


def test_write_with_cas_conflict_raises(tmp_path):
    target = tmp_path / "existing.md"
    target.write_text("v1")
    with pytest.raises(CASConflict) as exc_info:
        write_with_cas(target, "v2", expected_hash="sha256:deadbeef")
    # The exception message or .expected should contain our wrong hash
    assert "deadbeef" in str(exc_info.value) or exc_info.value.expected == "sha256:deadbeef"


def test_audit_log_appends_jsonl(tmp_path, monkeypatch):
    """log_operation writes one JSONL line per call to the audit path."""
    from prism_rag.vault_ops import audit_log as al

    monkeypatch.setattr(al, "_audit_path", lambda: tmp_path / "audit.jsonl")

    al.log_operation(tool="write_note", target="foo.md", action="write",
                     status="ok", cas_before="sha256:a", cas_after="sha256:b")
    al.log_operation(tool="write_note", target="foo.md", action="write",
                     status="conflict", cas_before="sha256:a", cas_after="sha256:c",
                     error="CASConflict")

    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 2
    r1 = json.loads(lines[0])
    r2 = json.loads(lines[1])
    assert r1["status"] == "ok" and r1["target"] == "foo.md"
    assert r2["status"] == "conflict" and r2["error"] == "CASConflict"
    # ISO-8601 timestamp with timezone
    assert "T" in r1["ts"]


@pytest.mark.asyncio
async def test_write_note_logs_audit_on_success(tmp_path, monkeypatch):
    """Successful write_note produces audit entry with status='ok'."""
    from prism_rag.config import PrismRagSettings, GraphSource
    from prism_rag.vault_ops import audit_log as al

    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()

    # Seed empty graph
    from prism_rag.store.graph import KnowledgeGraph
    KnowledgeGraph().save(data / "graph.json")

    audit_path = data / "audit.jsonl"
    monkeypatch.setattr(al, "_audit_path", lambda: audit_path)

    settings = PrismRagSettings(
        graphs=[GraphSource(namespace="default", vault_path=vault, data_dir=data, writable=True)],
    )
    monkeypatch.setattr(
        "prism_rag.mcp_server.vault_tools.PrismRagSettings",
        lambda: settings,
    )

    from unittest.mock import patch

    def _stub_sync(path, settings, tool_name):
        return {"node_id": "new", "action": "added"}

    from prism_rag.mcp_server.vault_tools import _write_note_impl
    with patch("prism_rag.mcp_server.vault_tools._sync_graph", side_effect=_stub_sync):
        result = await _write_note_impl(
            path="new.md",
            content="# Hello",
            cas_hash="",
            namespace="default",
        )
    parsed = json.loads(result)
    assert parsed.get("status") == "ok"

    assert audit_path.exists()
    entries = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert any(e["tool"] == "write_note" and e["status"] == "ok" for e in entries)


@pytest.mark.asyncio
async def test_write_note_cas_conflict_logs_audit(tmp_path, monkeypatch):
    """CAS conflict produces audit entry with status='conflict'."""
    from prism_rag.config import PrismRagSettings, GraphSource
    from prism_rag.vault_ops import audit_log as al

    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()
    (vault / "existing.md").write_text("v1")

    from prism_rag.store.graph import KnowledgeGraph
    KnowledgeGraph().save(data / "graph.json")

    audit_path = data / "audit.jsonl"
    monkeypatch.setattr(al, "_audit_path", lambda: audit_path)

    settings = PrismRagSettings(
        graphs=[GraphSource(namespace="default", vault_path=vault, data_dir=data, writable=True)],
    )
    monkeypatch.setattr(
        "prism_rag.mcp_server.vault_tools.PrismRagSettings",
        lambda: settings,
    )

    from prism_rag.mcp_server.vault_tools import _write_note_impl
    result = await _write_note_impl(
        path="existing.md",
        content="v2",
        cas_hash="sha256:wronghash",
        namespace="default",
    )
    parsed = json.loads(result)
    assert parsed.get("status") != "ok"

    entries = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert any(e["status"] == "conflict" for e in entries)
