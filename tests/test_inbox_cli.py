from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from prism_rag.cli import app
from prism_rag.config import PrismRagSettings


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(PrismRagSettings, "data_dir",
                        property(lambda self: tmp_path), raising=False)
    inbox_path = tmp_path / "inbox.jsonl"
    entries = [
        {"id": "e1", "source": "nimbus::doc1", "target": "code::a.py::Foo",
         "edge_kind": "mentions_symbol", "confidence": 0.80, "confidence_tier": "INFERRED",
         "model_id": "bge-m3", "probe_signals": [], "top_k_rank": 1,
         "status": "pending", "created_at": "2026-05-04T00:00:00Z",
         "decided_at": None, "decided_by": None, "decision_note": None},
        {"id": "e2", "source": "nimbus::doc2", "target": "code::b.py::Bar",
         "edge_kind": "mentions_symbol", "confidence": 0.72, "confidence_tier": "INFERRED",
         "model_id": "bge-m3", "probe_signals": [], "top_k_rank": 1,
         "status": "approved", "created_at": "2026-05-04T00:00:00Z",
         "decided_at": "x", "decided_by": "u", "decision_note": "ok"},
    ]
    inbox_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return tmp_path


def test_inbox_list_pending_default(env):
    runner = CliRunner()
    result = runner.invoke(app, ["inbox"])
    assert result.exit_code == 0
    assert "e1" in result.output
    assert "e2" not in result.output    # approved hidden by default


def test_inbox_list_status_approved(env):
    runner = CliRunner()
    result = runner.invoke(app, ["inbox", "--status", "approved"])
    assert result.exit_code == 0
    assert "e2" in result.output


def test_inbox_show(env):
    runner = CliRunner()
    result = runner.invoke(app, ["inbox", "show", "e1"])
    assert result.exit_code == 0
    assert "e1" in result.output
    assert "0.8" in result.output


def test_inbox_approve(env):
    runner = CliRunner()
    result = runner.invoke(app, ["inbox", "approve", "e1", "--note", "ok"])
    assert result.exit_code == 0
    text = (env / "inbox.jsonl").read_text()
    assert '"status": "approved"' in text or '"status":"approved"' in text


def test_inbox_reject(env):
    runner = CliRunner()
    result = runner.invoke(app, ["inbox", "reject", "e1", "--note", "no"])
    assert result.exit_code == 0


def test_inbox_approve_all_min_conf(env):
    runner = CliRunner()
    result = runner.invoke(app, ["inbox", "approve-all", "--min-conf", "0.75", "--yes"])
    assert result.exit_code == 0
    # e1 has conf 0.80 → approved; e2 already approved
    text = (env / "inbox.jsonl").read_text()
    lines = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    by_id = {l["id"]: l for l in lines}
    assert by_id["e1"]["status"] == "approved"
