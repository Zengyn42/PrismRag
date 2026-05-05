"""Tests for prism-rag migrate-lifecycle (Sprint 1 protocol)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from prism_rag.cli import app


def _write_graph(path: Path, edges: list[dict]) -> None:
    payload = {
        "metadata": {"version": "v4.0", "node_count": 0, "edge_count": len(edges), "community_count": 0},
        "nodes": [],
        "edges": edges,
        "communities": [],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def test_migrate_lifecycle_classifies_v51a_edges_deterministic(tmp_path):
    g = tmp_path / "graph.json"
    _write_graph(g, [
        {"source": "doc1", "target": "code::a.py::Foo", "relation": "mentions_symbol",
         "confidence": "EXTRACTED", "confidence_score": 1.0, "weight": 1.0, "source_pass": "ast"},
    ])
    runner = CliRunner()
    result = runner.invoke(app, ["migrate-lifecycle", "--graph-path", str(g)])
    assert result.exit_code == 0, result.output
    after = json.loads(g.read_text())
    assert after["edges"][0]["lifecycle_class"] == "deterministic"


def test_migrate_lifecycle_classifies_v52_edges_anchored(tmp_path):
    g = tmp_path / "graph.json"
    _write_graph(g, [
        {"source": "doc1", "target": "code::a.py::Foo", "relation": "mentions_symbol",
         "confidence": "INFERRED", "confidence_score": 0.74, "weight": 1.0, "source_pass": "conv"},
    ])
    runner = CliRunner()
    result = runner.invoke(app, ["migrate-lifecycle", "--graph-path", str(g)])
    assert result.exit_code == 0
    after = json.loads(g.read_text())
    assert after["edges"][0]["lifecycle_class"] == "anchored"


def test_migrate_lifecycle_hard_fails_on_unknown_source(tmp_path):
    g = tmp_path / "graph.json"
    _write_graph(g, [
        {"source": "doc1", "target": "code::a.py::Foo", "relation": "mentions_symbol",
         "confidence": "INFERRED", "confidence_score": 0.5, "weight": 1.0, "source_pass": "mystery"},
    ])
    original = g.read_text()
    runner = CliRunner()
    result = runner.invoke(app, ["migrate-lifecycle", "--graph-path", str(g)])
    assert result.exit_code != 0
    assert "blocked" in result.output.lower() or "blocked" in (result.stderr or "").lower()
    assert g.read_text() == original   # rollback


def test_migrate_lifecycle_idempotent(tmp_path):
    g = tmp_path / "graph.json"
    _write_graph(g, [
        {"source": "doc1", "target": "code::a.py::Foo", "relation": "mentions_symbol",
         "confidence": "EXTRACTED", "confidence_score": 1.0, "weight": 1.0,
         "source_pass": "ast", "lifecycle_class": "deterministic"},
    ])
    runner = CliRunner()
    runner.invoke(app, ["migrate-lifecycle", "--graph-path", str(g)])
    result = runner.invoke(app, ["migrate-lifecycle", "--graph-path", str(g)])
    assert result.exit_code == 0
    assert "0 migrated" in result.output or "already aligned" in result.output.lower()


def test_migrate_lifecycle_skips_non_mentions_edges(tmp_path):
    g = tmp_path / "graph.json"
    _write_graph(g, [
        {"source": "x", "target": "y", "relation": "links_to",
         "confidence": "EXTRACTED", "confidence_score": 1.0, "weight": 1.0, "source_pass": "ast"},
    ])
    runner = CliRunner()
    result = runner.invoke(app, ["migrate-lifecycle", "--graph-path", str(g)])
    assert result.exit_code == 0
    after = json.loads(g.read_text())
    assert "lifecycle_class" not in after["edges"][0] or \
        after["edges"][0]["lifecycle_class"] == "probabilistic"
