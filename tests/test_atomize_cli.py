"""Tests for CLI atomize group: list, show, apply."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from prism_rag.cli import app


runner = CliRunner()


def _make_fake_proposal(data_dir: Path, vault: Path) -> dict:
    """Create a minimal proposal file for testing."""
    from prism_rag.ingest.atomize import atomize_scan_impl, atomize_propose_impl
    doc = vault / "source.md"
    doc.write_text("# Source\n\nContent", encoding="utf-8")
    scan_dir = data_dir / "atomize-proposals" / "scan_cache"
    pending_dir = data_dir / "atomize-proposals" / "pending"
    scan = atomize_scan_impl(doc, vault_root=vault, scan_dir=scan_dir)
    return atomize_propose_impl(
        scan_id=scan["scan_id"],
        claims=[{
            "section_id": scan["sections"][0]["section_id"],
            "knowledge_id": "KNOW-000001",
            "title": "Test Concept",
            "body": "Body",
            "ontology_type": "concept",
        }],
        scan_dir=scan_dir,
        proposal_dir=pending_dir,
    )


def test_atomize_list_shows_pending_proposals(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    proposal = _make_fake_proposal(tmp_path, vault)

    result = runner.invoke(app, ["atomize", "list", "--output", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert proposal["proposal_id"][:8] in result.output


def test_atomize_list_empty_when_no_proposals(tmp_path):
    result = runner.invoke(app, ["atomize", "list", "--output", str(tmp_path)])
    assert result.exit_code == 0
    assert "No pending" in result.output


def test_atomize_show_displays_claims(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    proposal = _make_fake_proposal(tmp_path, vault)
    pid = proposal["proposal_id"]

    result = runner.invoke(app, ["atomize", "show", pid[:12], "--output", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "KNOW-000001" in result.output
    assert "Test Concept" in result.output


def test_atomize_apply_creates_knowledge_file(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    proposal = _make_fake_proposal(tmp_path, vault)
    pid = proposal["proposal_id"]

    result = runner.invoke(app, [
        "atomize", "apply", pid[:12],
        "--vault", str(vault),
        "--output", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert "Applied" in result.output
    assert (vault / "knowledge" / "KNOW-000001.md").exists()
