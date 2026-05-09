"""Tests for atomize_apply crash recovery (idempotency)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_rag.ingest.atomize import (
    atomize_scan_impl,
    atomize_propose_impl,
    atomize_apply_impl,
)


def _make_proposal(tmp_path: Path, kid: str = "KNOW-000001") -> tuple[Path, dict]:
    vault = tmp_path / "vault"
    vault.mkdir()
    doc = vault / "source.md"
    doc.write_text("# Source\n\nContent here.", encoding="utf-8")
    scan_dir = tmp_path / "scan_cache"
    pending_dir = tmp_path / "pending"
    scan = atomize_scan_impl(doc, vault_root=vault, scan_dir=scan_dir)
    proposal = atomize_propose_impl(
        scan_id=scan["scan_id"],
        claims=[{
            "section_id": scan["sections"][0]["section_id"],
            "knowledge_id": kid,
            "title": "Concept",
            "body": "Body",
            "ontology_type": "concept",
        }],
        scan_dir=scan_dir,
        proposal_dir=pending_dir,
    )
    return doc, proposal


def test_apply_idempotent_if_knowledge_file_already_exists(tmp_path):
    """If KNOW-ID already in atomized_nodes, re-apply should not raise or duplicate."""
    doc, proposal = _make_proposal(tmp_path)
    vault = tmp_path / "vault"
    pending_dir = tmp_path / "pending"
    applied_dir = tmp_path / "applied"

    # First apply
    atomize_apply_impl(
        proposal_id=proposal["proposal_id"],
        vault_root=vault,
        pending_dir=pending_dir,
        applied_dir=applied_dir,
    )

    # Restore proposal file to pending (simulate crash after first apply)
    (applied_dir / f"{proposal['proposal_id']}.json").rename(
        pending_dir / f"{proposal['proposal_id']}.json"
    )

    # Second apply — should not fail even though knowledge file exists
    atomize_apply_impl(
        proposal_id=proposal["proposal_id"],
        vault_root=vault,
        pending_dir=pending_dir,
        applied_dir=applied_dir,
    )

    # File should still exist and be correct
    knowledge_file = vault / "knowledge" / "KNOW-000001.md"
    assert knowledge_file.exists()
