"""Tests for atomize_apply_impl — file creation and document patching."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from prism_rag.ingest.atomize import (
    atomize_scan_impl,
    atomize_propose_impl,
    atomize_apply_impl,
    StaleDocError,
)


def _setup_proposal(tmp_path: Path, content: str) -> tuple[Path, dict]:
    """Helper: scan + propose, return (doc_path, proposal_info)."""
    vault = tmp_path / "vault"
    vault.mkdir()
    doc = vault / "source.md"
    doc.write_text(content, encoding="utf-8")

    scan_dir = tmp_path / "scan_cache"
    pending_dir = tmp_path / "pending"

    scan = atomize_scan_impl(doc, vault_root=vault, scan_dir=scan_dir)
    proposal = atomize_propose_impl(
        scan_id=scan["scan_id"],
        claims=[{
            "section_id": scan["sections"][0]["section_id"],
            "knowledge_id": "KNOW-000001",
            "title": "Test Concept",
            "body": "This is a test concept.",
            "ontology_type": "concept",
        }],
        scan_dir=scan_dir,
        proposal_dir=pending_dir,
    )
    return doc, proposal


def test_apply_creates_knowledge_file(tmp_path):
    content = "# Source Doc\n\nIntro text here."
    doc, proposal = _setup_proposal(tmp_path, content)
    vault = tmp_path / "vault"
    applied_dir = tmp_path / "applied"

    result = atomize_apply_impl(
        proposal_id=proposal["proposal_id"],
        vault_root=vault,
        pending_dir=tmp_path / "pending",
        applied_dir=applied_dir,
    )

    knowledge_file = vault / "knowledge" / "KNOW-000001.md"
    assert knowledge_file.exists()
    file_content = knowledge_file.read_text()
    assert "KNOW-000001" in file_content
    assert "Test Concept" in file_content


def test_apply_knowledge_file_has_correct_frontmatter(tmp_path):
    content = "# Source Doc\n\nContent"
    doc, proposal = _setup_proposal(tmp_path, content)
    vault = tmp_path / "vault"
    applied_dir = tmp_path / "applied"

    atomize_apply_impl(
        proposal_id=proposal["proposal_id"],
        vault_root=vault,
        pending_dir=tmp_path / "pending",
        applied_dir=applied_dir,
    )

    knowledge_file = vault / "knowledge" / "KNOW-000001.md"
    text = knowledge_file.read_text()
    assert "knowledge_id: KNOW-000001" in text
    assert "title: Test Concept" in text
    assert "ontology_type: concept" in text
    assert "atomized_from:" in text


def test_apply_patches_source_doc_atomized_nodes(tmp_path):
    content = "# Source Doc\n\nContent"
    doc, proposal = _setup_proposal(tmp_path, content)
    vault = tmp_path / "vault"
    applied_dir = tmp_path / "applied"

    atomize_apply_impl(
        proposal_id=proposal["proposal_id"],
        vault_root=vault,
        pending_dir=tmp_path / "pending",
        applied_dir=applied_dir,
    )

    updated = doc.read_text()
    assert "atomized_nodes" in updated
    assert "KNOW-000001" in updated


def test_apply_moves_proposal_to_applied(tmp_path):
    content = "# Source Doc\n\nContent"
    doc, proposal = _setup_proposal(tmp_path, content)
    vault = tmp_path / "vault"
    applied_dir = tmp_path / "applied"
    pending_dir = tmp_path / "pending"

    atomize_apply_impl(
        proposal_id=proposal["proposal_id"],
        vault_root=vault,
        pending_dir=pending_dir,
        applied_dir=applied_dir,
    )

    assert not (pending_dir / f"{proposal['proposal_id']}.json").exists()
    assert (applied_dir / f"{proposal['proposal_id']}.json").exists()


def test_apply_raises_on_stale_doc(tmp_path):
    content = "# Source Doc\n\nContent"
    doc, proposal = _setup_proposal(tmp_path, content)
    vault = tmp_path / "vault"

    # Modify the source doc after proposing
    doc.write_text(content + "\n\nModified after proposal.", encoding="utf-8")

    with pytest.raises(StaleDocError):
        atomize_apply_impl(
            proposal_id=proposal["proposal_id"],
            vault_root=vault,
            pending_dir=tmp_path / "pending",
            applied_dir=tmp_path / "applied",
        )
