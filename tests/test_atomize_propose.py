"""Tests for atomize_propose_impl — proposal creation."""
from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

import pytest

from prism_rag.ingest.atomize import atomize_scan_impl, atomize_propose_impl, ScanExpiredError


def _scan(tmp_path: Path, content: str) -> dict:
    doc = tmp_path / "doc.md"
    doc.write_text(content, encoding="utf-8")
    scan_dir = tmp_path / "scan_cache"
    return atomize_scan_impl(doc, vault_root=tmp_path, scan_dir=scan_dir)


def _make_claim(section_id: str, knowledge_id: str, title: str = "Test") -> dict:
    return {
        "section_id": section_id,
        "knowledge_id": knowledge_id,
        "title": title,
        "body": f"Body text for {title}.",
        "ontology_type": "concept",
    }


def test_propose_creates_pending_proposal(tmp_path):
    content = "# Title\n\nIntro\n\n## Section A\n\nContent A"
    scan = _scan(tmp_path, content)
    proposal_dir = tmp_path / "pending"

    result = atomize_propose_impl(
        scan_id=scan["scan_id"],
        claims=[_make_claim(scan["sections"][0]["section_id"], "KNOW-000001")],
        scan_dir=tmp_path / "scan_cache",
        proposal_dir=proposal_dir,
    )

    assert "proposal_id" in result
    proposal_file = proposal_dir / f"{result['proposal_id']}.json"
    assert proposal_file.exists()


def test_propose_sets_claim_status_pending(tmp_path):
    content = "# Title\n\nIntro\n\n## Section A\n\nContent A"
    scan = _scan(tmp_path, content)
    proposal_dir = tmp_path / "pending"

    result = atomize_propose_impl(
        scan_id=scan["scan_id"],
        claims=[_make_claim(scan["sections"][0]["section_id"], "KNOW-000001")],
        scan_dir=tmp_path / "scan_cache",
        proposal_dir=proposal_dir,
    )

    proposal = json.loads((proposal_dir / f"{result['proposal_id']}.json").read_text())
    for claim in proposal["claims"]:
        assert claim["claim_status"] == "pending"


def test_propose_includes_doc_sha(tmp_path):
    content = "# Title\n\nContent"
    scan = _scan(tmp_path, content)
    proposal_dir = tmp_path / "pending"

    result = atomize_propose_impl(
        scan_id=scan["scan_id"],
        claims=[_make_claim(scan["sections"][0]["section_id"], "KNOW-000001")],
        scan_dir=tmp_path / "scan_cache",
        proposal_dir=proposal_dir,
    )

    proposal = json.loads((proposal_dir / f"{result['proposal_id']}.json").read_text())
    assert "doc_sha" in proposal
    assert proposal["doc_sha"] == scan["doc_sha"]


def test_propose_deduplicates_by_knowledge_id(tmp_path):
    content = "# Title\n\nA\n\n## B\n\nB"
    scan = _scan(tmp_path, content)
    proposal_dir = tmp_path / "pending"
    sid = scan["sections"][0]["section_id"]

    result = atomize_propose_impl(
        scan_id=scan["scan_id"],
        claims=[
            _make_claim(sid, "KNOW-000001", "First"),
            _make_claim(sid, "KNOW-000001", "Duplicate"),  # same KNOW-ID
        ],
        scan_dir=tmp_path / "scan_cache",
        proposal_dir=proposal_dir,
    )

    proposal = json.loads((proposal_dir / f"{result['proposal_id']}.json").read_text())
    know_ids = [c["knowledge_id"] for c in proposal["claims"]]
    assert know_ids.count("KNOW-000001") == 1  # deduplicated


def test_propose_raises_on_unknown_scan_id(tmp_path):
    proposal_dir = tmp_path / "pending"
    with pytest.raises(ScanExpiredError):
        atomize_propose_impl(
            scan_id="nonexistent-scan-id",
            claims=[],
            scan_dir=tmp_path / "scan_cache",
            proposal_dir=proposal_dir,
        )


def test_propose_raises_on_expired_scan(tmp_path, monkeypatch):
    content = "# Title\n\nContent"
    scan = _scan(tmp_path, content)
    proposal_dir = tmp_path / "pending"

    # Manually set scanned_at to 25 hours ago
    cache_file = tmp_path / "scan_cache" / f"{scan['scan_id']}.json"
    cached = json.loads(cache_file.read_text())
    from datetime import datetime, timezone, timedelta
    old_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    cached["scanned_at"] = old_time
    cache_file.write_text(json.dumps(cached))

    with pytest.raises(ScanExpiredError):
        atomize_propose_impl(
            scan_id=scan["scan_id"],
            claims=[_make_claim(scan["sections"][0]["section_id"], "KNOW-000001")],
            scan_dir=tmp_path / "scan_cache",
            proposal_dir=proposal_dir,
        )
