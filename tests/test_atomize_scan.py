"""Tests for atomize_scan_impl — document structure reader."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from prism_rag.ingest.atomize import atomize_scan_impl, ScanExpiredError


def _write_doc(tmp_path: Path, content: str, name: str = "design.md") -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def test_scan_returns_sections(tmp_path):
    content = textwrap.dedent("""\
        ---
        title: Design Doc
        ---
        # Design Doc

        Intro paragraph.

        ## Architecture

        Some architecture text.

        ## Implementation

        Implementation details.
    """)
    doc_path = _write_doc(tmp_path, content)
    scan_dir = tmp_path / "scan_cache"
    result = atomize_scan_impl(doc_path, vault_root=tmp_path, scan_dir=scan_dir)
    sections = result["sections"]
    assert len(sections) >= 2
    headings = [s["heading"] for s in sections]
    assert any("Architecture" in h for h in headings)
    assert any("Implementation" in h for h in headings)


def test_scan_returns_scan_id(tmp_path):
    doc_path = _write_doc(tmp_path, "# Title\n\nContent")
    scan_dir = tmp_path / "scan_cache"
    result = atomize_scan_impl(doc_path, vault_root=tmp_path, scan_dir=scan_dir)
    assert "scan_id" in result
    assert len(result["scan_id"]) > 8  # UUID-like


def test_scan_returns_doc_sha(tmp_path):
    doc_path = _write_doc(tmp_path, "# Title\n\nContent")
    scan_dir = tmp_path / "scan_cache"
    result = atomize_scan_impl(doc_path, vault_root=tmp_path, scan_dir=scan_dir)
    assert "doc_sha" in result
    assert result["doc_sha"].startswith("sha256:")


def test_scan_does_not_return_content_snapshot(tmp_path):
    """content_snapshot stays server-side — must NOT be returned to LLM."""
    doc_path = _write_doc(tmp_path, "# Title\n\nSecret content")
    scan_dir = tmp_path / "scan_cache"
    result = atomize_scan_impl(doc_path, vault_root=tmp_path, scan_dir=scan_dir)
    result_str = json.dumps(result)
    assert "Secret content" not in result_str
    assert "content_snapshot" not in result_str


def test_scan_stores_content_snapshot_in_cache(tmp_path):
    """content_snapshot must be written to scan_cache dir."""
    doc_path = _write_doc(tmp_path, "# Title\n\nSecret content")
    scan_dir = tmp_path / "scan_cache"
    result = atomize_scan_impl(doc_path, vault_root=tmp_path, scan_dir=scan_dir)
    scan_id = result["scan_id"]
    cache_file = scan_dir / f"{scan_id}.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text())
    # At least one section should have content snapshot
    assert any("content_snapshot" in s for s in cached.get("sections", []))


def test_scan_section_ids_are_unique(tmp_path):
    content = textwrap.dedent("""\
        # Title

        Intro

        ## Part A

        Content A

        ## Part B

        Content B
    """)
    doc_path = _write_doc(tmp_path, content)
    scan_dir = tmp_path / "scan_cache"
    result = atomize_scan_impl(doc_path, vault_root=tmp_path, scan_dir=scan_dir)
    section_ids = [s["section_id"] for s in result["sections"]]
    assert len(section_ids) == len(set(section_ids))


def test_scan_section_has_token_estimate(tmp_path):
    doc_path = _write_doc(tmp_path, "# Title\n\nContent here\n\n## Section B\n\nMore content")
    scan_dir = tmp_path / "scan_cache"
    result = atomize_scan_impl(doc_path, vault_root=tmp_path, scan_dir=scan_dir)
    for section in result["sections"]:
        assert "token_estimate" in section
        assert isinstance(section["token_estimate"], int)
        assert section["token_estimate"] >= 0
