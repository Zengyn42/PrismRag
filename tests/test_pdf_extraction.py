"""Tests for Pass 2 PDF media extraction (Section 2)."""
from __future__ import annotations

from pathlib import Path

from prism_rag.ingest.vault_loader import VaultMedia, discover_vault_files


def test_vault_media_id_from_path(tmp_path):
    p = tmp_path / "docs" / "report.pdf"
    p.parent.mkdir()
    p.write_bytes(b"%PDF-1.4\n...")
    media = VaultMedia.from_path(p, tmp_path)
    assert media.id == "docs/report"
    assert media.path == p
    assert media.kind == "pdf"


def test_discover_vault_files_returns_md_and_pdf(tmp_path):
    (tmp_path / "note.md").write_text("# note")
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4\n...")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")

    files = discover_vault_files(tmp_path)
    suffixes = sorted(f.suffix for f in files)
    assert ".md" in suffixes
    assert ".pdf" in suffixes
    # Images not yet supported (MVP = PDF only)
    assert ".png" not in suffixes
