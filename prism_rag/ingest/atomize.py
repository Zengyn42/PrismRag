"""atomize_document — three-phase vault document atomization.

Phase 1: atomize_scan    — read structure, cache content snapshots server-side
Phase 2: atomize_propose — LLM submits claims; server writes proposal file
Phase 3: atomize_apply   — create knowledge/*.md, patch source doc, ingest

Content snapshots are stored server-side (scan_cache/<scan_id>.json) to avoid
returning 10k+ tokens to the LLM. Proposals live at
atomize-proposals/pending/<id>.json and move to applied/ on completion.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ScanExpiredError(Exception):
    """Raised when a scan_id is not found or is older than the TTL."""


_SCAN_TTL_HOURS = 24
_HEADING_RE = re.compile(r"^(#{1,2})\s+(.*)", re.MULTILINE)


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_estimate(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(0, len(text) // 4)


def _parse_sections(content: str) -> list[dict[str, Any]]:
    """Split markdown content into sections at H1/H2 headings.

    Returns list of dicts with: section_id, heading, level, start_line, content_snapshot.
    Intro text before first heading is included as section_id=0.
    """
    lines = content.splitlines(keepends=True)
    sections: list[dict[str, Any]] = []
    current_heading = "(intro)"
    current_level = 0
    current_start = 0
    current_lines: list[str] = []

    def _flush(heading: str, level: int, start: int, body_lines: list[str], idx: int) -> None:
        text = "".join(body_lines).strip()
        if text or heading != "(intro)":
            sections.append({
                "section_id": str(idx),
                "heading": heading,
                "level": level,
                "start_line": start,
                "content_snapshot": text,
            })

    section_idx = 0
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            _flush(current_heading, current_level, current_start, current_lines, section_idx)
            section_idx += 1
            current_heading = m.group(2).strip()
            current_level = len(m.group(1))
            current_start = i
            current_lines = []
        else:
            current_lines.append(line)

    _flush(current_heading, current_level, current_start, current_lines, section_idx)
    return sections


def atomize_scan_impl(
    doc_path: Path,
    vault_root: Path,
    scan_dir: Path,
) -> dict[str, Any]:
    """Phase 1: read document structure, cache snapshots, return section metadata.

    Does NOT return content_snapshot to caller — only section headers and estimates.
    """
    doc_path = Path(doc_path).expanduser().resolve()
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    raw = doc_path.read_text(encoding="utf-8")
    doc_sha = _sha256(raw)
    scan_id = str(uuid.uuid4())
    sections_full = _parse_sections(raw)

    # Build what gets cached (includes content_snapshot)
    cached_sections = [
        {
            "section_id": s["section_id"],
            "heading": s["heading"],
            "level": s["level"],
            "start_line": s["start_line"],
            "content_snapshot": s["content_snapshot"],
            "token_estimate": _token_estimate(s["content_snapshot"]),
        }
        for s in sections_full
    ]

    # Persist to scan cache
    scan_dir = Path(scan_dir)
    scan_dir.mkdir(parents=True, exist_ok=True)
    cache_entry = {
        "scan_id": scan_id,
        "doc_path": str(doc_path),
        "doc_sha": doc_sha,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "sections": cached_sections,
    }
    (scan_dir / f"{scan_id}.json").write_text(
        json.dumps(cache_entry, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Return to caller: no content_snapshot
    public_sections = [
        {
            "section_id": s["section_id"],
            "heading": s["heading"],
            "level": s["level"],
            "start_line": s["start_line"],
            "token_estimate": s["token_estimate"],
        }
        for s in cached_sections
    ]

    return {
        "scan_id": scan_id,
        "doc_path": str(doc_path.relative_to(vault_root) if doc_path.is_relative_to(vault_root) else doc_path),
        "doc_sha": doc_sha,
        "section_count": len(public_sections),
        "sections": public_sections,
    }
