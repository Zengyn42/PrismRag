"""label_resolver — KNOW node label resolution with three-layer fallback.

Shared by incremental.py and vault_loader.py. Kept in the ingest layer
(not graph.py) so storage has no dependency on frontmatter / slug formats.

Fallback chain: frontmatter title → clean_slug → stem
"""

from __future__ import annotations


def _clean_slug(stem: str) -> str:
    """Extract readable slug from a KNOW file stem.

    Input:  'KNOW-000043-fresh-per-call-decision'
    Output: 'fresh per call decision'

    split('-', 2) splits into at most 3 parts:
      ['KNOW', '000043', 'fresh-per-call-decision']
    The third part retains internal hyphens, which are then replaced with spaces.

    Returns '' if the stem has fewer than 3 hyphen-separated segments
    (e.g. 'KNOW-000001' with no slug → empty string → fallback to stem).
    """
    parts = stem.split('-', 2)
    return parts[2].replace('-', ' ') if len(parts) > 2 else ''


def resolve_knowledge_label(frontmatter: dict, stem: str) -> str:
    """Resolve a human-readable label for a KNOW node.

    Priority:
      1. frontmatter['title']  — explicit title set by atomize_apply
      2. _clean_slug(stem)     — slug extracted from file name
      3. stem                  — raw file name stem (never empty)

    Args:
        frontmatter: Parsed YAML frontmatter dict from VaultDocument.
        stem:        Path stem of the KNOW file (e.g. 'KNOW-000043-fresh-per-call-decision').

    Returns:
        Non-empty label string.
    """
    return (
        frontmatter.get('title')
        or _clean_slug(stem)
        or stem
    )
