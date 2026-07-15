"""Docs source extractor — Obsidian vault markdown, excluding KNOT files.

Wraps ObsidianParser but filters out:
  - Files under knowledge/ directories
  - Files with knowledge_id in frontmatter (those are KNOT nodes)

KNOT (Knowledge Ontology Token) files are loaded separately by KnotLoader.
"""

from __future__ import annotations

from pathlib import Path

from prism_rag.ingest.obsidian_parser import ObsidianParser
from prism_rag.ingest.parse_result import ParseResult
from prism_rag.ingest.vault_loader import VaultDocument, discover_markdown_files
from prism_rag.sources.base import SourceExtractor, SourceKind


def _is_knot_file(doc: VaultDocument) -> bool:
    """Return True if this document is a KNOT node (knowledge_id in frontmatter
    or resides under a knowledge/ directory)."""
    if doc.frontmatter.get("knowledge_id"):
        return True
    # Check if any path component is "knowledge"
    return "knowledge" in doc.relative_path.parts


class DocsSourceExtractor(SourceExtractor):
    """Extracts doc nodes from an Obsidian vault, excluding KNOT files."""

    @property
    def kind(self) -> SourceKind:
        return "docs"

    def discover(self, root: Path) -> list[Path]:
        root = root.expanduser().resolve()
        all_md = discover_markdown_files(root)
        # Pre-filter: exclude knowledge/ directories.
        # Full frontmatter filtering happens in parse() after loading.
        return [p for p in all_md if "knowledge" not in p.relative_to(root).parts]

    def parse(self, root: Path) -> ParseResult:
        root = root.expanduser().resolve()
        all_md = discover_markdown_files(root)
        docs = [VaultDocument.from_path(p, root) for p in all_md]
        # Exclude KNOT files (knowledge_id in frontmatter OR under knowledge/ dir)
        docs = [d for d in docs if not _is_knot_file(d)]

        parser = ObsidianParser()
        return parser.parse(docs, vault_root=root)
