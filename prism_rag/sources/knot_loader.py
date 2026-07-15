"""KNOT loader — loads knowledge/*.md files with knowledge_id frontmatter.

KNOT (Knowledge Ontology Token) files are the product of atomize. They have
a knowledge_id (KNOW-NNNNNN) in their frontmatter and live under knowledge/
directories in the vault.

This loader finds and parses them into the graph as kind='knowledge' nodes,
preserving existing behavior from ObsidianParser.

File prefix KNOW- kept for data compat.
"""

from __future__ import annotations

from pathlib import Path

from prism_rag.ingest.obsidian_parser import ObsidianParser
from prism_rag.ingest.parse_result import ParseResult
from prism_rag.ingest.vault_loader import VaultDocument, discover_markdown_files


def _is_knot_file(doc: VaultDocument) -> bool:
    """Return True if this document is a KNOT node."""
    if doc.frontmatter.get("knowledge_id"):
        return True
    return "knowledge" in doc.relative_path.parts


class KnotLoader:
    """Loads KNOT (knowledge) files into the graph.

    KNOT files are identified by:
      - Having knowledge_id in YAML frontmatter, OR
      - Residing under a knowledge/ directory
    """

    def discover(self, root: Path) -> list[Path]:
        """Find all KNOT files under root."""
        root = root.expanduser().resolve()
        all_md = discover_markdown_files(root)
        # Pre-filter: files under knowledge/ directories
        knowledge_paths = [p for p in all_md if "knowledge" in p.relative_to(root).parts]
        # Also need to check frontmatter for files outside knowledge/ that have knowledge_id
        # But that requires reading the file, so we return a superset here
        # and filter more precisely in parse()
        return knowledge_paths

    def parse(self, root: Path) -> ParseResult:
        """Parse KNOT files into a ParseResult.

        Loads all markdown files, filters to only KNOT files (knowledge_id in
        frontmatter or under knowledge/ dir), then parses them via ObsidianParser.
        """
        root = root.expanduser().resolve()
        all_md = discover_markdown_files(root)
        docs = [VaultDocument.from_path(p, root) for p in all_md]
        knot_docs = [d for d in docs if _is_knot_file(d)]

        if not knot_docs:
            # Return empty ParseResult
            return ParseResult(
                nodes=[], edges=[], parser_id="KnotLoader", namespace="nimbus"
            )

        parser = ObsidianParser()
        return parser.parse(knot_docs, vault_root=root)
