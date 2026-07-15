"""Pluggable source extractors for the PrismRag ingest pipeline.

Three source kinds:
  - docs   — Obsidian vault markdown (excludes knowledge/ files)
  - code   — Tree-sitter Python AST
  - memory — Agent memory files (MEMORY.md, session logs); requires atomize

Plus a standalone KNOT loader for knowledge/*.md files (atomize products).
"""

from prism_rag.sources.base import SourceExtractor, SourceKind
from prism_rag.sources.code_source import CodeSourceExtractor
from prism_rag.sources.docs_source import DocsSourceExtractor
from prism_rag.sources.knot_loader import KnotLoader
from prism_rag.sources.memory_source import MemorySourceExtractor

__all__ = [
    "SourceExtractor",
    "SourceKind",
    "CodeSourceExtractor",
    "DocsSourceExtractor",
    "KnotLoader",
    "MemorySourceExtractor",
]
