"""Memory source extractor — agent memory files (MEMORY.md, session logs).

Memory files MUST go through atomize to produce KNOT nodes before entering the
graph. Raw memory content does not become graph nodes directly.

Since atomize is currently disabled (feature paused by boss), enabling the memory
source will raise a clear error rather than silently skipping.

KNOT (Knowledge Ontology Token), file prefix KNOW- kept for data compat.
"""

from __future__ import annotations

from pathlib import Path

from prism_rag.ingest.parse_result import ParseResult
from prism_rag.sources.base import SourceExtractor, SourceKind


class AtomizeUnavailableError(Exception):
    """Raised when memory source is enabled but atomize is not available."""


class MemorySourceExtractor(SourceExtractor):
    """Extracts knowledge from agent memory files via atomize.

    Since atomize is currently disabled, this extractor raises a clear error
    when invoked, rather than silently skipping.
    """

    def __init__(self, memory_paths: list[Path] | None = None):
        self._memory_paths = memory_paths or []

    @property
    def kind(self) -> SourceKind:
        return "memory"

    def discover(self, root: Path) -> list[Path]:
        """Discover markdown files from configured memory_paths."""
        results: list[Path] = []
        for mp in self._memory_paths:
            mp = mp.expanduser().resolve()
            if mp.is_file() and mp.suffix == ".md":
                results.append(mp)
            elif mp.is_dir():
                results.extend(sorted(mp.rglob("*.md")))
        return sorted(set(results))

    def parse(self, root: Path) -> ParseResult:
        """Memory source requires atomize (currently disabled).

        Raises:
            AtomizeUnavailableError: Always, since atomize is currently paused.
        """
        raise AtomizeUnavailableError(
            "memory source requires atomize (currently disabled)"
        )
