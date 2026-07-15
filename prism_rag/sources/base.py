"""Base protocol for pluggable source extractors.

Each extractor knows how to discover files from a root directory and parse them
into (nodes, edges) that merge into a KnowledgeGraph via ParseResult.

KNOT (Knowledge Ontology Token), file prefix KNOW- kept for data compat.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from prism_rag.ingest.parse_result import ParseResult

# Valid source kinds for the ingest pipeline.
SourceKind = Literal["code", "docs", "memory"]

# All valid source identifiers (sources + knot which is loaded separately).
VALID_SOURCES: frozenset[str] = frozenset({"code", "docs", "memory", "knot"})


class SourceExtractor(ABC):
    """Protocol for a pluggable graph source extractor."""

    @property
    @abstractmethod
    def kind(self) -> SourceKind:
        """The source kind this extractor handles."""
        ...

    @abstractmethod
    def discover(self, root: Path) -> list[Path]:
        """Discover relevant files under *root*.

        Returns:
            Sorted list of file paths to process.
        """
        ...

    @abstractmethod
    def parse(self, root: Path) -> ParseResult:
        """Parse discovered files into a ParseResult.

        Args:
            root: Root directory to extract from.

        Returns:
            ParseResult containing NodeRecords and EdgeRecords.
        """
        ...
