"""Parser plugin interface — base class for all data-source parsers.

Every parser converts a raw data source (vault directory, code repo,
conversation history, …) into a list of Nodes and Edges that the
ingest pipeline can process identically regardless of origin.

Implementing a new parser:

    class MyParser(Parser):
        namespace = "myns"

        def parse(self, source: Path) -> tuple[list[Node], list[Edge]]:
            ...

Then wire it into the ingest CLI:

    prism-rag ingest-myns --source /path/to/data
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from prism_rag.store.graph import Edge, Node


class Parser(ABC):
    """Abstract base class for all PrismRag data-source parsers.

    Subclasses own the domain-specific extraction logic.
    Everything downstream (embedding, Leiden, MCP) is parser-agnostic.
    """

    @property
    @abstractmethod
    def namespace(self) -> str:
        """Namespace tag written into every produced Node.

        Convention:
          "nimbus"  — Obsidian / markdown vault
          "code"    — source-code repository
          "conv"    — conversation / dialogue extraction
        """
        ...

    @abstractmethod
    def parse(self, source: Path) -> tuple[list[Node], list[Edge]]:
        """Parse *source* and return graph primitives.

        Args:
            source: Root path to parse. May be a single file or a directory.

        Returns:
            Tuple of (nodes, edges). Both lists may be empty but never None.
            Every Node must have ``namespace`` set to ``self.namespace``.
        """
        ...
