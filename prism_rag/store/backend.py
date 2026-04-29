"""StorageBackend — abstract write interface for ParseTree persistence.

Today: NetworkXBackend (graph.json + in-memory DiGraph).
Future: KuzuBackend (per-kind node tables + typed REL tables + Cypher queries).

ParseTree is backend-agnostic; only the backend knows how to store and query.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from prism_rag.ingest.base_tree import ParseTree


class StorageBackend(ABC):
    """Write interface for persisting ParseTrees into a graph store."""

    @abstractmethod
    def write_tree(self, tree: ParseTree) -> None:
        """Write a ParseTree.

        For updates: call delete_by_source first, then write_tree.
        """
        ...

    @abstractmethod
    def delete_by_source(self, source_file: str, namespace: str) -> int:
        """Remove all nodes from source_file in namespace.

        Returns the number of nodes removed.
        """
        ...

    @abstractmethod
    def file_hash(self, source_file: str, namespace: str) -> str | None:
        """Return the stored content_hash for source_file, or None if not indexed."""
        ...

    def has_changed(self, source_file: str, namespace: str, content_hash: str) -> bool:
        """True if source_file needs re-indexing (hash mismatch or not found)."""
        return self.file_hash(source_file, namespace) != content_hash
