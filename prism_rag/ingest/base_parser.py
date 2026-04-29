"""Parser plugin interface — base class for all data-source parsers.

Every parser converts a raw data source into a ParseTree. The StorageBackend
then decides how to persist it (NetworkX today, Kuzu tomorrow).

Content rule:
  Each TreeNode.content = the node's OWN text only, not its children's.
  This keeps embeddings precise and avoids BM25 duplication.

Implementing a new parser:

    class MyParser(Parser):
        @property
        def namespace(self) -> str:
            return "myns"

        def parse(self, source: Path) -> ParseTree:
            root = TreeNode(id=..., kind="note", ...)
            root.add_child(TreeNode(...))
            return ParseTree(root=root, namespace=self.namespace, source_file=str(source))

Namespaces:
  "nimbus"  — Obsidian / markdown vault     (NimbusParser — planned)
  "code"    — source-code repository         (CodeParser — planned)
  "conv"    — conversation extraction        (reserved, not yet implemented)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from prism_rag.ingest.base_tree import ParseTree


class Parser(ABC):
    """Abstract base class for all PrismRag data-source parsers."""

    @property
    @abstractmethod
    def namespace(self) -> str:
        """Namespace tag written into every produced TreeNode.

        Convention:
          "nimbus"  — Obsidian / markdown vault
          "code"    — source-code repository
          "conv"    — conversation / dialogue extraction (reserved)
        """
        ...

    @abstractmethod
    def parse(self, source: Path) -> ParseTree:
        """Parse *source* and return a ParseTree.

        Args:
            source: Root path to parse (single file or directory).

        Returns:
            ParseTree whose root.content_hash reflects the full file content
            (for file-level change detection in incremental ingest).
        """
        ...
