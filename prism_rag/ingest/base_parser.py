"""Parser plugin interface — base class for all data-source parsers.

Every parser converts a raw data source into a validated ParseResult. Internally
parsers build a ParseTree and call ParseResult.from_tree() to validate and flatten.
The StorageBackend then decides how to persist it (NetworkX today, Kuzu tomorrow).

Content rule:
  Each TreeNode.content = the node's OWN text only, not its children's.
  This keeps embeddings precise and avoids BM25 duplication.

Implementing a new parser:

    class MyParser(Parser):
        @property
        def namespace(self) -> str:
            return "myns"

        def parse(self, source: Path) -> ParseResult:
            root = TreeNode(id=..., kind="note", ...)
            root.add_child(TreeNode(...))
            tree = ParseTree(root=root, namespace=self.namespace, source_file=str(source))
            return ParseResult.from_tree(tree, parser_id="MyParser")

Namespaces:
  "nimbus"  — Obsidian / markdown vault     (NimbusParser)
  "code"    — source-code repository         (CodeParser — planned)
  "conv"    — conversation extraction        (reserved, not yet implemented)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from prism_rag.ingest.parse_result import ParseResult


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
    def parse(self, source: Path) -> ParseResult:
        """Parse *source* and return a validated ParseResult.

        Args:
            source: Root path to parse (single file or directory).

        Returns:
            ParseResult with validated NodeRecords and EdgeRecords. The root
            node's content_hash must reflect the full source content for
            file-level change detection in incremental ingest.
        """
        ...
