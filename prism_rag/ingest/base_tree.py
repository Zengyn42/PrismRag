"""Intermediate parse tree — produced by Parser, consumed by StorageBackend.

Decouples parsing from storage: parsers build trees, backends decide how to
store them (NetworkX today, Kuzu tomorrow).

Content rule (following GitNexus):
  Each TreeNode.content = the node's OWN text only, not its children's.
  Children are linked via the `children` list. This keeps embeddings precise
  and avoids duplication in the BM25 index.

Namespace convention:
  "nimbus"  — Obsidian / markdown vault
  "code"    — source-code repository
  "conv"    — conversation extraction (interface reserved, not yet implemented)

Metadata schema
───────────────
metadata keys are kind-specific. KuzuBackend maps them directly to table columns,
so keys must be stable. See the TypedDicts below for the per-kind contracts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from typing_extensions import TypedDict

from prism_rag.store.graph import NodeKind


# ── Metadata TypedDicts ───────────────────────────────────────────────────────
# Define the contract between parsers and storage backends.
# KuzuBackend maps these keys directly to typed table columns.
# NetworkXBackend stores them verbatim in Node.metadata.

class SectionMeta(TypedDict, total=False):
    heading_level: int       # 1–6

class BlockMeta(TypedDict, total=False):
    block_type: str          # "paragraph"|"callout"|"code_block"|"list"|"table"
    callout_type: str        # Obsidian callout kind: "note"|"warning"|"info"|...

class NoteMeta(TypedDict, total=False):
    frontmatter: dict        # raw YAML frontmatter
    tags: list               # list[str]

class KnowledgeMeta(TypedDict, total=False):
    frontmatter: dict
    tags: list               # list[str]
    know_id: str             # e.g. "KNOW-042"

class FunctionMeta(TypedDict, total=False):
    language: str
    line_start: int
    line_end: int
    signature: str           # full signature string
    is_exported: bool
    is_async: bool
    parameters: list         # list[str]
    return_type: str         # absent key = no return type
    docstring: str

class ClassMeta(TypedDict, total=False):
    language: str
    line_start: int
    line_end: int
    is_exported: bool
    bases: list              # list[str] — parent class names
    docstring: str

class ModuleMeta(TypedDict, total=False):
    language: str
    line_count: int

# conv namespace — reserved, not yet implemented
# FactMeta: source_session: str, extracted_at: str, speaker: str | None


# ── File-level node kinds ─────────────────────────────────────────────────────
# The root TreeNode of a ParseTree always has one of these kinds.
# Used by NetworkXBackend to locate the root node for file_hash lookup.
FILE_LEVEL_KINDS: frozenset[str] = frozenset({
    "note", "knowledge", "module", "pdf", "audio", "image",
})


# ── Core dataclasses ──────────────────────────────────────────────────────────

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass
class TreeNode:
    """A single node in the parse tree.

    `content` = own text only — NOT including children's text.
    `metadata` keys are kind-specific (see TypedDicts above).
    """

    id: str
    kind: NodeKind
    label: str
    content: str            # own text only
    namespace: str
    source_file: str        # vault-relative or absolute path string

    tokens: int = 0
    content_hash: str = "" # SHA1 of content; auto-computed if empty

    children: list["TreeNode"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash and self.content:
            self.content_hash = _sha1(self.content)

    def add_child(self, child: "TreeNode") -> "TreeNode":
        self.children.append(child)
        return child

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def walk(self):
        """Depth-first pre-order traversal of this node and all descendants."""
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class ParseTree:
    """Complete parse result for one source file."""

    root: TreeNode
    namespace: str
    source_file: str        # canonical path string

    @property
    def file_hash(self) -> str:
        """Hash for file-level change detection.

        Parsers should set root.content_hash to the hash of the full source
        file (not just root's own content) so that any change anywhere in the
        file triggers re-indexing.
        """
        return self.root.content_hash

    def all_nodes(self) -> list[TreeNode]:
        """All nodes in DFS pre-order."""
        return list(self.root.walk())
