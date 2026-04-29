"""NetworkX implementation of StorageBackend.

Walks a ParseTree and writes each TreeNode as a Node into a KnowledgeGraph,
with Edge(relation="contains") for every parent→child relationship.

file_hash() scans for the root node (FILE_LEVEL_KINDS) to retrieve the
stored content_hash for incremental ingest change detection.
"""

from __future__ import annotations

from prism_rag.ingest.base_tree import FILE_LEVEL_KINDS, ParseTree, TreeNode
from prism_rag.store.backend import StorageBackend
from prism_rag.store.graph import Edge, KnowledgeGraph, Node


class NetworkXBackend(StorageBackend):
    """StorageBackend backed by a KnowledgeGraph (NetworkX DiGraph)."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

    @property
    def graph(self) -> KnowledgeGraph:
        return self._graph

    # ── Write ─────────────────────────────────────────────────────────

    def write_tree(self, tree: ParseTree) -> None:
        self._walk(tree.root, parent_id=None)

    def _walk(self, node: TreeNode, parent_id: str | None) -> None:
        n = Node(
            id=node.id,
            label=node.label,
            kind=node.kind,
            source_file=node.source_file,
            content=node.content,
            content_hash=node.content_hash,
            tokens=node.tokens,
            namespace=node.namespace,
            # docs: YAML frontmatter lives at metadata["frontmatter"]
            frontmatter=node.metadata.get("frontmatter", {}),
            # code + shared: all other metadata keys
            metadata={k: v for k, v in node.metadata.items() if k != "frontmatter"},
            # Am attributes (populated by upstream agent, passed through metadata)
            maturity=node.metadata.get("maturity"),
            confidence=node.metadata.get("confidence"),
            actionability=node.metadata.get("actionability"),
            ontology_type=node.metadata.get("ontology_type"),
        )
        self._graph.add_node(n)

        if parent_id is not None:
            self._graph.add_edge(Edge(
                source=parent_id,
                target=node.id,
                relation="contains",
                confidence="EXTRACTED",
                source_pass="ast",
            ))

        for child in node.children:
            self._walk(child, parent_id=node.id)

    # ── Incremental ingest helpers ─────────────────────────────────────

    def delete_by_source(self, source_file: str, namespace: str) -> int:
        to_remove = [
            nid for nid, data in self._graph.g.nodes(data=True)
            if data.get("source_file") == source_file
            and data.get("namespace") == namespace
        ]
        self._graph.g.remove_nodes_from(to_remove)
        return len(to_remove)

    def file_hash(self, source_file: str, namespace: str) -> str | None:
        """Return content_hash stored on the root node for this source_file."""
        for _, data in self._graph.g.nodes(data=True):
            if (
                data.get("source_file") == source_file
                and data.get("namespace") == namespace
                and data.get("kind") in FILE_LEVEL_KINDS
            ):
                return data.get("content_hash") or None
        return None
