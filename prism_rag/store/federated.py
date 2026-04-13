"""Federated multi-graph layer.

Loads multiple independent KnowledgeGraph instances and provides unified
query access with namespace-prefixed node IDs.

Bridge edges (cross-graph) are computed at load time (serve-time),
not persisted — they depend on which graphs are loaded.
"""
from __future__ import annotations

import logging
from typing import Any

from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class FederatedGraph:
    """Runtime federation over multiple KnowledgeGraph instances."""

    def __init__(self, graphs: dict[str, KnowledgeGraph]) -> None:
        self._graphs: dict[str, KnowledgeGraph] = dict(graphs)
        self._single = len(self._graphs) == 1
        self._bridges: list[dict] = []

    @property
    def namespaces(self) -> list[str]:
        return sorted(self._graphs.keys())

    @property
    def node_count(self) -> int:
        return sum(g.node_count for g in self._graphs.values())

    @property
    def edge_count(self) -> int:
        return sum(g.edge_count for g in self._graphs.values()) + len(self._bridges)

    @property
    def is_single(self) -> bool:
        return self._single

    @property
    def bridges(self) -> list[dict]:
        return self._bridges

    def get_graph(self, namespace: str) -> KnowledgeGraph | None:
        return self._graphs.get(namespace)

    def get_node(self, qualified_id: str) -> dict[str, Any] | None:
        """Get node data by qualified ID ("namespace::node_id").
        In single-graph mode, bare node_id (no prefix) is accepted.
        """
        ns, node_id = self._parse_id(qualified_id)
        graph = self._graphs.get(ns)
        if graph is None:
            return None
        if node_id not in graph.g:
            return None
        return dict(graph.g.nodes[node_id])

    def _parse_id(self, qualified_id: str) -> tuple[str, str]:
        """Parse "namespace::node_id" → (namespace, node_id).
        In single-graph mode, bare IDs map to the only namespace.
        """
        if "::" in qualified_id:
            ns, _, node_id = qualified_id.partition("::")
            return ns, node_id
        if self._single:
            return next(iter(self._graphs)), qualified_id
        # Multi-graph but no prefix — search all graphs
        for ns, g in self._graphs.items():
            if qualified_id in g.g:
                return ns, qualified_id
        return "", qualified_id

    def build_bridges(self) -> int:
        """Compute cross-graph bridge edges.

        Bridge types:
        1. Shared tags: same tag node ID exists in multiple graphs
           → bridge between the tag nodes

        Returns: number of bridge edges created.
        """
        self._bridges.clear()
        if self._single:
            return 0

        # Shared tag bridges
        tag_index: dict[str, list[str]] = {}  # tag_id → [namespace, ...]
        for ns, g in self._graphs.items():
            for node_id, data in g.g.nodes(data=True):
                if data.get("kind") == "tag":
                    tag_index.setdefault(node_id, []).append(ns)

        for tag_id, namespaces in tag_index.items():
            if len(namespaces) < 2:
                continue
            for i in range(len(namespaces)):
                for j in range(i + 1, len(namespaces)):
                    self._bridges.append({
                        "source_ns": namespaces[i],
                        "source_id": tag_id,
                        "target_ns": namespaces[j],
                        "target_id": tag_id,
                        "relation": "shared_tag",
                        "confidence": "INFERRED",
                        "weight": 0.5,
                    })

        logger.info(f"[federated] built {len(self._bridges)} bridge edges across {len(self._graphs)} graphs")
        return len(self._bridges)

    @classmethod
    def load(cls, sources: list) -> "FederatedGraph":
        """Load a FederatedGraph from a list of GraphSource configs.
        Skips sources whose graph.json doesn't exist (logs warning).
        Automatically computes bridge edges after loading.
        """
        graphs: dict[str, KnowledgeGraph] = {}
        for src in sources:
            gpath = src.graph_path
            if not gpath.exists():
                logger.warning(f"[federated] graph not found: {gpath} (namespace={src.namespace}), skipping")
                continue
            g = KnowledgeGraph.load(gpath)
            graphs[src.namespace] = g
            logger.info(f"[federated] loaded {src.namespace}: {g.node_count} nodes, {g.edge_count} edges")
        fg = cls(graphs)
        fg.build_bridges()
        return fg
