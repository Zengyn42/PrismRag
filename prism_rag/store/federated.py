"""Federated multi-graph layer.

Loads multiple independent KnowledgeGraph instances and provides unified
query access with namespace-prefixed node IDs.

Bridge edges (cross-graph) are computed at load time (serve-time),
not persisted — they depend on which graphs are loaded.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prism_rag.store.embedding_store import EmbeddingStore

import networkx as nx

from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class FederatedGraph:
    """Runtime federation over multiple KnowledgeGraph instances."""

    def __init__(self, graphs: dict[str, KnowledgeGraph]) -> None:
        self._graphs: dict[str, KnowledgeGraph] = dict(graphs)
        self._single = len(self._graphs) == 1
        self._bridges: list[dict] = []
        self._unified: nx.DiGraph | None = None

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

    @property
    def unified_view(self) -> nx.DiGraph:
        """Lazy-built unified graph with namespace-prefixed node IDs + bridge edges.
        Single-graph mode returns the original graph directly (zero-copy).
        """
        if self._single:
            return next(iter(self._graphs.values())).g

        if self._unified is not None:
            return self._unified

        unified = nx.DiGraph()

        for ns, kg in self._graphs.items():
            for node_id, data in kg.g.nodes(data=True):
                qid = f"{ns}::{node_id}"
                unified.add_node(qid, **{**data, "namespace": ns})
            for src, tgt, data in kg.g.edges(data=True):
                unified.add_edge(f"{ns}::{src}", f"{ns}::{tgt}", **data)

        for bridge in self._bridges:
            src_qid = f"{bridge['source_ns']}::{bridge['source_id']}"
            tgt_qid = f"{bridge['target_ns']}::{bridge['target_id']}"
            unified.add_edge(src_qid, tgt_qid,
                             relation=bridge["relation"],
                             confidence=bridge["confidence"],
                             weight=bridge.get("weight", 0.5))
            unified.add_edge(tgt_qid, src_qid,
                             relation=bridge["relation"],
                             confidence=bridge["confidence"],
                             weight=bridge.get("weight", 0.5))

        self._unified = unified
        return self._unified

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

    def build_bridges(
        self,
        stores: dict[str, "EmbeddingStore"] | None = None,
        bridge_threshold: float = 0.70,
        bridge_top_k: int = 5,
    ) -> int:
        """Compute cross-graph bridge edges.

        Bridge types:
        1. Shared tags: same tag node ID exists in multiple graphs
           → bridge between the tag nodes
        2. Embedding similarity: nodes with similar embeddings across namespaces

        Returns: number of bridge edges created.
        """
        self._bridges.clear()
        self._unified = None
        if self._single:
            return 0

        # 1. Shared tag bridges
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

        # 2. Embedding similarity bridges
        if stores and len(stores) >= 2:
            self._build_embedding_bridges(stores, bridge_threshold, bridge_top_k)

        logger.info(f"[federated] built {len(self._bridges)} bridge edges across {len(self._graphs)} graphs")
        return len(self._bridges)

    def _build_embedding_bridges(
        self,
        stores: dict[str, "EmbeddingStore"],
        threshold: float,
        top_k: int,
    ) -> None:
        """Add embedding similarity bridges between namespace pairs."""
        ns_list = sorted(stores.keys())
        existing_bridges: set[tuple[str, str, str, str]] = set()

        for i in range(len(ns_list)):
            for j in range(i + 1, len(ns_list)):
                ns_a, ns_b = ns_list[i], ns_list[j]
                store_a, store_b = stores[ns_a], stores[ns_b]
                vecs_a = store_a.all_embeddings()

                for node_id_a, vec_a in vecs_a.items():
                    # Verify node still exists in graph
                    if node_id_a not in self._graphs[ns_a].g:
                        continue

                    results = store_b.search(vec_a, top_k=top_k)
                    for node_id_b, distance in results:
                        # Verify node still exists in graph
                        graph_b = self._graphs.get(ns_b)
                        if graph_b is None or node_id_b not in graph_b.g:
                            continue

                        # LanceDB returns L2 distance; convert to cosine similarity
                        # For normalized vectors: cosine_sim ≈ 1 - (L2² / 2)
                        similarity = 1.0 - (distance / 2.0)

                        if similarity < threshold:
                            continue

                        # Avoid duplicate bridges
                        key = (ns_a, node_id_a, ns_b, node_id_b)
                        rev_key = (ns_b, node_id_b, ns_a, node_id_a)
                        if key in existing_bridges or rev_key in existing_bridges:
                            continue
                        existing_bridges.add(key)

                        self._bridges.append({
                            "source_ns": ns_a,
                            "source_id": node_id_a,
                            "target_ns": ns_b,
                            "target_id": node_id_b,
                            "relation": "embedding_similar",
                            "confidence": "INFERRED",
                            "weight": round(similarity, 4),
                            "source_pass": "embedding",
                        })

        emb_count = sum(1 for b in self._bridges if b["relation"] == "embedding_similar")
        logger.info(f"[federated] embedding bridges: {emb_count}")

    @classmethod
    def load(cls, sources: list, settings=None) -> "FederatedGraph":
        """Load a FederatedGraph from a list of GraphSource configs.
        Skips sources whose graph.json doesn't exist (logs warning).
        Automatically computes bridge edges after loading.
        Loads EmbeddingStores from lance/ subdirs when available.
        """
        graphs: dict[str, KnowledgeGraph] = {}
        stores: dict[str, "EmbeddingStore"] = {}
        for src in sources:
            gpath = src.graph_path
            if not gpath.exists():
                logger.warning(f"[federated] graph not found: {gpath} (namespace={src.namespace}), skipping")
                continue
            g = KnowledgeGraph.load(gpath)
            graphs[src.namespace] = g
            logger.info(f"[federated] loaded {src.namespace}: {g.node_count} nodes, {g.edge_count} edges")

            lance_path = src.data_dir / "lance"
            if lance_path.exists():
                try:
                    from prism_rag.store.embedding_store import EmbeddingStore as ES
                    store = ES(lance_path)
                    if store.count() > 0:
                        stores[src.namespace] = store
                        logger.info(f"[federated] loaded embeddings for {src.namespace}: {store.count()} vectors")
                except Exception as e:
                    logger.warning(f"[federated] failed to load embeddings for {src.namespace}: {e}")

        fg = cls(graphs)

        bridge_threshold = 0.70
        bridge_top_k = 5
        if settings:
            bridge_threshold = getattr(settings, "bridge_similarity_threshold", 0.70)
            bridge_top_k = getattr(settings, "bridge_top_k", 5)

        fg.build_bridges(
            stores=stores or None,
            bridge_threshold=bridge_threshold,
            bridge_top_k=bridge_top_k,
        )
        return fg
