"""Pass 3b: Generate semantically_similar_to edges from embedding vectors.

For each node, find its top-K nearest neighbors by cosine similarity,
then create INFERRED edges for pairs above the similarity threshold.

This is the core of "流派 B": embedding is used INDEX-TIME ONLY to populate
graph edges. Query-time retrieval is pure graph traversal — no vector search.

Edges are deduplicated against existing EXTRACTED edges to avoid redundancy.
"""

from __future__ import annotations

import logging
import math

from prism_rag.config import PrismRagSettings
from prism_rag.store.graph import Edge, KnowledgeGraph

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _find_top_k(
    node_id: str,
    vectors: dict[str, list[float]],
    k: int,
    threshold: float,
) -> list[tuple[str, float]]:
    """Find top-K most similar nodes to node_id by cosine similarity.

    Returns list of (other_node_id, similarity_score) pairs, sorted descending.
    Only includes pairs above the threshold.
    """
    if node_id not in vectors:
        return []

    vec_a = vectors[node_id]
    similarities: list[tuple[str, float]] = []

    for other_id, vec_b in vectors.items():
        if other_id == node_id:
            continue
        sim = _cosine_similarity(vec_a, vec_b)
        if sim >= threshold:
            similarities.append((other_id, sim))

    # Sort by similarity descending, take top-K
    similarities.sort(key=lambda pair: pair[1], reverse=True)
    return similarities[:k]


def link_similar_nodes(
    graph: KnowledgeGraph,
    vectors: dict[str, list[float]],
    settings: PrismRagSettings,
) -> int:
    """Generate semantically_similar_to edges from embedding vectors.

    For each node with an embedding, find its top-K nearest neighbors
    and create INFERRED edges if:
    1. Cosine similarity >= threshold
    2. No existing EXTRACTED edge already connects the pair

    Args:
        graph: Knowledge graph to add edges to (mutated in place).
        vectors: dict mapping node_id → embedding vector.
        settings: Contains similarity_threshold and top_k_similarity.

    Returns:
        Number of new edges created.
    """
    threshold = settings.similarity_threshold
    top_k = settings.top_k_similarity

    if not vectors:
        logger.info("[similarity_linker] no vectors provided, skipping")
        return 0

    logger.info(
        f"[similarity_linker] linking {len(vectors)} nodes "
        f"(threshold={threshold}, top_k={top_k})"
    )

    # Build set of existing edges to avoid duplicates
    existing_edges: set[tuple[str, str]] = set()
    for u, v in graph.g.edges():
        existing_edges.add((u, v))
        existing_edges.add((v, u))  # treat as undirected for dedup

    new_edges = 0
    for node_id in vectors:
        neighbors = _find_top_k(node_id, vectors, k=top_k, threshold=threshold)
        for other_id, similarity in neighbors:
            # Skip if edge already exists (in either direction)
            if (node_id, other_id) in existing_edges:
                continue

            edge = Edge(
                source=node_id,
                target=other_id,
                relation="semantically_similar_to",
                confidence="INFERRED",
                confidence_score=round(similarity, 4),
                weight=round(similarity, 4),
                source_pass="embedding",
            )
            graph.add_edge(edge)
            existing_edges.add((node_id, other_id))
            existing_edges.add((other_id, node_id))
            new_edges += 1

    logger.info(f"[similarity_linker] created {new_edges} new similarity edges")
    return new_edges
