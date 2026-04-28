"""Hybrid search: BM25 + Embedding + RRF.

Three independent retrieval signals are fused via Reciprocal Rank Fusion
into a single ranked list of candidate entry nodes for BFS/DFS traversal.

    BM25     → keyword relevance (exact terms, IDs, Chinese words)
    Embedding → semantic similarity (synonyms, cross-lingual)
    Exact     → substring / ID match (precise node addressing)

RRF score: score(d) = Σ 1 / (k + rank_i(d))   where k=60

All three signals are optional — the fuser degrades to whatever is available.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from prism_rag.store.bm25_index import BM25Index
    from prism_rag.store.embedding_store import EmbeddingStore
    from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = _RRF_K,
) -> list[str]:
    """Fuse multiple ranked node-ID lists using Reciprocal Rank Fusion.

    Args:
        rankings: Each inner list is an ordered sequence of node IDs
                  (best first) from one retrieval signal.
        k: Smoothing constant (standard value: 60).

    Returns:
        Deduplicated list of node IDs ordered by fused RRF score (best first).

    Why RRF instead of score averaging:
        BM25 scores and cosine similarities live in incompatible ranges.
        RRF operates on *ranks*, so normalisation is implicit.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, node_id in enumerate(ranking, start=1):
            scores[node_id] += 1.0 / (k + rank)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def hybrid_search(
    query: str,
    graph: "KnowledgeGraph",
    bm25_index: "BM25Index | None" = None,
    embed_fn: "Callable[[str], list[float]] | None" = None,
    embedding_store: "EmbeddingStore | None" = None,
    top_k: int = 10,
    namespace: str = "",
) -> list[str]:
    """Return top-K node IDs ranked by hybrid BM25 + embedding + exact fusion.

    Args:
        query: Free-text query (Chinese or English).
        graph: The KnowledgeGraph to search.
        bm25_index: Pre-built BM25Index (optional; skipped if None or not ready).
        embed_fn: Callable that maps a query string → embedding vector.
                  Typically ``OllamaEmbedder().embed_query``.
        embedding_store: EmbeddingStore with ``nearest()`` method.
        top_k: Number of node IDs to return.
        namespace: If non-empty, filter results to nodes with this namespace.

    Returns:
        Ordered list of node IDs (best match first), length ≤ top_k.
    """
    pool = top_k * 4  # over-retrieve before RRF so fusion has room to work
    rankings: list[list[str]] = []

    # ── Signal 1: BM25 ───────────────────────────────────────────────────────
    if bm25_index is not None and bm25_index.is_ready:
        bm25_hits = [nid for nid, _ in bm25_index.search(query, top_k=pool)]
        if bm25_hits:
            rankings.append(bm25_hits)
            logger.debug(f"[hybrid] bm25: {len(bm25_hits)} hits")

    # ── Signal 2: Embedding similarity ──────────────────────────────────────
    if embed_fn is not None and embedding_store is not None:
        try:
            qvec = embed_fn(query)
            emb_hits = embedding_store.nearest(qvec, top_k=pool)
            if emb_hits:
                rankings.append(emb_hits)
                logger.debug(f"[hybrid] embedding: {len(emb_hits)} hits")
        except Exception as exc:
            logger.warning(f"[hybrid] embedding search failed: {exc}")

    # ── Signal 3: Exact / substring match ───────────────────────────────────
    query_lower = query.lower()
    exact_hits = [
        nid for nid, data in graph.g.nodes(data=True)
        if query_lower in nid.lower() or query_lower in data.get("label", "").lower()
    ]
    if exact_hits:
        rankings.append(exact_hits[:pool])
        logger.debug(f"[hybrid] exact: {len(exact_hits)} hits")

    if not rankings:
        return []

    fused = reciprocal_rank_fusion(rankings)

    # ── Namespace filter ─────────────────────────────────────────────────────
    if namespace:
        fused = [
            nid for nid in fused
            if graph.g.nodes[nid].get("namespace", "nimbus") == namespace
        ]

    return fused[:top_k]
