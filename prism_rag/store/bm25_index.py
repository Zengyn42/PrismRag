"""BM25 keyword index over KnowledgeGraph nodes.

Built at serve-time from node labels + content.
Used as one of the three retrieval signals in hybrid search.

Degrades gracefully if rank_bm25 or jieba are not installed:
  - Missing rank_bm25 → search() returns []
  - Missing jieba     → falls back to whitespace tokenisation
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    logger.warning("[bm25] rank_bm25 not installed — BM25 retrieval disabled. "
                   "pip install rank-bm25")

try:
    import jieba
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False


def _tokenize(text: str) -> list[str]:
    """Tokenise mixed Chinese/English text.

    Chinese: jieba cut_for_search (if available), else character n-grams.
    English / numbers: split on whitespace + punctuation.
    """
    if not text:
        return []

    if _JIEBA_AVAILABLE:
        import jieba as _jieba
        tokens = list(_jieba.cut_for_search(text, cut_all=False))
    else:
        # Fallback: split CJK chars individually, keep ASCII words
        tokens = []
        current = []
        for ch in text:
            if '一' <= ch <= '鿿':  # CJK Unified Ideographs
                if current:
                    tokens.extend(re.split(r'\W+', ''.join(current)))
                    current = []
                tokens.append(ch)
            else:
                current.append(ch)
        if current:
            tokens.extend(re.split(r'\W+', ''.join(current)))

    return [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]


class BM25Index:
    """In-memory BM25 index built from a KnowledgeGraph.

    Usage::

        index = BM25Index()
        index.build(graph)
        results = index.search("embedding pipeline", top_k=20)
        # → [("node_id_1", 4.21), ("node_id_2", 3.87), ...]
    """

    def __init__(self) -> None:
        self._index = None          # BM25Okapi instance
        self._node_ids: list[str] = []
        self._built = False

    def build(self, graph: "KnowledgeGraph") -> None:
        """Build (or rebuild) the BM25 index from all graph nodes."""
        if not _BM25_AVAILABLE:
            return

        corpus: list[list[str]] = []
        node_ids: list[str] = []

        for node_id, data in graph.g.nodes(data=True):
            label = data.get("label", "")
            content = data.get("content", "")
            # Weight label higher by repeating it
            text = f"{label} {label} {content}"
            tokens = _tokenize(text)
            corpus.append(tokens)
            node_ids.append(node_id)

        if not corpus:
            return

        self._index = BM25Okapi(corpus)
        self._node_ids = node_ids
        self._built = True
        logger.debug(f"[bm25] index built: {len(node_ids)} nodes")

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return top-K (node_id, score) pairs ordered by BM25 relevance.

        Returns an empty list if the index was not built or rank_bm25 is missing.
        """
        if not self._built or self._index is None:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)
        ranked = sorted(
            zip(self._node_ids, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(nid, float(score)) for nid, score in ranked[:top_k] if score > 0]

    @property
    def is_ready(self) -> bool:
        return self._built
