"""Entry point resolution: map a user query to a starting node in the graph.

Resolution order:
1. Exact label match (case-insensitive)
2. Alias match (from frontmatter aliases)
3. Substring match on labels
4. (Future) Embedding fallback (top-1 vector search)

Returns the best-matching node ID, or None if no match.
"""

from __future__ import annotations

import logging

from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def resolve_entry_point(
    graph: KnowledgeGraph,
    query: str,
) -> str | None:
    """Find the best entry node for a query string.

    Args:
        graph: The knowledge graph to search.
        query: User's query string.

    Returns:
        Node ID of the best match, or None.
    """
    query_lower = query.lower().strip()
    if not query_lower:
        return None

    # 1. Exact label match (case-insensitive)
    for node_id, data in graph.g.nodes(data=True):
        label = data.get("label", "")
        if label.lower() == query_lower:
            logger.debug(f"[entry] exact label match: {node_id}")
            return node_id

    # 2. Exact ID match (case-insensitive)
    for node_id in graph.g.nodes():
        if node_id.lower() == query_lower:
            logger.debug(f"[entry] exact id match: {node_id}")
            return node_id

    # 3. Alias match (frontmatter aliases)
    for node_id, data in graph.g.nodes(data=True):
        aliases = data.get("frontmatter", {}).get("aliases", [])
        if isinstance(aliases, list):
            for alias in aliases:
                if str(alias).lower() == query_lower:
                    logger.debug(f"[entry] alias match: {node_id} via {alias!r}")
                    return node_id

    # 4. Substring match on labels (return best = shortest label containing query)
    candidates: list[tuple[str, str]] = []
    for node_id, data in graph.g.nodes(data=True):
        label = data.get("label", "")
        if query_lower in label.lower():
            candidates.append((node_id, label))

    if candidates:
        # Prefer shortest label (most specific match)
        best = min(candidates, key=lambda pair: len(pair[1]))
        logger.debug(f"[entry] substring match: {best[0]} (label={best[1]!r})")
        return best[0]

    # 5. Tag match: try "tag:{query}"
    tag_id = f"tag:{query_lower}"
    if tag_id in graph.g:
        logger.debug(f"[entry] tag match: {tag_id}")
        return tag_id

    logger.debug(f"[entry] no match for query={query!r}")
    return None
