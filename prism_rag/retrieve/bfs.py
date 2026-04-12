"""BFS graph traversal with token budget pruning.

Breadth-first search from an entry node, collecting nodes until the token
budget is exhausted. Neighbors are explored in order of edge weight (highest
first), so high-confidence connections are prioritized.

This is the DEFAULT query mode — broad context collection.
"""

from __future__ import annotations

from collections import deque

from prism_rag.store.graph import KnowledgeGraph


def bfs_traverse(
    graph: KnowledgeGraph,
    entry_id: str,
    budget: int = 4000,
    max_depth: int = 10,
) -> list[dict]:
    """BFS from entry_id, collecting nodes up to token budget.

    Args:
        graph: The knowledge graph.
        entry_id: Starting node ID.
        budget: Maximum total tokens to collect.
        max_depth: Maximum BFS depth (prevents runaway on large graphs).

    Returns:
        List of node data dicts (including 'id'), ordered by traversal.
    """
    if entry_id not in graph.g:
        return []

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(entry_id, 0)])
    result: list[dict] = []
    accumulated_tokens = 0

    while queue:
        node_id, depth = queue.popleft()

        if node_id in visited:
            continue
        if depth > max_depth:
            continue

        node_data = graph.g.nodes[node_id]
        node_tokens = node_data.get("tokens", 0)

        # Check budget (always include at least the entry node)
        if result and accumulated_tokens + node_tokens > budget:
            continue

        visited.add(node_id)
        accumulated_tokens += node_tokens
        result.append({"id": node_id, **node_data})

        if accumulated_tokens >= budget:
            break

        # Get neighbors sorted by edge weight (highest first)
        neighbors: list[tuple[str, float]] = []
        for neighbor_id in graph.g.neighbors(node_id):
            if neighbor_id in visited:
                continue
            edge_data = graph.g.edges[node_id, neighbor_id]
            weight = float(edge_data.get("weight", 1.0))
            neighbors.append((neighbor_id, weight))

        # Also check incoming edges (graph is directed, but we want to traverse both ways)
        for predecessor_id in graph.g.predecessors(node_id):
            if predecessor_id in visited:
                continue
            edge_data = graph.g.edges[predecessor_id, node_id]
            weight = float(edge_data.get("weight", 1.0))
            neighbors.append((predecessor_id, weight))

        # Sort by weight descending (prefer stronger connections)
        neighbors.sort(key=lambda pair: pair[1], reverse=True)
        for neighbor_id, _ in neighbors:
            queue.append((neighbor_id, depth + 1))

    return result
