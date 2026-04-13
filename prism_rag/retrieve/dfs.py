"""DFS graph traversal with token budget pruning.

Depth-first search from an entry node, following single dependency chains.
Useful for tracing "how did A lead to B" type questions.

Neighbors are explored in order of edge weight (highest first).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prism_rag.store.graph import KnowledgeGraph

if TYPE_CHECKING:
    from prism_rag.store.federated import FederatedGraph


def dfs_traverse(
    graph: KnowledgeGraph,
    entry_id: str,
    budget: int = 4000,
    max_depth: int = 10,
) -> list[dict]:
    """DFS from entry_id, collecting nodes up to token budget.

    Args:
        graph: The knowledge graph.
        entry_id: Starting node ID.
        budget: Maximum total tokens to collect.
        max_depth: Maximum DFS depth.

    Returns:
        List of node data dicts (including 'id'), ordered by traversal.
    """
    if entry_id not in graph.g:
        return []

    visited: set[str] = set()
    result: list[dict] = []
    accumulated_tokens = 0

    def _dfs(node_id: str, depth: int) -> None:
        nonlocal accumulated_tokens

        if node_id in visited or depth > max_depth:
            return

        node_data = graph.g.nodes[node_id]
        node_tokens = node_data.get("tokens", 0)

        if result and accumulated_tokens + node_tokens > budget:
            return

        visited.add(node_id)
        accumulated_tokens += node_tokens
        result.append({"id": node_id, **node_data})

        if accumulated_tokens >= budget:
            return

        # Collect all neighbors (outgoing + incoming)
        neighbors: list[tuple[str, float]] = []
        for neighbor_id in graph.g.neighbors(node_id):
            if neighbor_id not in visited:
                weight = float(graph.g.edges[node_id, neighbor_id].get("weight", 1.0))
                neighbors.append((neighbor_id, weight))
        for predecessor_id in graph.g.predecessors(node_id):
            if predecessor_id not in visited:
                weight = float(graph.g.edges[predecessor_id, node_id].get("weight", 1.0))
                neighbors.append((predecessor_id, weight))

        # Sort by weight descending, then DFS into each
        neighbors.sort(key=lambda pair: pair[1], reverse=True)
        for neighbor_id, _ in neighbors:
            _dfs(neighbor_id, depth + 1)

    _dfs(entry_id, 0)
    return result


def federated_dfs(
    federated: "FederatedGraph",
    namespace: str,
    entry_id: str,
    budget: int = 4000,
    max_depth: int = 10,
) -> list[dict]:
    """DFS traversal starting from a node in a specific namespace.
    Results include a "namespace" key on each node dict.
    """
    graph = federated.get_graph(namespace)
    if graph is None:
        return []
    nodes = dfs_traverse(graph, entry_id, budget=budget, max_depth=max_depth)
    for n in nodes:
        n["namespace"] = namespace
    return nodes
