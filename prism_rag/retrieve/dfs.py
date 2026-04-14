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
    scope: str = "",
) -> list[dict]:
    """DFS traversal starting from a node, crossing bridge edges.

    In single-graph mode, delegates to dfs_traverse() for zero overhead.
    In multi-graph mode, operates on the unified_view.

    Args:
        scope: If non-empty, restrict traversal to this namespace only.
    """
    if federated.is_single:
        graph = federated.get_graph(namespace)
        if graph is None:
            return []
        nodes = dfs_traverse(graph, entry_id, budget=budget, max_depth=max_depth)
        for n in nodes:
            n["namespace"] = namespace
        return nodes

    # Multi-graph: use unified_view
    uv = federated.unified_view
    entry_qid = f"{namespace}::{entry_id}"
    if entry_qid not in uv:
        return []

    visited: set[str] = set()
    result: list[dict] = []
    accumulated_tokens = 0

    def _dfs(qid: str, depth: int) -> None:
        nonlocal accumulated_tokens

        if qid in visited or depth > max_depth:
            return

        node_data = uv.nodes[qid]
        node_ns = node_data.get("namespace", namespace)

        if scope and node_ns != scope:
            return

        node_tokens = node_data.get("tokens", 0)
        if result and accumulated_tokens + node_tokens > budget:
            return

        visited.add(qid)
        accumulated_tokens += node_tokens

        bare_id = qid.split("::", 1)[1] if "::" in qid else qid
        result.append({"id": bare_id, "namespace": node_ns, **{
            k: v for k, v in node_data.items() if k != "namespace"
        }})

        if accumulated_tokens >= budget:
            return

        neighbors: list[tuple[str, float]] = []
        for nbr in uv.neighbors(qid):
            if nbr not in visited:
                w = float(uv.edges[qid, nbr].get("weight", 1.0))
                neighbors.append((nbr, w))
        for pred in uv.predecessors(qid):
            if pred not in visited:
                w = float(uv.edges[pred, qid].get("weight", 1.0))
                neighbors.append((pred, w))

        neighbors.sort(key=lambda p: p[1], reverse=True)
        for nbr, _ in neighbors:
            _dfs(nbr, depth + 1)

    _dfs(entry_qid, 0)
    return result
