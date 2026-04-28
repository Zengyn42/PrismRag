"""BFS graph traversal with token budget pruning.

Breadth-first search from an entry node, collecting nodes until the token
budget is exhausted. Neighbors are explored in order of edge weight (highest
first), so high-confidence connections are prioritized.

This is the DEFAULT query mode — broad context collection.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from prism_rag.store.graph import KnowledgeGraph

if TYPE_CHECKING:
    from prism_rag.store.federated import FederatedGraph


def bfs_traverse(
    graph: KnowledgeGraph,
    entry_id: str,
    budget: int = 4000,
    max_depth: int = 10,
    min_confidence: float = 0.0,
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
            if float(edge_data.get("confidence_score", 1.0)) < min_confidence:
                continue
            weight = float(edge_data.get("weight", 1.0))
            neighbors.append((neighbor_id, weight))

        # Also check incoming edges (graph is directed, but we want to traverse both ways)
        for predecessor_id in graph.g.predecessors(node_id):
            if predecessor_id in visited:
                continue
            edge_data = graph.g.edges[predecessor_id, node_id]
            if float(edge_data.get("confidence_score", 1.0)) < min_confidence:
                continue
            weight = float(edge_data.get("weight", 1.0))
            neighbors.append((predecessor_id, weight))

        # Sort by weight descending (prefer stronger connections)
        neighbors.sort(key=lambda pair: pair[1], reverse=True)
        for neighbor_id, _ in neighbors:
            queue.append((neighbor_id, depth + 1))

    return result


def federated_bfs(
    federated: "FederatedGraph",
    namespace: str,
    entry_id: str,
    budget: int = 4000,
    max_depth: int = 10,
    scope: str = "",
) -> list[dict]:
    """BFS traversal starting from a node, crossing bridge edges.

    In single-graph mode, delegates to bfs_traverse() for zero overhead.
    In multi-graph mode, operates on the unified_view.

    Args:
        scope: If non-empty, restrict traversal to this namespace only
               (no bridge crossing).
    """
    if federated.is_single:
        graph = federated.get_graph(namespace)
        if graph is None:
            return []
        nodes = bfs_traverse(graph, entry_id, budget=budget, max_depth=max_depth)
        for n in nodes:
            n["namespace"] = namespace
        return nodes

    # Multi-graph: use unified_view
    uv = federated.unified_view
    entry_qid = f"{namespace}::{entry_id}"
    if entry_qid not in uv:
        return []

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(entry_qid, 0)])
    result: list[dict] = []
    accumulated_tokens = 0

    while queue:
        qid, depth = queue.popleft()
        if qid in visited or depth > max_depth:
            continue

        node_data = uv.nodes[qid]
        node_ns = node_data.get("namespace", namespace)

        # Scope filter: skip nodes outside the requested namespace
        if scope and node_ns != scope:
            continue

        node_tokens = node_data.get("tokens", 0)
        if result and accumulated_tokens + node_tokens > budget:
            continue

        visited.add(qid)
        accumulated_tokens += node_tokens

        # Parse bare ID from qualified ID for the result
        bare_id = qid.split("::", 1)[1] if "::" in qid else qid
        result.append({"id": bare_id, "namespace": node_ns, **{
            k: v for k, v in node_data.items() if k != "namespace"
        }})

        if accumulated_tokens >= budget:
            break

        # Neighbors: outgoing + incoming, sorted by weight
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
            queue.append((nbr, depth + 1))

    return result
