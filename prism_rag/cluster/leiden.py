"""Pass 4: Leiden community detection.

Runs the Leiden algorithm on the knowledge graph using `leidenalg` + `python-igraph`.
The directed NetworkX graph is converted to an undirected igraph for community
detection (Leiden works on undirected graphs; direction is preserved in the original
graph for query-time traversal).

After clustering:
1. Every node gets a `community_id` attribute
2. Each community gets a `Community` object in `graph.communities`
3. God nodes (top-degree within each community) are identified
4. A human-readable label is generated from the god nodes' labels

References:
- Traag, V.A., Waltman, L. & van Eck, N.J. "From Louvain to Leiden: guaranteeing
  well-connected communities." Sci Rep 9, 5233 (2019).
- https://leidenalg.readthedocs.io/
"""

from __future__ import annotations

import logging

import igraph as ig
import leidenalg

from prism_rag.store.graph import Community, KnowledgeGraph

logger = logging.getLogger(__name__)


def _networkx_to_igraph(graph: KnowledgeGraph) -> tuple[ig.Graph, list[str]]:
    """Convert a KnowledgeGraph's DiGraph to an undirected igraph with edge weights.

    Returns (ig_graph, node_list) where node_list[i] is the original node ID at igraph vertex i.
    """
    node_list = list(graph.g.nodes())
    if not node_list:
        return ig.Graph(directed=False), []

    node_idx = {nid: i for i, nid in enumerate(node_list)}
    edge_tuples: list[tuple[int, int]] = []
    weights: list[float] = []

    # Collect undirected edges (dedupe both directions, keeping max weight)
    seen: dict[tuple[int, int], float] = {}
    for u, v, data in graph.g.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        key = (min(i, j), max(i, j))
        w = float(data.get("weight", 1.0))
        if key in seen:
            seen[key] = max(seen[key], w)
        else:
            seen[key] = w

    for (i, j), w in seen.items():
        edge_tuples.append((i, j))
        weights.append(w)

    g_ig = ig.Graph(n=len(node_list), edges=edge_tuples, directed=False)
    if weights:
        g_ig.es["weight"] = weights
    return g_ig, node_list


def _label_from_god_nodes(
    graph: KnowledgeGraph, god_node_ids: list[str], max_tokens: int = 3
) -> str:
    """Generate a human-readable community label from the top god nodes."""
    labels = []
    for nid in god_node_ids[:max_tokens]:
        node_data = graph.g.nodes.get(nid, {})
        labels.append(node_data.get("label", nid))
    return " · ".join(labels) if labels else "unnamed"


def _compute_internal_density(graph: KnowledgeGraph, member_ids: list[str]) -> float:
    """Compute the edge density of a subgraph (0.0 to 1.0)."""
    n = len(member_ids)
    if n < 2:
        return 0.0
    subgraph = graph.g.subgraph(member_ids)
    n_edges = subgraph.number_of_edges()
    max_edges = n * (n - 1)  # directed, so n*(n-1)
    return round(n_edges / max_edges, 4) if max_edges > 0 else 0.0


def run_leiden(
    graph: KnowledgeGraph,
    resolution: float = 1.0,
    seed: int = 42,
    god_nodes_per_community: int = 5,
) -> int:
    """Run Leiden community detection on the graph.

    Mutates `graph` in place:
    - Every node gets a `community_id` attribute
    - `graph.communities` is populated with Community objects

    Returns: the number of communities detected.
    """
    if graph.node_count == 0:
        logger.info("[leiden] empty graph, skipping")
        return 0

    g_ig, node_list = _networkx_to_igraph(graph)
    logger.info(f"[leiden] running on {len(node_list)} nodes, {g_ig.ecount()} undirected edges")

    # Use CPM (Constant Potts Model) for resolution control,
    # or ModularityVertexPartition for resolution-less modularity optimization.
    # For the MVP we use ModularityVertexPartition with n_iterations=-1 (run to convergence).
    try:
        partition = leidenalg.find_partition(
            g_ig,
            leidenalg.ModularityVertexPartition,
            weights="weight" if g_ig.ecount() > 0 else None,
            n_iterations=-1,
            seed=seed,
        )
    except Exception as exc:
        logger.warning(f"[leiden] clustering failed ({exc}), creating single community")
        # Fallback: put everything in one community
        partition = [list(range(len(node_list)))]  # type: ignore[assignment]

    graph.communities.clear()
    for community_idx, member_indices in enumerate(partition):
        community_id = f"community_{community_idx:03d}"
        member_node_ids = [node_list[idx] for idx in member_indices]

        # Identify god nodes (highest degree within the community's induced subgraph)
        subgraph = graph.g.subgraph(member_node_ids)
        degree_sorted = sorted(
            member_node_ids,
            key=lambda nid: subgraph.degree(nid),
            reverse=True,
        )
        god_nodes = degree_sorted[:god_nodes_per_community]

        # Write community_id back to the main graph
        for nid in member_node_ids:
            graph.g.nodes[nid]["community_id"] = community_id

        # Build Community record
        community = Community(
            id=community_id,
            label=_label_from_god_nodes(graph, god_nodes),
            god_nodes=god_nodes,
            member_count=len(member_node_ids),
            internal_density=_compute_internal_density(graph, member_node_ids),
        )
        graph.communities[community_id] = community

    logger.info(f"[leiden] detected {len(graph.communities)} communities")
    return len(graph.communities)
