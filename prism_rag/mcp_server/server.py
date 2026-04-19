"""PrismRag MCP Server — unified vault + knowledge-graph tools.

Graph query tools (6):
  search_knowledge   BFS/DFS traversal from a query → related nodes
  explain_node       All info about a specific node + its neighbors
  trace_path         Shortest path between two nodes
  list_communities   Overview of all Leiden communities
  explore_community  Drill into a specific community's members
  list_namespaces    List all loaded knowledge graph namespaces

Vault CRUD tools (11) — ported from ZenithLoom's Obsidian MCP:
  read_note          Read a note's content + frontmatter + cas_hash
  list_files         List markdown files under a directory
  get_frontmatter    Return just the YAML frontmatter of a note
  write_note         Full-write (create or overwrite) with CAS + atomic + audit
  patch_note         Replace one heading-delimited section, preserving the rest
  update_frontmatter Merge changes into frontmatter (other fields untouched)
  move_note          Rename / relocate a note; graph node re-indexed
  delete_note        Soft-delete into .trash/; node removed from graph
  manage_tags        Add/remove frontmatter tags
  search_files       Keyword search over filename / content
  get_links          Outgoing and incoming wikilink references

Writes trigger incremental graph sync; conflicts and writes are audited
to data/audit.jsonl.

Usage:
  prism-rag serve                    # start MCP stdio server
  prism-rag serve --transport sse    # SSE mode (for network access)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx
from mcp.server.fastmcp import FastMCP

from prism_rag.config import PrismRagSettings
from prism_rag.retrieve.bfs import bfs_traverse, federated_bfs
from prism_rag.retrieve.dfs import dfs_traverse, federated_dfs
from prism_rag.retrieve.entry import resolve_entry_point, resolve_entry_points
from prism_rag.store.federated import FederatedGraph
from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# -- Global state (loaded once at startup) ------------------------------------

_federated: FederatedGraph | None = None

mcp = FastMCP(
    "PrismRag",
    instructions=(
        "PrismRag is a graph-first RAG system for knowledge bases. "
        "Use search_knowledge for broad queries, explain_node for specific concepts, "
        "trace_path to understand how two concepts connect, list_communities / "
        "explore_community for structural overview, and list_namespaces to see "
        "all loaded knowledge graphs."
    ),
)


def _ensure_federated() -> FederatedGraph:
    global _federated
    if _federated is None:
        settings = PrismRagSettings()
        _federated = FederatedGraph.load(settings.resolved_graphs, settings=settings)
        logger.info(
            f"[mcp] federated loaded: {_federated.node_count} nodes, "
            f"{_federated.edge_count} edges across {len(_federated.namespaces)} namespaces"
        )
    return _federated


def _node_summary(graph: KnowledgeGraph, node_id: str, include_content: bool = False) -> dict:
    """Build a concise summary dict for a single node."""
    data = graph.g.nodes.get(node_id, {})
    summary = {
        "id": node_id,
        "label": data.get("label", node_id),
        "kind": data.get("kind", "?"),
        "ontology_type": data.get("ontology_type"),
        "tokens": data.get("tokens", 0),
        "community": data.get("community_id", ""),
        "degree": graph.degree(node_id),
    }
    if data.get("source_file"):
        summary["source_file"] = data["source_file"]
    if include_content and data.get("content"):
        summary["content"] = data["content"]
    return summary


def _federated_node_summary(fg: FederatedGraph, qualified_id: str) -> dict:
    """Build node summary from a qualified ID (namespace::node_id) in federated graph."""
    if "::" in qualified_id:
        ns, _, bare_id = qualified_id.partition("::")
    elif fg.is_single:
        ns = fg.namespaces[0]
        bare_id = qualified_id
    else:
        ns, bare_id = "", qualified_id

    graph = fg.get_graph(ns)
    if graph is None:
        return {"id": bare_id, "namespace": ns, "label": bare_id, "kind": "?"}

    summary = _node_summary(graph, bare_id)
    summary["namespace"] = ns
    return summary


# -- Tool 1: search_knowledge ------------------------------------------------


@mcp.tool()
def search_knowledge(
    query: str,
    budget: int = 4000,
    mode: str = "bfs",
    scope: str = "",
    ontology_type: str = "",
) -> str:
    """Search the knowledge graph for information about a topic.

    Finds the best matching entry node, then traverses the graph
    collecting related nodes up to the token budget.

    Args:
        query: Natural language query or node name (e.g., "session management", "Colony Coder")
        budget: Maximum tokens to return (default 4000)
        mode: Traversal mode — "bfs" (broad context) or "dfs" (follow chains)
        scope: Namespace to search (e.g., "nimbus"). Empty = search all.
        ontology_type: Filter results to nodes with this ontology_type (e.g., "decision",
                       "concept", "fact"). Empty string (default) = no filter.

    Returns:
        JSON with entry point, traversed nodes, and their content.
    """
    fg = _ensure_federated()
    entries = resolve_entry_points(fg, query, scope=scope or None)
    if not entries:
        return json.dumps({"error": f"No matching node for query: {query!r}"}, ensure_ascii=False)

    ns, entry_id = entries[0]  # best match
    graph = fg.get_graph(ns)

    if mode == "dfs":
        nodes = federated_dfs(fg, ns, entry_id, budget=budget, scope=scope or None)
    else:
        nodes = federated_bfs(fg, ns, entry_id, budget=budget, scope=scope or None)

    # Build node list — include note and knowledge nodes; filter by ontology_type if given
    node_list = []
    for n in nodes:
        if n.get("kind") not in ("note", "knowledge"):
            continue
        if ontology_type and n.get("ontology_type") != ontology_type:
            continue
        node_list.append({
            "id": f"{ns}::{n['id']}" if not fg.is_single else n["id"],
            "label": n.get("label", n["id"]),
            "kind": n.get("kind", "?"),
            "ontology_type": n.get("ontology_type"),
            "tokens": n.get("tokens", 0),
            "community": n.get("community_id", ""),
            "content": n.get("content", "")[:2000],
        })

    result = {
        "entry_point": _node_summary(graph, entry_id),
        "namespace": ns,
        "total_nodes": len(node_list),
        "total_tokens": sum(n.get("tokens", 0) for n in nodes),
        "nodes": node_list,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# -- Tool 2: explain_node ----------------------------------------------------


@mcp.tool()
def explain_node(node: str, scope: str = "") -> str:
    """Get detailed information about a specific node and its connections.

    Args:
        node: Node ID, label, or partial name to look up.
        scope: Namespace to search (e.g., "nimbus"). Empty = search all.

    Returns:
        JSON with node details, incoming edges, outgoing edges, and community info.
    """
    fg = _ensure_federated()
    entries = resolve_entry_points(fg, node, scope=scope or None)
    if not entries:
        return json.dumps({"error": f"Node not found: {node!r}"}, ensure_ascii=False)

    ns, node_id = entries[0]
    graph = fg.get_graph(ns)
    data = graph.g.nodes[node_id]

    # Outgoing edges
    outgoing = []
    for target in graph.g.neighbors(node_id):
        edge_data = graph.g.edges[node_id, target]
        outgoing.append({
            "target": graph.g.nodes[target].get("label", target),
            "target_id": target,
            "relation": edge_data.get("relation", "?"),
            "confidence": edge_data.get("confidence", "?"),
            "score": edge_data.get("confidence_score", 0),
        })

    # Incoming edges
    incoming = []
    for source in graph.g.predecessors(node_id):
        edge_data = graph.g.edges[source, node_id]
        incoming.append({
            "source": graph.g.nodes[source].get("label", source),
            "source_id": source,
            "relation": edge_data.get("relation", "?"),
            "confidence": edge_data.get("confidence", "?"),
            "score": edge_data.get("confidence_score", 0),
        })

    # Community info
    community_id = data.get("community_id")
    community_info = None
    if community_id and community_id in graph.communities:
        comm = graph.communities[community_id]
        community_info = {
            "id": comm.id,
            "label": comm.label,
            "member_count": comm.member_count,
            "god_nodes": [graph.g.nodes[n].get("label", n) for n in comm.god_nodes],
        }

    result = {
        "node": _node_summary(graph, node_id, include_content=True),
        "namespace": ns,
        "outgoing_edges": outgoing,
        "incoming_edges": incoming,
        "community": community_info,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# -- Tool 3: trace_path ------------------------------------------------------


@mcp.tool()
def trace_path(from_node: str, to_node: str, max_length: int = 5) -> str:
    """Find the shortest path between two nodes in the knowledge graph.

    Supports cross-namespace paths via bridge edges.

    Args:
        from_node: Starting node (ID, label, partial name, or namespace::node_id)
        to_node: Ending node (ID, label, partial name, or namespace::node_id)
        max_length: Maximum path length to search (default 5)

    Returns:
        JSON with the shortest path as a sequence of nodes and edges.
    """
    fg = _ensure_federated()
    src_entries = resolve_entry_points(fg, from_node)
    tgt_entries = resolve_entry_points(fg, to_node)

    if not src_entries:
        return json.dumps({"error": f"Source node not found: {from_node!r}"}, ensure_ascii=False)
    if not tgt_entries:
        return json.dumps({"error": f"Target node not found: {to_node!r}"}, ensure_ascii=False)

    src_ns, src_id = src_entries[0]
    tgt_ns, tgt_id = tgt_entries[0]

    if fg.is_single:
        # Single-graph: use original graph directly (no prefixing)
        graph = fg.get_graph(src_ns)
        undirected = graph.g.to_undirected()
        src_qid, tgt_qid = src_id, tgt_id
    else:
        # Multi-graph: use unified view
        undirected = fg.unified_view.to_undirected()
        src_qid = f"{src_ns}::{src_id}"
        tgt_qid = f"{tgt_ns}::{tgt_id}"

    try:
        path = nx.shortest_path(undirected, source=src_qid, target=tgt_qid)
    except nx.NetworkXNoPath:
        return json.dumps({
            "error": "No path found",
            "from": _federated_node_summary(fg, f"{src_ns}::{src_id}"),
            "to": _federated_node_summary(fg, f"{tgt_ns}::{tgt_id}"),
        }, ensure_ascii=False)

    if len(path) - 1 > max_length:
        return json.dumps({
            "error": f"Shortest path has {len(path)-1} hops (max_length={max_length})",
            "from": _federated_node_summary(fg, f"{src_ns}::{src_id}"),
            "to": _federated_node_summary(fg, f"{tgt_ns}::{tgt_id}"),
        }, ensure_ascii=False)

    # Build path steps
    steps = []
    for i, qid in enumerate(path):
        step = _federated_node_summary(fg, qid if "::" in qid else f"{src_ns}::{qid}")
        if i < len(path) - 1:
            next_qid = path[i + 1]
            edge_data = undirected.edges.get((qid, next_qid), {})
            step["edge_to_next"] = {
                "relation": edge_data.get("relation", "?"),
                "confidence": edge_data.get("confidence", "?"),
                "score": edge_data.get("confidence_score", 0),
            }
        steps.append(step)

    result = {
        "path_length": len(path) - 1,
        "steps": steps,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# -- Tool 4: list_communities ------------------------------------------------


@mcp.tool()
def list_communities(ontology_type: str = "") -> str:
    """List all Leiden communities in the knowledge graph.

    Returns an overview with community labels, sizes, god nodes, and density.
    Aggregates communities across all loaded namespaces.

    Args:
        ontology_type: Filter — only include communities that contain at least one
                       member with this ontology_type. Empty string (default) = no filter.
    """
    fg = _ensure_federated()

    communities = []
    total_nodes = 0
    total_edges = 0

    for ns in fg.namespaces:
        graph = fg.get_graph(ns)
        total_nodes += graph.node_count
        total_edges += graph.edge_count

        for comm in sorted(graph.communities.values(), key=lambda c: -c.member_count):
            # If ontology_type filter active, skip communities with no matching members
            if ontology_type:
                has_match = any(
                    graph.g.nodes[nid].get("ontology_type") == ontology_type
                    for nid in graph.g.nodes
                    if graph.g.nodes[nid].get("community_id") == comm.id
                )
                if not has_match:
                    continue

            comm_id = comm.id if fg.is_single else f"{ns}::{comm.id}"
            communities.append({
                "id": comm_id,
                "namespace": ns,
                "label": comm.label,
                "member_count": comm.member_count,
                "internal_density": comm.internal_density,
                "god_nodes": [
                    graph.g.nodes[n].get("label", n) for n in comm.god_nodes
                ],
            })

    result = {
        "total_communities": len(communities),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "communities": communities,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# -- Tool 5: explore_community -----------------------------------------------


@mcp.tool()
def explore_community(community: str, ontology_type: str = "") -> str:
    """Explore a specific community's members and connections.

    Args:
        community: Community ID (e.g., "community_000" or "namespace::community_000")
                   or label substring.
        ontology_type: Filter returned members to only those with this ontology_type
                       (e.g., "decision", "concept"). Empty string (default) = no filter.

    Returns:
        JSON with all members, internal edges, and bridge edges to other communities.
    """
    fg = _ensure_federated()

    # Parse optional namespace prefix
    if "::" in community:
        ns_hint, _, comm_query = community.partition("::")
    else:
        ns_hint = None
        comm_query = community

    # Resolve community across namespaces
    target_comm = None
    target_graph = None
    target_ns = None
    comm_lower = comm_query.lower()

    search_ns = [ns_hint] if ns_hint else fg.namespaces
    for ns in search_ns:
        graph = fg.get_graph(ns)
        if graph is None:
            continue
        for comm in graph.communities.values():
            if comm.id == comm_query or comm_lower in comm.label.lower():
                target_comm = comm
                target_graph = graph
                target_ns = ns
                break
        if target_comm:
            break

    if target_comm is None:
        # Collect all available communities for error message
        available = []
        for ns in fg.namespaces:
            graph = fg.get_graph(ns)
            for c in graph.communities.values():
                cid = c.id if fg.is_single else f"{ns}::{c.id}"
                available.append({"id": cid, "label": c.label})
        return json.dumps({
            "error": f"Community not found: {community!r}",
            "available": available,
        }, ensure_ascii=False)

    graph = target_graph

    # Members (optionally filtered by ontology_type)
    members = [
        _node_summary(graph, nid)
        for nid, data in graph.g.nodes(data=True)
        if data.get("community_id") == target_comm.id
        and (not ontology_type or data.get("ontology_type") == ontology_type)
    ]

    # Internal edges (both endpoints in this community)
    internal_edges = []
    bridge_edges = []
    member_ids = {m["id"] for m in members}

    for u, v, data in graph.g.edges(data=True):
        u_in = u in member_ids
        v_in = v in member_ids
        edge_info = {
            "source": graph.g.nodes[u].get("label", u),
            "target": graph.g.nodes[v].get("label", v),
            "relation": data.get("relation", "?"),
            "confidence": data.get("confidence", "?"),
            "score": data.get("confidence_score", 0),
        }
        if u_in and v_in:
            internal_edges.append(edge_info)
        elif u_in or v_in:
            edge_info["cross_to"] = graph.g.nodes[v if u_in else u].get("community_id", "?")
            bridge_edges.append(edge_info)

    result = {
        "community": {
            "id": target_comm.id if fg.is_single else f"{target_ns}::{target_comm.id}",
            "namespace": target_ns,
            "label": target_comm.label,
            "member_count": target_comm.member_count,
            "internal_density": target_comm.internal_density,
            "god_nodes": [graph.g.nodes[n].get("label", n) for n in target_comm.god_nodes],
        },
        "members": members,
        "internal_edges": len(internal_edges),
        "bridge_edges": len(bridge_edges),
        "top_bridge_edges": sorted(bridge_edges, key=lambda e: -e["score"])[:10],
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# -- Tool 6: list_namespaces -------------------------------------------------


@mcp.tool()
def list_namespaces() -> str:
    """List all loaded knowledge graph namespaces with statistics."""
    fg = _ensure_federated()
    namespaces = []
    for ns in fg.namespaces:
        g = fg.get_graph(ns)
        namespaces.append({
            "namespace": ns,
            "nodes": g.node_count,
            "edges": g.edge_count,
            "communities": len(g.communities),
        })
    return json.dumps({
        "namespaces": namespaces,
        "bridges": len(fg.bridges),
        "total_nodes": fg.node_count,
    }, ensure_ascii=False, indent=2)


# -- Register ported Obsidian MCP tools (vault_tools.py) ---------------------

from prism_rag.mcp_server.vault_tools import register_vault_tools  # noqa: E402
register_vault_tools(mcp)


# -- Server startup ----------------------------------------------------------


def run_server(transport: str = "stdio") -> None:
    """Start the MCP server."""
    _ensure_federated()  # pre-load
    mcp.run(transport=transport)
