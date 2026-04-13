"""PrismRag MCP Server — 5 tools for knowledge graph queries.

Tools:
  search_knowledge  BFS/DFS traversal from a query → related nodes
  explain_node      All info about a specific node + its neighbors
  trace_path        Shortest path between two nodes
  list_communities  Overview of all Leiden communities
  explore_community Drill into a specific community's members

Usage:
  prism-rag serve                    # start MCP stdio server
  prism-rag serve --transport sse    # SSE mode (for network access)

The server loads graph.json from PRISM_DATA_DIR on startup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx
from mcp.server.fastmcp import FastMCP

from prism_rag.config import PrismRagSettings
from prism_rag.retrieve.bfs import bfs_traverse
from prism_rag.retrieve.dfs import dfs_traverse
from prism_rag.retrieve.entry import resolve_entry_point
from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# ── Global state (loaded once at startup) ────────────────────────────

_graph: KnowledgeGraph | None = None

mcp = FastMCP(
    "PrismRag",
    instructions=(
        "PrismRag is a graph-first RAG system for the NimbusVault knowledge base. "
        "Use search_knowledge for broad queries, explain_node for specific concepts, "
        "trace_path to understand how two concepts connect, and list_communities / "
        "explore_community for structural overview."
    ),
)


def _ensure_graph() -> KnowledgeGraph:
    global _graph
    if _graph is None:
        settings = PrismRagSettings()
        graph_path = settings.graph_path
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Graph not found at {graph_path}. Run 'prism-rag ingest' first."
            )
        _graph = KnowledgeGraph.load(graph_path)
        logger.info(
            f"[mcp] loaded graph: {_graph.node_count} nodes, "
            f"{_graph.edge_count} edges, {len(_graph.communities)} communities"
        )
    return _graph


def _node_summary(graph: KnowledgeGraph, node_id: str, include_content: bool = False) -> dict:
    """Build a concise summary dict for a single node."""
    data = graph.g.nodes.get(node_id, {})
    summary = {
        "id": node_id,
        "label": data.get("label", node_id),
        "kind": data.get("kind", "?"),
        "tokens": data.get("tokens", 0),
        "community": data.get("community_id", ""),
        "degree": graph.degree(node_id),
    }
    if data.get("source_file"):
        summary["source_file"] = data["source_file"]
    if include_content and data.get("content"):
        summary["content"] = data["content"]
    return summary


# ── Tool 1: search_knowledge ─────────────────────────────────────────


@mcp.tool()
def search_knowledge(
    query: str,
    budget: int = 4000,
    mode: str = "bfs",
) -> str:
    """Search the knowledge graph for information about a topic.

    Finds the best matching entry node, then traverses the graph
    collecting related nodes up to the token budget.

    Args:
        query: Natural language query or node name (e.g., "session management", "Colony Coder")
        budget: Maximum tokens to return (default 4000)
        mode: Traversal mode — "bfs" (broad context) or "dfs" (follow chains)

    Returns:
        JSON with entry point, traversed nodes, and their content.
    """
    graph = _ensure_graph()
    entry = resolve_entry_point(graph, query)
    if entry is None:
        return json.dumps({"error": f"No matching node for query: {query!r}"}, ensure_ascii=False)

    if mode == "dfs":
        nodes = dfs_traverse(graph, entry, budget=budget)
    else:
        nodes = bfs_traverse(graph, entry, budget=budget)

    result = {
        "entry_point": _node_summary(graph, entry),
        "total_nodes": len(nodes),
        "total_tokens": sum(n.get("tokens", 0) for n in nodes),
        "nodes": [
            {
                "id": n["id"],
                "label": n.get("label", n["id"]),
                "kind": n.get("kind", "?"),
                "tokens": n.get("tokens", 0),
                "community": n.get("community_id", ""),
                "content": n.get("content", "")[:2000],  # truncate for MCP response size
            }
            for n in nodes
            if n.get("kind") == "note"  # only return note content, not tags
        ],
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ── Tool 2: explain_node ─────────────────────────────────────────────


@mcp.tool()
def explain_node(node: str) -> str:
    """Get detailed information about a specific node and its connections.

    Args:
        node: Node ID, label, or partial name to look up.

    Returns:
        JSON with node details, incoming edges, outgoing edges, and community info.
    """
    graph = _ensure_graph()
    node_id = resolve_entry_point(graph, node)
    if node_id is None:
        return json.dumps({"error": f"Node not found: {node!r}"}, ensure_ascii=False)

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
        "outgoing_edges": outgoing,
        "incoming_edges": incoming,
        "community": community_info,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ── Tool 3: trace_path ───────────────────────────────────────────────


@mcp.tool()
def trace_path(from_node: str, to_node: str, max_length: int = 5) -> str:
    """Find the shortest path between two nodes in the knowledge graph.

    Args:
        from_node: Starting node (ID, label, or partial name)
        to_node: Ending node (ID, label, or partial name)
        max_length: Maximum path length to search (default 5)

    Returns:
        JSON with the shortest path as a sequence of nodes and edges.
    """
    graph = _ensure_graph()
    src = resolve_entry_point(graph, from_node)
    tgt = resolve_entry_point(graph, to_node)

    if src is None:
        return json.dumps({"error": f"Source node not found: {from_node!r}"}, ensure_ascii=False)
    if tgt is None:
        return json.dumps({"error": f"Target node not found: {to_node!r}"}, ensure_ascii=False)

    # Use undirected view for pathfinding (wikilinks are directional but we want reachability)
    undirected = graph.g.to_undirected()
    try:
        path = nx.shortest_path(undirected, source=src, target=tgt)
    except nx.NetworkXNoPath:
        return json.dumps({
            "error": "No path found",
            "from": _node_summary(graph, src),
            "to": _node_summary(graph, tgt),
        }, ensure_ascii=False)

    if len(path) - 1 > max_length:
        return json.dumps({
            "error": f"Shortest path has {len(path)-1} hops (max_length={max_length})",
            "from": _node_summary(graph, src),
            "to": _node_summary(graph, tgt),
        }, ensure_ascii=False)

    # Build path description
    steps = []
    for i, node_id in enumerate(path):
        step = _node_summary(graph, node_id)
        if i < len(path) - 1:
            next_id = path[i + 1]
            # Find edge (either direction)
            if graph.g.has_edge(node_id, next_id):
                edge_data = graph.g.edges[node_id, next_id]
            elif graph.g.has_edge(next_id, node_id):
                edge_data = graph.g.edges[next_id, node_id]
            else:
                edge_data = {}
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


# ── Tool 4: list_communities ─────────────────────────────────────────


@mcp.tool()
def list_communities() -> str:
    """List all Leiden communities in the knowledge graph.

    Returns an overview with community labels, sizes, god nodes, and density.
    """
    graph = _ensure_graph()

    communities = []
    for comm in sorted(graph.communities.values(), key=lambda c: -c.member_count):
        communities.append({
            "id": comm.id,
            "label": comm.label,
            "member_count": comm.member_count,
            "internal_density": comm.internal_density,
            "god_nodes": [
                graph.g.nodes[n].get("label", n) for n in comm.god_nodes
            ],
        })

    result = {
        "total_communities": len(communities),
        "total_nodes": graph.node_count,
        "total_edges": graph.edge_count,
        "communities": communities,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ── Tool 5: explore_community ────────────────────────────────────────


@mcp.tool()
def explore_community(community: str) -> str:
    """Explore a specific community's members and connections.

    Args:
        community: Community ID (e.g., "community_000") or label substring.

    Returns:
        JSON with all members, internal edges, and bridge edges to other communities.
    """
    graph = _ensure_graph()

    # Resolve community
    target_comm = None
    community_lower = community.lower()
    for comm in graph.communities.values():
        if comm.id == community or community_lower in comm.label.lower():
            target_comm = comm
            break

    if target_comm is None:
        return json.dumps({
            "error": f"Community not found: {community!r}",
            "available": [
                {"id": c.id, "label": c.label} for c in graph.communities.values()
            ],
        }, ensure_ascii=False)

    # Members
    members = [
        _node_summary(graph, nid)
        for nid, data in graph.g.nodes(data=True)
        if data.get("community_id") == target_comm.id
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
            "id": target_comm.id,
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


# ── Tool 6: write_note ───────────────────────────────────────────────


@mcp.tool()
def write_note(path: str, content: str, cas_hash: str = "") -> str:
    """Write a note to the vault (create or overwrite).

    After writing, the knowledge graph is automatically updated.

    Args:
        path: Relative path within the vault (e.g., "设计细节/new_doc.md")
        content: Full markdown content (including frontmatter if desired)
        cas_hash: Empty string = create new file (fails if exists).
                  Non-empty = overwrite (fails if hash doesn't match).
                  Get cas_hash from read_note first.

    Returns:
        JSON with new cas_hash and graph update stats.
    """
    from prism_rag.vault_ops.cas import compute_hash, get_file_lock, verify_cas
    from prism_rag.vault_ops.errors import VaultErrorCode, fail, ok
    from prism_rag.vault_ops.vault import Vault
    from prism_rag.ingest.incremental import ingest_file

    settings = PrismRagSettings()
    vault = Vault(settings.vault_path)

    resolved = vault.resolve_path(path)
    if isinstance(resolved, dict):
        return json.dumps(resolved, ensure_ascii=False)

    import asyncio
    lock = get_file_lock(resolved)

    # Sync write (MCP tools are sync in FastMCP)
    expected = cas_hash if cas_hash else None
    is_valid, actual = verify_cas(resolved, expected)

    if not is_valid:
        if expected is None:
            return json.dumps(fail(
                VaultErrorCode.ALREADY_EXISTS,
                f"文件已存在: {path}。先用 read_note 获取 cas_hash。",
                actual_hash=actual,
            ), ensure_ascii=False)
        return json.dumps(fail(
            VaultErrorCode.CONFLICT,
            f"CAS 冲突: 文件已被修改。",
            expected_hash=expected, actual_hash=actual,
        ), ensure_ascii=False)

    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    new_hash = compute_hash(content)

    # Incrementally update graph
    graph_stats = {}
    try:
        graph_stats = ingest_file(
            resolved, settings=settings, skip_embed=True, skip_persist=False,
        )
        # Reload the updated graph into memory
        global _graph
        _graph = KnowledgeGraph.load(settings.graph_path)
    except Exception as e:
        logger.warning(f"[write_note] graph update failed: {e}")
        graph_stats = {"error": str(e)}

    result = {
        "status": "ok",
        "data": {"cas_hash": new_hash, "path": path},
        "graph_update": graph_stats,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ── Tool 7: read_note ────────────────────────────────────────────────


@mcp.tool()
def read_note(path: str) -> str:
    """Read a note's content and cas_hash (needed before writing).

    Args:
        path: Relative path within the vault (e.g., "设计细节/some_doc.md")

    Returns:
        JSON with content, frontmatter, and cas_hash.
    """
    from prism_rag.vault_ops.cas import compute_hash
    from prism_rag.vault_ops.errors import VaultErrorCode, fail, ok
    from prism_rag.vault_ops.vault import Vault

    settings = PrismRagSettings()
    vault = Vault(settings.vault_path)

    resolved = vault.resolve_path(path)
    if isinstance(resolved, dict):
        return json.dumps(resolved, ensure_ascii=False)

    if not resolved.exists():
        return json.dumps(fail(VaultErrorCode.NOT_FOUND, f"文件不存在: {path}"), ensure_ascii=False)

    content = resolved.read_text(encoding="utf-8")
    cas = compute_hash(content)

    # Parse frontmatter if present
    try:
        import frontmatter
        post = frontmatter.loads(content)
        fm = dict(post.metadata or {})
    except Exception:
        fm = {}

    result = ok(data={
        "path": path,
        "content": content,
        "cas_hash": cas,
        "frontmatter": fm,
    })
    return json.dumps(result, ensure_ascii=False, indent=2)


# ── Server startup ───────────────────────────────────────────────────


def run_server(transport: str = "stdio") -> None:
    """Start the MCP server."""
    _ensure_graph()  # pre-load graph
    mcp.run(transport=transport)
