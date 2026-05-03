"""PrismRag MCP Server — unified vault + knowledge-graph tools.

Graph query tools (7):
  search_knowledge   BFS/DFS traversal from a query → related nodes
  explain_node       All info about a specific node + its neighbors
  trace_path         Shortest path between two nodes
  list_communities   Overview of all Leiden communities
  explore_community  Drill into a specific community's members
  list_namespaces    List all loaded knowledge graph namespaces
  check_drift        Detect stale vault→code mentions_symbol references

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
from prism_rag.retrieve.hybrid import hybrid_search
from prism_rag.retrieve.impact import format_impact_report, impact_bfs
from prism_rag.store.bm25_index import BM25Index
from prism_rag.store.federated import FederatedGraph
from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# -- Global state (loaded once at startup) ------------------------------------

_federated: FederatedGraph | None = None
_bm25_indices: dict[str, BM25Index] = {}
_embedding_stores: dict = {}   # dict[str, EmbeddingStore]
_embedder = None               # OllamaEmbedder | GeminiEmbedder | None
_cross_ns_probe = None         # CrossNamespaceProbe | None

mcp = FastMCP(
    "PrismRag",
    instructions=(
        "PrismRag is a graph-first RAG system for knowledge bases. "
        "Use search_knowledge for broad queries, explain_node for specific concepts, "
        "trace_path to understand how two concepts connect, list_communities / "
        "explore_community for structural overview, and list_namespaces to see "
        "all loaded knowledge graphs. "
        "IMPORTANT USAGE RULES: "
        "(1) When a tool returns an error (node not found, no path), report the error "
        "directly — do not silently substitute an alternative node or answer a different "
        "question. "
        "(2) When a query targets a specific node by name, use that exact name. If it is "
        "not found, say so — do not pick a similar-sounding alternative and present it as "
        "equivalent. "
        "(3) After trace_path, check the 'namespace' field of each step to confirm the "
        "path actually crosses the expected namespace boundaries (e.g. nimbus → code). "
        "A path that stays inside nimbus (vault docs → tags) is NOT a vault-to-code "
        "connection even if a tag label matches a code class name."
    ),
)


def _ensure_federated() -> FederatedGraph:
    global _federated, _bm25_indices, _embedding_stores, _embedder, _cross_ns_probe
    if _federated is None:
        settings = PrismRagSettings()

        # Create probe before load so bridge edges are captured at startup
        from prism_rag.store.cross_namespace_probe import CrossNamespaceProbe
        log_path = settings.data_dir / "cross_namespace_log.jsonl"
        _cross_ns_probe = CrossNamespaceProbe(log_path=log_path)

        _federated = FederatedGraph.load(
            settings.resolved_graphs, settings=settings, probe=_cross_ns_probe
        )
        logger.info(
            f"[mcp] federated loaded: {_federated.node_count} nodes, "
            f"{_federated.edge_count} edges across {len(_federated.namespaces)} namespaces"
        )

        # Build BM25 index per namespace
        for ns in _federated.namespaces:
            graph = _federated.get_graph(ns)
            if graph is not None:
                idx = BM25Index()
                idx.build(graph)
                _bm25_indices[ns] = idx
                logger.info(f"[mcp] bm25 built for {ns}: {graph.node_count} nodes")

        # Load embedding stores per namespace
        try:
            from prism_rag.store.embedding_store import EmbeddingStore
            for src in settings.resolved_graphs:
                lance_path = src.data_dir / "lance"
                if lance_path.exists():
                    store = EmbeddingStore(lance_path, dim=settings.embedding_dim)
                    if store.count() > 0:
                        _embedding_stores[src.namespace] = store
                        logger.info(f"[mcp] embedding store loaded for {src.namespace}: {store.count()} vectors")
        except Exception as exc:
            logger.warning(f"[mcp] embedding store unavailable: {exc}")

        # Create query-time embedder (graceful fallback if Ollama/Gemini unavailable)
        try:
            from prism_rag.ingest.embedder import OllamaEmbedder
            _embedder = OllamaEmbedder(
                model=settings.ollama_model,
                base_url=settings.ollama_host,
            )
            logger.info(f"[mcp] embedder ready: ollama/{settings.ollama_model}")
        except Exception as exc:
            logger.warning(f"[mcp] no embedder for hybrid search: {exc}")

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
    min_confidence: float = 0.0,
) -> str:
    """Search the knowledge graph for information about a topic.

    Uses hybrid BM25 + embedding + exact matching (RRF fusion) to find the
    best entry node, then traverses the graph up to the token budget.

    Args:
        query: Natural language query or node name (e.g., "session management", "Colony Coder")
        budget: Maximum tokens to return (default 4000)
        mode: Traversal mode — "bfs" (broad context) or "dfs" (follow chains)
        scope: Namespace to search. Use "nimbus" for vault design docs / notes,
               "code" for code implementation details, empty string to search both.
               When the query is about design intent or architecture, prefer scope="nimbus".
               When the query is about a specific code symbol or implementation, prefer scope="code".
        ontology_type: Filter results to nodes with this ontology_type (e.g., "decision",
                       "concept", "fact"). Empty string (default) = no filter.
        min_confidence: Skip edges below this confidence score during traversal (default 0.0).

    Returns:
        JSON with entry point, traversed nodes, and their content.
    """
    fg = _ensure_federated()

    # Hybrid entry-point selection: BM25 + embedding + exact, fused via RRF
    embed_fn = _embedder.embed_query if _embedder is not None else None
    target_namespaces = [scope] if scope else fg.namespaces
    entry_candidates: list[tuple[str, str]] = []   # [(ns, node_id), ...]
    for ns in target_namespaces:
        graph = fg.get_graph(ns)
        if graph is None:
            continue
        hits = hybrid_search(
            query,
            graph,
            bm25_index=_bm25_indices.get(ns),
            embed_fn=embed_fn,
            embedding_store=_embedding_stores.get(ns),
            top_k=5,
        )
        entry_candidates.extend((ns, nid) for nid in hits)

    # Always try exact/substring name resolution and prepend results.
    # When the query contains an exact symbol name, label-match precision beats
    # semantic similarity — prevents returning an unrelated node that happens to
    # score well on BM25/embedding while the target exists under its exact name.
    exact_entries = resolve_entry_points(fg, query, scope=scope or None)
    if exact_entries:
        existing = {(ns, nid) for ns, nid in entry_candidates}
        deduped = [(ns, nid) for ns, nid in exact_entries if (ns, nid) not in existing]
        entry_candidates = deduped + entry_candidates

    if not entry_candidates:
        return json.dumps({"error": f"No matching node for query: {query!r}"}, ensure_ascii=False)

    ns, entry_id = entry_candidates[0]
    graph = fg.get_graph(ns)

    if mode == "dfs":
        nodes = federated_dfs(fg, ns, entry_id, budget=budget, scope=scope or None,
                               min_confidence=min_confidence)
    else:
        nodes = federated_bfs(fg, ns, entry_id, budget=budget, scope=scope or None,
                               min_confidence=min_confidence)

    # Build node list — include all content-bearing kinds; filter by ontology_type if given
    _CONTENT_KINDS = frozenset({"note", "knowledge", "function", "class", "module", "flow"})
    node_list = []
    for n in nodes:
        if n.get("kind") not in _CONTENT_KINDS:
            continue
        if ontology_type and n.get("ontology_type") != ontology_type:
            continue
        entry: dict = {
            "id": f"{ns}::{n['id']}" if not fg.is_single else n["id"],
            "label": n.get("label", n["id"]),
            "kind": n.get("kind", "?"),
            "ontology_type": n.get("ontology_type"),
            "tokens": n.get("tokens", 0),
            "community": n.get("community_id", ""),
            "content": n.get("content", "")[:2000],
        }
        # For code nodes, surface structured metadata (signature, file location, callers)
        meta = n.get("metadata") or {}
        if meta:
            entry["signature"] = meta.get("signature")
            entry["file"] = f"{n.get('source_file', '')}:{meta.get('line_start')}-{meta.get('line_end')}"
            entry["docstring"] = meta.get("docstring", "")[:300] or None
        node_list.append(entry)

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
        If the node is not found, returns {"error": "Node not found: ..."}.
        Report this error directly — do not guess an alternative name and retry
        silently. Old names from renames/refactors (e.g. AgentLoader → EntityLoader)
        will not be found; report the absence rather than substituting.
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
        max_length: Maximum path length to search (default 5). For cross-namespace
            paths (e.g. vault doc → code class), use max_length=6 or higher since
            bridge edges add hops.

    Returns:
        JSON with the shortest path as a sequence of nodes and edges. Each step
        includes a "namespace" field showing which graph it belongs to.

        Usage rules:
        - If either endpoint returns "not found", report the error directly.
          Do not substitute a different node and answer a different question.
        - After receiving a path, check the "namespace" field of each step.
          A cross-namespace path (nimbus → code) must contain steps in both
          namespaces. A path that stays entirely within "nimbus" — even if it
          passes through a tag node whose label matches a code class name — is
          NOT a vault-to-code connection.
        - If no path exists, report "No path found" as-is. Do not fabricate
          an indirect connection or claim semantic proximity implies a graph path.
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
        src_qid = fg.qualify_id(src_ns, src_id)
        tgt_qid = fg.qualify_id(tgt_ns, tgt_id)

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


# -- Tool 7: impact ----------------------------------------------------------


@mcp.tool()
def impact(
    target: str,
    direction: str = "upstream",
    max_depth: int = 3,
    min_confidence: float = 0.7,
    scope: str = "",
    allowed_tiers: str = "EXTRACTED,INFERRED",
    allowed_edge_kinds: str = "",
    path_score_fn: str = "weakest_link",
) -> str:
    """Analyse the impact radius of changing a node.

    Answers: "If I change *target*, what else is affected?"

    Args:
        target: Node ID or label to analyse (supports ``namespace::id`` syntax).
        direction:
            "upstream"   — who depends on / calls / references target?
            "downstream" — what does target depend on / call / reference?
            "both"       — union of both directions.
        max_depth: Maximum traversal hops (default 3).
        min_confidence: Skip edges below this confidence score (default 0.7).
        scope: Restrict to this namespace (empty = search all).
        allowed_tiers: Comma-separated confidence tiers to follow.
            Default "EXTRACTED,INFERRED" excludes AMBIGUOUS edges.
            Pass "EXTRACTED,INFERRED,AMBIGUOUS" to include all.
        allowed_edge_kinds: Comma-separated edge kinds to follow (e.g.
            "calls,imports"). Empty string means follow all kinds.
        path_score_fn: Scoring mode — "weakest_link" (default) or
            "cumulative_decay".

    Returns:
        Human-readable impact report grouped by depth with path scores.
    """
    fg = _ensure_federated()
    entries = resolve_entry_points(fg, target, scope=scope or None)
    if not entries:
        return f"No node found matching: {target!r}"

    ns, node_id = entries[0]
    graph = fg.get_graph(ns)
    if graph is None:
        return f"Namespace {ns!r} not loaded."

    if direction not in ("upstream", "downstream", "both"):
        direction = "upstream"

    tiers: frozenset[str] | None = (
        frozenset(t.strip() for t in allowed_tiers.split(",") if t.strip())
        if allowed_tiers.strip()
        else None
    )
    kinds: frozenset[str] | None = (
        frozenset(k.strip() for k in allowed_edge_kinds.split(",") if k.strip())
        if allowed_edge_kinds.strip()
        else None
    )
    score_fn = path_score_fn if path_score_fn in ("weakest_link", "cumulative_decay") else "weakest_link"

    result = impact_bfs(
        graph,
        node_id,
        direction=direction,  # type: ignore[arg-type]
        max_depth=max_depth,
        min_confidence=min_confidence,
        allowed_tiers=tiers,
        allowed_edge_kinds=kinds,
        path_score_fn=score_fn,  # type: ignore[arg-type]
    )
    report = format_impact_report(graph, node_id, result, direction)  # type: ignore[arg-type]

    # Cross-namespace: scan vault graph for mentions_symbol edges pointing to this code node.
    # These edges live in the vault (nimbus) graph and are invisible to impact_bfs on the code graph.
    wants_mentions = kinds is None or "mentions_symbol" in kinds
    if wants_mentions and direction in ("upstream", "both") and ns != "nimbus":
        vault_g = fg.get_graph("nimbus")
        if vault_g is not None:
            mention_lines: list[str] = []
            for src, tgt, d in vault_g.g.edges(data=True):
                if d.get("relation") != "mentions_symbol":
                    continue
                if tgt != node_id:
                    continue
                tier = d.get("confidence", "INFERRED")
                if tiers and tier not in tiers:
                    continue
                score = d.get("confidence_score", 0.0)
                if score < min_confidence:
                    continue
                src_label = vault_g.g.nodes[src].get("label", src)
                mention_lines.append(f"  - {src_label} [{tier} score={score:.2f}]")
            if mention_lines:
                report += (
                    f"\n\n### Vault documents mentioning `{node_id.split('::')[-1]}` "
                    f"(cross-namespace mentions_symbol)\n"
                    + "\n".join(mention_lines)
                )

    return report


# -- Tool 8: list_cross_namespace_edges --------------------------------------


@mcp.tool()
def list_cross_namespace_edges(
    since: str = "",
    scope: str = "",
    min_confidence: float = 0.0,
    allowed_tiers: str = "",
) -> str:
    """List cross-namespace bridge edges tracked by the CrossNamespaceProbe.

    Args:
        since: ISO 8601 datetime string — return only edges first seen at or
            after this timestamp (e.g. "2026-04-30T00:00:00+00:00").
            Empty string means return all edges.
        scope: Restrict to edges where source or target node starts with this
            namespace prefix (e.g. "code" or "docs"). Empty means all.
        min_confidence: Minimum confidence score [0.0–1.0] (default 0.0 = no filter).
        allowed_tiers: Comma-separated confidence tiers to include
            (e.g. "EXTRACTED,INFERRED"). Empty means all tiers.

    Returns:
        JSON object with ``total`` count and ``edges`` list, each entry
        containing edge_id, source_node, target_node, edge_kind,
        confidence_tier, confidence, and first_seen_at.
    """
    _ensure_federated()
    if _cross_ns_probe is None:
        return json.dumps({"total": 0, "edges": [], "note": "CrossNamespaceProbe not initialised"})

    tiers: list[str] | None = (
        [t.strip() for t in allowed_tiers.split(",") if t.strip()]
        if allowed_tiers.strip()
        else None
    )

    if since.strip():
        try:
            from datetime import datetime
            since_dt = datetime.fromisoformat(since.strip())
            edges = _cross_ns_probe.list_new_cross_edges(since_dt)
        except ValueError:
            return json.dumps({"error": f"Invalid ISO datetime: {since!r}"})
    else:
        edges = _cross_ns_probe.list_cross_edges(
            min_confidence=min_confidence,
            allowed_tiers=tiers,
        )

    if scope.strip():
        scope_ns = scope.strip()
        edges = [
            e for e in edges
            if e.source_node.startswith(scope_ns) or e.target_node.startswith(scope_ns)
        ]

    return json.dumps({
        "total": len(edges),
        "edges": [e.to_dict() for e in edges],
    }, ensure_ascii=False, indent=2)


# -- Tool 9: check_drift --------------------------------------------------------


@mcp.tool()
def check_drift(
    vault_namespace: str = "nimbus",
    code_namespace: str = "code",
    include_valid: bool = False,
) -> str:
    """Detect stale symbol references in vault documents (documentation drift).

    Scans every mentions_symbol edge in the vault graph and checks whether the
    referenced code symbol still exists in the code graph. Returns dangling
    references where the target was renamed, moved, or deleted after the last
    link-symbols run.

    Note: this catches *post-link drift* only (symbol gone after link-symbols
    ran). If vault text references a symbol that was already absent at link
    time, no edge was ever created, so it won't appear here. Re-running
    link-symbols after fixing the rename will surface new links.

    Args:
        vault_namespace: Namespace of the vault (doc) graph. Default "nimbus".
        code_namespace: Namespace of the code graph. Default "code".
        include_valid: If True, also include confirmed-valid references in the
            output. Default False (stale-only is more actionable).

    Returns:
        JSON with stale_count, valid_count, stale_refs list (and optionally
        valid_refs). Each stale entry includes the source doc label, the
        referenced symbol name, confidence tier, score, and full target ID
        so the caller can cross-check or re-run link-symbols after a fix.
    """
    fg = _ensure_federated()
    vault_g = fg.get_graph(vault_namespace)
    code_g = fg.get_graph(code_namespace)

    if vault_g is None:
        return json.dumps(
            {"error": f"Vault namespace not loaded: {vault_namespace!r}"},
            ensure_ascii=False,
        )
    if code_g is None:
        return json.dumps(
            {"error": f"Code namespace not loaded: {code_namespace!r}"},
            ensure_ascii=False,
        )

    stale: list[dict] = []
    valid: list[dict] = []

    for src, tgt, d in vault_g.g.edges(data=True):
        if d.get("relation") != "mentions_symbol":
            continue

        src_data = vault_g.g.nodes[src]
        src_label = src_data.get("label", src)
        src_file = src_data.get("source_file", "")
        tier = d.get("confidence", "INFERRED")
        score = float(d.get("confidence_score", 0.0))
        # tgt is the full code node ID as stored in vault (e.g. "code::path::ClassName")
        symbol_name = tgt.split("::")[-1] if "::" in tgt else tgt

        entry = {
            "doc": src_label,
            "doc_file": src_file,
            "symbol": symbol_name,
            "target_id": tgt,
            "tier": tier,
            "score": round(score, 3),
            "status": "valid" if tgt in code_g.g else "not_found_in_code",
        }
        if tgt in code_g.g:
            valid.append(entry)
        else:
            stale.append(entry)

    # Sort stale by score descending so high-confidence broken links surface first
    stale.sort(key=lambda x: -x["score"])

    result: dict = {
        "stale_count": len(stale),
        "valid_count": len(valid),
        "stale_refs": stale,
    }
    if include_valid:
        valid.sort(key=lambda x: -x["score"])
        result["valid_refs"] = valid

    return json.dumps(result, ensure_ascii=False, indent=2)


# -- Register ported Obsidian MCP tools (vault_tools.py) ---------------------

from prism_rag.mcp_server.vault_tools import register_vault_tools  # noqa: E402
register_vault_tools(mcp)


# -- Server startup ----------------------------------------------------------


def run_server(transport: str = "stdio", port: int = 8102) -> None:
    """Start the MCP server.

    Args:
        transport: "stdio" (default, subprocess) or "sse" (HTTP Server-Sent Events).
        port: TCP port for SSE transport; ignored when transport="stdio".
    """
    _ensure_federated()  # pre-load
    if transport == "sse":
        # FastMCP's SSE host/port are controlled via .settings
        mcp.settings.port = port
        mcp.settings.host = "127.0.0.1"
    mcp.run(transport=transport)
