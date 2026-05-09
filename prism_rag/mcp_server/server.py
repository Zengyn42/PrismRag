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
from prism_rag.store.registry import Registry

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
        "PrismRag: graph-first RAG system for Obsidian vaults and code repositories. "
        "Exposes graph query tools (search, explain, path, communities, impact, drift) "
        "and vault CRUD tools (read, write, patch, move, delete, tags, links). "
        "Read prismrag://namespaces to see which namespace covers which codebase or vault."
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

    # Code graph nodes embed the ns:: prefix in their stored IDs.
    # Try bare_id first, then the reconstructed full ID.
    node_id = bare_id
    if node_id not in graph.g:
        prefixed = f"{ns}::{bare_id}"
        if prefixed in graph.g:
            node_id = prefixed

    if node_id not in graph.g:
        return {"id": bare_id, "namespace": ns, "label": bare_id, "kind": "?"}

    summary = _node_summary(graph, node_id)
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
    """Search the knowledge graph by topic, symbol name, or natural language query.
    Returns JSON with entry node and traversed neighbors up to the token budget.
    Does NOT search raw file content — use search_files() for exact keyword matching.

    Ranking: BM25 keyword + embedding vector + exact name match, fused via RRF.
    scope="nimbus" for vault/design-doc nodes; scope="code" for code symbols; scope="" for both.
    mode="bfs" for broad topic context; mode="dfs" to follow call/reference chains.
    ontology_type filters to a specific type e.g. "decision", "concept", "fact" (see prismrag://schema).
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
    """Return all edges (incoming/outgoing), community membership, and full content for a single graph node.
    Returns JSON with node metadata, outgoing_edges list, incoming_edges list, and community info.
    Does NOT rank or filter edges — returns everything. Use search_knowledge() for ranked traversal from a starting point.

    scope narrows lookup to a single namespace; omit to search all namespaces.
    Node lookup is by exact label or ID. Returns an error object if the node is not found.
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
def trace_path(
    from_node: str,
    to_node: str,
    max_length: int = 5,
    scope: str = "",
    min_confidence: float = 0.0,
) -> str:
    """Find the shortest undirected path between two graph nodes, including cross-namespace paths via bridge edges.
    Returns JSON with path_length and steps list (each step has node metadata and edge_to_next relation).
    Does NOT return multiple paths or ranked alternatives — returns the single shortest path only.

    scope="" (default) searches across all namespaces; set scope to restrict to one namespace.
    Each step includes a namespace field — a vault→code path must contain steps in both "nimbus" and "code".
    max_length=6+ recommended for cross-namespace queries since bridge edges add hops.
    Returns an error object if no path exists within max_length.
    """
    fg = _ensure_federated()
    scope_arg = scope or None
    src_entries = resolve_entry_points(fg, from_node, scope=scope_arg)
    tgt_entries = resolve_entry_points(fg, to_node, scope=scope_arg)

    if not src_entries:
        return json.dumps({"error": f"Source node not found: {from_node!r}"}, ensure_ascii=False)
    if not tgt_entries:
        return json.dumps({"error": f"Target node not found: {to_node!r}"}, ensure_ascii=False)

    src_ns, src_id = src_entries[0]
    tgt_ns, tgt_id = tgt_entries[0]

    if scope:
        # Scope-restricted: never use the unified view, even in multi-graph mode
        graph = fg.get_graph(scope)
        if graph is None:
            return json.dumps({"error": f"Unknown scope namespace: {scope!r}"}, ensure_ascii=False)
        undirected = graph.g.to_undirected()
        src_qid, tgt_qid = src_id, tgt_id
    elif fg.is_single:
        # Single-graph: use original graph directly (no prefixing)
        graph = fg.get_graph(src_ns)
        undirected = graph.g.to_undirected()
        src_qid, tgt_qid = src_id, tgt_id
    else:
        # Multi-graph: use unified view
        undirected = fg.unified_view.to_undirected()
        src_qid = fg.qualify_id(src_ns, src_id)
        tgt_qid = fg.qualify_id(tgt_ns, tgt_id)

    # Apply edge confidence filter as a SubGraph view if needed
    if min_confidence > 0.0:
        def _edge_passes(u: str, v: str) -> bool:
            data = undirected.edges.get((u, v), {})
            return float(data.get("confidence_score", 1.0)) >= min_confidence
        search_graph = nx.subgraph_view(undirected, filter_edge=_edge_passes)
    else:
        search_graph = undirected

    try:
        path = nx.shortest_path(search_graph, source=src_qid, target=tgt_qid)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
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


# -- Tool 4: communities (merged list_communities + explore_community) --------


@mcp.tool()
def communities(namespace: str = "", community_id: str = "", ontology_type: str = "") -> str:
    """List all Leiden communities in a namespace, or return all members and bridge edges for one community.
    No community_id → returns JSON with community list (label, member_count, god_nodes, internal_density).
    With community_id → returns JSON with members list and top_bridge_edges to neighboring communities.
    Does NOT traverse edges — use search_knowledge() for topic-based traversal within a community.

    community_id accepts exact ID (e.g. "community_000"), namespace::ID, or label substring match.
    """
    fg = _ensure_federated()

    if not community_id:
        # List mode — aggregate across namespaces (or filter to one)
        result_communities = []
        total_nodes = 0
        total_edges = 0
        target_ns_list = [namespace] if namespace else fg.namespaces
        for ns in target_ns_list:
            graph = fg.get_graph(ns)
            if graph is None:
                continue
            total_nodes += graph.node_count
            total_edges += graph.edge_count
            for comm in sorted(graph.communities.values(), key=lambda c: -c.member_count):
                if ontology_type:
                    has_match = any(
                        graph.g.nodes[nid].get("ontology_type") == ontology_type
                        for nid in graph.g.nodes
                        if graph.g.nodes[nid].get("community_id") == comm.id
                    )
                    if not has_match:
                        continue
                comm_id_str = comm.id if fg.is_single else f"{ns}::{comm.id}"
                result_communities.append({
                    "id": comm_id_str,
                    "namespace": ns,
                    "label": comm.label,
                    "member_count": comm.member_count,
                    "internal_density": comm.internal_density,
                    "god_nodes": [graph.g.nodes[n].get("label", n) for n in comm.god_nodes],
                })
        return json.dumps({
            "total_communities": len(result_communities),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "communities": result_communities,
        }, ensure_ascii=False, indent=2)

    # Drill-in mode — resolve specific community
    if "::" in community_id:
        ns_hint, _, comm_query = community_id.partition("::")
    else:
        ns_hint = namespace or None
        comm_query = community_id

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
        available = []
        for ns in fg.namespaces:
            g = fg.get_graph(ns)
            for c in g.communities.values():
                cid = c.id if fg.is_single else f"{ns}::{c.id}"
                available.append({"id": cid, "label": c.label})
        return json.dumps({
            "error": f"Community not found: {community_id!r}",
            "available": available,
        }, ensure_ascii=False)

    graph = target_graph
    members = [
        _node_summary(graph, nid)
        for nid, data in graph.g.nodes(data=True)
        if data.get("community_id") == target_comm.id
        and (not ontology_type or data.get("ontology_type") == ontology_type)
    ]
    member_ids = {m["id"] for m in members}
    internal_edges = []
    bridge_edges = []
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

    return json.dumps({
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
    }, ensure_ascii=False, indent=2)


# -- Tool 6: list_namespaces -------------------------------------------------


@mcp.tool()
def list_namespaces() -> str:
    """Return all loaded knowledge graph namespaces with node/edge/community counts, indexed directories, and sample node IDs.
    Returns JSON with namespaces list (per-namespace stats) and total_nodes across all namespaces.
    Does NOT reload or refresh the graph — reflects what was loaded at server startup.
    """
    fg = _ensure_federated()
    settings = PrismRagSettings()
    src_map = {src.namespace: src for src in settings.resolved_graphs}

    namespaces = []
    for ns in fg.namespaces:
        g = fg.get_graph(ns)
        src = src_map.get(ns)
        entry: dict = {
            "namespace": ns,
            "nodes": g.node_count,
            "edges": g.edge_count,
            "communities": len(g.communities),
        }
        # Detect actual indexed top-level dirs from node IDs (more accurate than config).
        # Skip stub nodes that belong to other namespaces (e.g. code:: stubs in nimbus).
        other_ns_prefixes = tuple(f"{other}::" for other in fg.namespaces if other != ns)
        own_prefix = f"{ns}::"
        top_dirs: dict[str, int] = {}
        for nid in g.g.nodes():
            if other_ns_prefixes and nid.startswith(other_ns_prefixes):
                continue  # stub node from another namespace
            bare = nid[len(own_prefix):] if nid.startswith(own_prefix) else nid
            parts = bare.split("/")
            if len(parts) >= 2:
                top_dirs[parts[0]] = top_dirs.get(parts[0], 0) + 1
        if top_dirs:
            entry["indexed_dirs"] = sorted(top_dirs, key=lambda d: -top_dirs[d])[:8]
        # Sample node IDs so callers can identify which project is indexed
        sample_nodes = []
        for nid in g.g.nodes():
            if other_ns_prefixes and nid.startswith(other_ns_prefixes):
                continue
            bare = nid[len(own_prefix):] if nid.startswith(own_prefix) else nid
            if "/" in bare:
                sample_nodes.append(nid)
            if len(sample_nodes) >= 3:
                break
        if sample_nodes:
            entry["sample_node_ids"] = sample_nodes
        namespaces.append(entry)

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
    """Return the blast radius of changing a graph node: affected symbols grouped by depth, confidence tiers, and cross-namespace vault mentions.
    Returns a Markdown report with affected symbols organized by hop depth and a vault mentions section for cross-namespace references.
    Does NOT modify the graph — read-only analysis.

    direction: "upstream" (callers/dependents), "downstream" (dependencies), or "both".
    allowed_tiers: comma-separated confidence tiers to include; default "EXTRACTED,INFERRED" excludes AMBIGUOUS edges.
    allowed_edge_kinds: comma-separated edge kind filter e.g. "calls,imports"; empty includes all edge kinds.
    path_score_fn: "weakest_link" scores by minimum edge confidence along the path; "cumulative_decay" multiplies scores.
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

    # Symmetric cross-namespace: from a vault doc, scan outgoing mentions_symbol
    # edges and enrich the code targets with real metadata from the code graph.
    # impact_bfs sees these edges, but their endpoints are stub nodes in the
    # vault graph (label = qualified ID, no signature/file). The code graph has
    # the real data.
    if wants_mentions and direction in ("downstream", "both") and ns == "nimbus":
        code_g = fg.get_graph("code")
        if code_g is not None:
            sym_lines: list[str] = []
            for src, tgt, d in graph.g.edges(data=True):
                if d.get("relation") != "mentions_symbol":
                    continue
                if src != node_id:
                    continue
                tier = d.get("confidence", "INFERRED")
                if tiers and tier not in tiers:
                    continue
                score = float(d.get("confidence_score", 0.0))
                if score < min_confidence:
                    continue
                tgt_data = code_g.g.nodes.get(tgt, {})
                tgt_label = tgt_data.get("label", tgt.split("::")[-1])
                tgt_kind = tgt_data.get("kind", "?")
                meta = tgt_data.get("metadata") or {}
                sig = meta.get("signature", "")
                sig_str = f" — `{sig}`" if sig else ""
                sym_lines.append(
                    f"  - **{tgt_label}** (kind={tgt_kind}){sig_str} [{tier} score={score:.2f}]"
                )
            if sym_lines:
                report += (
                    f"\n\n### Code symbols mentioned by `{node_id}` "
                    f"(cross-namespace mentions_symbol)\n"
                    + "\n".join(sym_lines)
                )

    return report


# -- Tool 8: check_drift --------------------------------------------------------


@mcp.tool()
def check_drift(
    vault_namespace: str = "nimbus",
    code_namespace: str = "code",
    include_valid: bool = False,
) -> str:
    """Scan all mentions_symbol edges in the vault graph and check whether each referenced code symbol still exists in the code graph.
    Returns JSON with stale_count, valid_count, and stale_refs list (doc label, symbol name, target_id, confidence tier, score).
    Does NOT detect symbols that were never linked — only catches post-link drift (symbol existed when link-symbols ran, then was renamed or deleted).

    Set include_valid=True to also include valid_refs in the response.
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


# -- Tool 9: pending_edges (merged list_pending_edges + get_pending_edge_context) --


@mcp.tool()
def pending_edges(edge_id: str = "", top_n: int = 10, sort_by: str = "confidence") -> str:
    """List or inspect cross-namespace edges awaiting human review in the EdgeClassifier inbox.
    No edge_id → returns JSON with pending queue (top_n entries sorted by sort_by: confidence, created_at, or consecutive_seen).
    With edge_id → returns JSON with vault_context (label, frontmatter, content_excerpt) and code_context (id, kind, metadata) for that entry.
    Does NOT apply a decision — use review_pending_edge() to approve or reject an entry.
    """
    from prism_rag.inbox.store import InboxStore
    settings = PrismRagSettings()
    srcs = settings.resolved_graphs
    nimbus_src = next((s for s in srcs if s.namespace == "nimbus"), srcs[0])
    inbox = InboxStore(nimbus_src.data_dir / "inbox.jsonl")

    if not edge_id:
        return json.dumps(inbox.list_pending(top_n=top_n, sort_by=sort_by), ensure_ascii=False, indent=2)

    entry = inbox.get(edge_id)
    if entry is None:
        return json.dumps({"status": "error", "msg": f"unknown edge_id: {edge_id}"})

    fg = _ensure_federated()
    sem_src = entry["source"]
    sem_tgt = entry["target"]
    bare_src = sem_src.split("::", 1)[1] if "::" in sem_src else sem_src
    nimbus = fg.get_graph("nimbus") if fg else None
    code_g = fg.get_graph("code") if fg else None
    vault_data = nimbus.g.nodes.get(bare_src, {}) if nimbus else {}
    code_data = code_g.g.nodes.get(sem_tgt, {}) if code_g else {}
    return json.dumps({
        "status": "ok",
        "edge_id": edge_id,
        "confidence": entry["confidence"],
        "model_id": entry["model_id"],
        "vault_context": {
            "id": bare_src,
            "label": vault_data.get("label", bare_src),
            "frontmatter": vault_data.get("frontmatter", {}),
            "content_excerpt": (vault_data.get("content", "") or "")[:500],
        },
        "code_context": {
            "id": sem_tgt,
            "label": code_data.get("label", sem_tgt),
            "kind": code_data.get("kind", "?"),
            "metadata": code_data.get("metadata", {}),
        },
    }, ensure_ascii=False, indent=2)


# -- Tool 10: review_pending_edge --------------------------------------------


@mcp.tool()
def review_pending_edge(edge_id: str, decision: str, note: str = "") -> str:
    """Apply an approve or reject decision to a pending cross-namespace edge in the EdgeClassifier inbox.
    Returns JSON with status, edge_id, decision, and note.
    Approved edges are committed to the vault graph immediately and the graph file is saved.
    decision must be "approve" or "reject"; any other value returns an error.
    """
    from prism_rag.inbox.store import InboxStore, StatusTransitionError
    from prism_rag.inbox.approval import apply_decision

    settings = PrismRagSettings()
    srcs = settings.resolved_graphs
    nimbus_src = next((s for s in srcs if s.namespace == "nimbus"), None)
    if nimbus_src is None:
        return json.dumps({"status": "error", "msg": "no nimbus namespace configured"})

    inbox = InboxStore(nimbus_src.data_dir / "inbox.jsonl")
    if inbox.get(edge_id) is None:
        return json.dumps({"status": "error", "msg": f"unknown edge_id: {edge_id}"})
    if decision not in ("approve", "reject"):
        return json.dumps({"status": "error", "msg": f"decision must be approve or reject; got {decision!r}"})

    fg = _ensure_federated()

    try:
        apply_decision(edge_id, decision, note,
                       inbox=inbox, fg=fg, src=nimbus_src,
                       decided_by="user_via_mcp")
    except StatusTransitionError as exc:
        return json.dumps({"status": "error", "msg": str(exc)})

    inbox.save_atomic()
    if decision == "approve":
        nimbus = fg.get_graph("nimbus")
        if nimbus is not None:
            nimbus.save(nimbus_src.graph_path)

    return json.dumps({"status": "ok", "edge_id": edge_id, "decision": decision, "note": note})


# -- Tool: alloc_knowledge_id --------------------------------------------------

@mcp.tool()
def alloc_knowledge_id(count: int = 1) -> str:
    """Allocate one or more KNOW-IDs for new knowledge nodes.
    Returns JSON with an "ids" list of allocated KNOW-IDs (format: KNOW-NNNNNN).
    IDs are globally unique and never reused. Use before calling atomize_propose.
    NOT a search tool — use search_knowledge() for finding existing nodes.
    """
    settings = PrismRagSettings()
    registry_path = settings.data_dir / "registry.json"
    reg = Registry(registry_path)
    ids = reg.batch_alloc(max(1, count))
    return json.dumps({"ids": ids}, ensure_ascii=False)


# -- Tool: list_knowledge_nodes ------------------------------------------------

@mcp.tool()
def list_knowledge_nodes(namespace: str = "") -> str:
    """List all knowledge nodes (kind="knowledge") in the graph.
    Returns JSON with a "nodes" list, each entry having id, label, knowledge_id, namespace.
    namespace="" searches all namespaces; namespace="nimbus" restricts to vault docs.
    NOT a full search — returns ALL knowledge nodes. Use search_knowledge() for topic queries.
    """
    fg = _ensure_federated()
    target_namespaces = [namespace] if namespace else fg.namespaces
    results = []
    for ns in target_namespaces:
        graph = fg.get_graph(ns)
        if graph is None:
            continue
        for node_id, data in graph.g.nodes(data=True):
            if data.get("kind") == "knowledge":
                results.append({
                    "id": node_id,
                    "label": data.get("label", node_id),
                    "knowledge_id": data.get("knowledge_id"),
                    "namespace": ns,
                    "tokens": data.get("tokens", 0),
                })
    return json.dumps({"nodes": results, "total": len(results)}, ensure_ascii=False)


# -- MCP Resources -----------------------------------------------------------
#
# Large reference material is exposed as resources (read on demand) rather than
# baked into tool descriptions. This keeps the cold-start tool-list context
# small (~400 tokens saved) while the data remains accessible when needed.


@mcp.resource(
    "prismrag://namespaces",
    name="namespaces",
    description="Overview of all loaded knowledge graph namespaces: node/edge counts, "
                "community counts, indexed directories, and sample node IDs. "
                "Read this first to understand what each namespace covers before querying.",
    mime_type="application/json",
)
def resource_namespaces() -> str:
    """Return namespace stats — same data as list_namespaces() but as a resource."""
    return list_namespaces()


@mcp.resource(
    "prismrag://schema",
    name="schema",
    description="PrismRag graph schema: node kinds, edge relation types, "
                "ontology_type vocabulary, and confidence tier definitions. "
                "Read when constructing ontology_type or allowed_tiers filters.",
    mime_type="application/json",
)
def resource_schema() -> str:
    schema = {
        "node_kinds": {
            "note":      "Obsidian markdown note (vault)",
            "knowledge": "Atomic knowledge chunk extracted from a note",
            "function":  "Python function or method (code)",
            "class":     "Python class definition (code)",
            "module":    "Python module / file (code)",
            "flow":      "Execution flow / entry point (code)",
        },
        "edge_relations": {
            "mentions_symbol":    "vault doc → code symbol (cross-namespace)",
            "semantically_similar_to": "embedding-based similarity edge",
            "calls":              "function/method calls another",
            "imports":            "module imports another",
            "defines":            "module defines a class or function",
            "inherits":           "class inherits from another",
            "part_of":            "knowledge chunk belongs to a note",
            "references":         "note references another note via wikilink",
            "tagged_with":        "note has a frontmatter tag",
            "BRIDGE":             "cross-namespace embedding bridge",
        },
        "ontology_types": [
            "decision", "concept", "fact", "procedure", "reference",
            "question", "observation", "hypothesis",
        ],
        "confidence_tiers": {
            "EXTRACTED":  "directly parsed from source (highest confidence)",
            "INFERRED":   "inferred by EdgeClassifier from context",
            "AMBIGUOUS":  "low-confidence inference (excluded by default)",
        },
        "confidence_score_range": "0.0–1.0 (float); default filters: min_confidence=0.7",
    }
    import json
    return json.dumps(schema, ensure_ascii=False, indent=2)


@mcp.resource(
    "prismrag://communities/{namespace}",
    name="communities",
    description="All Leiden communities in the given namespace with member counts, "
                "god nodes, and internal density. Namespace examples: 'nimbus', 'code'. "
                "Read instead of calling list_communities() for a one-shot overview.",
    mime_type="application/json",
)
def resource_communities(namespace: str) -> str:
    """Return community listing for one namespace."""
    import json
    fg = _ensure_federated()
    graph = fg.get_graph(namespace)
    if graph is None:
        available = list(fg.namespaces)
        return json.dumps(
            {"error": f"Unknown namespace: {namespace!r}", "available": available},
            ensure_ascii=False,
        )
    communities = []
    for comm in sorted(graph.communities.values(), key=lambda c: -c.member_count):
        communities.append({
            "id": comm.id,
            "label": comm.label,
            "member_count": comm.member_count,
            "internal_density": comm.internal_density,
            "god_nodes": [graph.g.nodes[n].get("label", n) for n in comm.god_nodes],
        })
    return json.dumps({
        "namespace": namespace,
        "total_communities": len(communities),
        "communities": communities,
    }, ensure_ascii=False, indent=2)


@mcp.resource(
    "prismrag://usage",
    name="usage",
    description="Workflow guide for PrismRag: tool selection, CAS write protocol, "
                "cross-namespace query patterns, and common error responses. "
                "Read once per session before starting complex queries.",
    mime_type="text/markdown",
)
def resource_usage() -> str:
    return """\
# PrismRag Usage Guide

## Tool Selection

| Goal | Tool |
|------|------|
| Find info about a topic | `search_knowledge` |
| Get all edges for a specific node | `explain_node` |
| Connect two concepts | `trace_path` |
| Structural overview | read `prismrag://communities/{namespace}` |
| What namespaces are loaded? | read `prismrag://namespaces` |
| What changed recently? | `list_cross_namespace_edges(since=...)` |
| Read a vault note | `read_note` |
| Write/edit a vault note | `write_note` / `patch_note` |

## Namespace Routing

- `scope="nimbus"` — vault design docs, architecture notes, meeting notes
- `scope="code"` — code symbols (functions, classes, modules)
- `scope=""` — federated search across both (use for cross-namespace queries)

When querying a specific code symbol (function/class name), always use `scope="code"`.
When querying design intent or architecture, use `scope="nimbus"`.

## CAS Write Protocol (mandatory for all vault writes)

1. `read_note(path)` → get `cas_hash`
2. Pass `cas_hash` to `write_note` / `patch_note` / `update_frontmatter`
3. If CAS conflict error, re-read and retry

NEVER write without a fresh `cas_hash` — this prevents overwriting concurrent edits.

## Cross-Namespace Queries

To trace from a vault doc to a code symbol:
1. `trace_path(from_node="Doc Title", to_node="ClassName", scope="")` — leave scope empty
2. Check `namespace` field on each step — path must contain both "nimbus" and "code" steps
3. A path entirely within "nimbus" is NOT a vault→code connection (tags ≠ code symbols)

## Error Handling

- Node not found → report the error directly, do not substitute a similar node
- No path found → report as-is, do not fabricate an indirect connection
- Namespace not found → read `prismrag://namespaces` to see available namespaces

## Search Tips

- For exact symbol names (e.g. `ClaudeSDKNode`), exact match beats semantic search
- Use `mode="dfs"` to follow call chains; `mode="bfs"` for broad topic context
- Use `ontology_type="decision"` to filter for architectural decisions
- `budget=8000` for detailed deep dives; `budget=2000` for quick lookups
"""


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
