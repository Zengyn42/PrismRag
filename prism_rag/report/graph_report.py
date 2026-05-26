"""Pass 5: Generate GRAPH_REPORT.md from a knowledge graph.

The report is designed to be the entry point for AI agents querying the graph.
It highlights:
1. Summary stats (node/edge/community counts)
2. Top god nodes across the whole graph
3. Each community's god nodes + size
4. "Surprising connections" — high-confidence edges that cross community boundaries

Agents (via MCP) can read this report first to get a quick mental model of the
knowledge base before issuing specific queries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prism_rag.store.graph import KnowledgeGraph


def _top_nodes_by_degree(graph: KnowledgeGraph, limit: int = 10) -> list[tuple[str, int]]:
    """Return top-N nodes by total degree."""
    degrees = [(nid, graph.degree(nid)) for nid in graph.g.nodes()]
    degrees.sort(key=lambda pair: pair[1], reverse=True)
    return degrees[:limit]


def _cross_community_edges(
    graph: KnowledgeGraph,
    min_confidence: float = 0.7,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find high-confidence edges whose endpoints are in different communities."""
    result: list[dict[str, Any]] = []
    for u, v, data in graph.g.edges(data=True):
        cu = graph.g.nodes[u].get("community_id")
        cv = graph.g.nodes[v].get("community_id")
        if not (cu and cv) or cu == cv:
            continue
        score = float(data.get("confidence_score", 0.0))
        if score < min_confidence:
            continue
        result.append(
            {
                "source": u,
                "source_label": graph.g.nodes[u].get("label", u),
                "source_community": cu,
                "target": v,
                "target_label": graph.g.nodes[v].get("label", v),
                "target_community": cv,
                "relation": data.get("relation", "?"),
                "score": score,
            }
        )
    # Sort by score descending
    result.sort(key=lambda r: r["score"], reverse=True)
    return result[:limit]


def _node_label(graph: KnowledgeGraph, node_id: str) -> str:
    return graph.g.nodes.get(node_id, {}).get("label", node_id)


def generate_report(
    graph: KnowledgeGraph,
    output_path: Path,
    vault_root: Path | None = None,
) -> str:
    """Generate GRAPH_REPORT.md content, write to `output_path`, and return the content."""
    lines: list[str] = []

    # ── Header ───────────────────────────────────────────────────────
    lines.append("# NimbusVault Knowledge Graph Report")
    lines.append("")
    lines.append(f"> Generated at: `{datetime.now(timezone.utc).isoformat()}`")
    if vault_root:
        lines.append(f"> Source vault: `{vault_root}`")
    lines.append(f"> Total nodes: **{graph.node_count}**")
    lines.append(f"> Total edges: **{graph.edge_count}**")
    lines.append(f"> Communities: **{len(graph.communities)}**")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Top God Nodes ────────────────────────────────────────────────
    lines.append("## Global Hub Nodes (top 10 by degree)")
    lines.append("")
    top_gods = _top_nodes_by_degree(graph, limit=10)
    if top_gods:
        for nid, degree in top_gods:
            node_data = graph.g.nodes[nid]
            label = node_data.get("label", nid)
            kind = node_data.get("kind", "?")
            community_id = node_data.get("community_id") or "—"
            lines.append(f"- **{label}** · `{kind}` · degree {degree} · community `{community_id}`")
    else:
        lines.append("_(graph is empty)_")
    lines.append("")

    # ── Communities ──────────────────────────────────────────────────
    lines.append("## Community Overview")
    lines.append("")
    if graph.communities:
        sorted_comms = sorted(
            graph.communities.values(),
            key=lambda c: c.member_count,
            reverse=True,
        )
        for comm in sorted_comms:
            lines.append(f"### `{comm.id}` — {comm.label}")
            lines.append("")
            lines.append(f"- Members: **{comm.member_count}**")
            lines.append(f"- Internal density: **{comm.internal_density}**")
            if comm.god_nodes:
                god_labels = [_node_label(graph, n) for n in comm.god_nodes]
                lines.append(f"- God nodes: {', '.join(f'`{g}`' for g in god_labels)}")
            lines.append("")
    else:
        lines.append("_(community detection has not been run)_")
        lines.append("")

    # ── Surprising connections ──────────────────────────────────────
    lines.append("## Surprising Connections (high-confidence cross-community edges)")
    lines.append("")
    cross_edges = _cross_community_edges(graph)
    if cross_edges:
        for edge in cross_edges:
            lines.append(
                f"- **{edge['source_label']}** "
                f"`[{edge['source_community']}]` "
                f"──{edge['relation']}──► "
                f"**{edge['target_label']}** "
                f"`[{edge['target_community']}]` "
                f"· score {edge['score']:.2f}"
            )
    else:
        lines.append("_(no high-confidence cross-community edges found)_")
    lines.append("")

    # ── Footer ───────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("*— Generated by PrismRag v4.0 · Zengyn42*")
    lines.append("")

    content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return content
