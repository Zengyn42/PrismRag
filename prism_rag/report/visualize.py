"""Generate an interactive HTML knowledge graph visualization using pyvis.

Produces a standalone graph.html file that can be opened in any browser.

Visual encoding:
- Node size: proportional to degree (god nodes are bigger)
- Node color: mapped to community_id (each community gets a distinct hue)
- Node shape: circle for notes, diamond for tags, triangle for categories
- Edge width: proportional to weight (confidence_score)
- Edge color: EXTRACTED=solid bright, INFERRED=semi-transparent
- Edge style: dashed for INFERRED edges (optional, depends on pyvis support)

Interaction:
- Drag to rearrange
- Scroll to zoom
- Hover to see node info
- Click to highlight connections
- Search bar to locate nodes by name
- Physics simulation (Barnes-Hut) for force-directed layout
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyvis.network import Network

from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Community colors — 12 distinct hues, enough for most vaults
_COMMUNITY_COLORS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#fabed4",  # pink
    "#469990",  # teal
    "#dcbeff",  # lavender
    "#9A6324",  # brown
]

# Node shapes by kind
_KIND_SHAPES = {
    "note": "dot",
    "tag": "diamond",
    "category": "triangle",
    "image": "square",
    "pdf": "square",
    "audio": "square",
    "section": "dot",
    "block": "dot",
}

# Edge colors by confidence
_EDGE_COLORS = {
    "EXTRACTED": "rgba(200, 200, 200, 0.8)",
    "INFERRED": "rgba(100, 180, 255, 0.4)",
    "AMBIGUOUS": "rgba(255, 200, 100, 0.3)",
}


def _community_color(community_id: str | None) -> str:
    """Map a community_id to a color."""
    if not community_id:
        return "#888888"
    try:
        idx = int(community_id.split("_")[-1])
        return _COMMUNITY_COLORS[idx % len(_COMMUNITY_COLORS)]
    except (ValueError, IndexError):
        return "#888888"


def generate_html(
    graph: KnowledgeGraph,
    output_path: Path,
    height: str = "900px",
    width: str = "100%",
    bg_color: str = "#1a1a2e",
    font_color: str = "#e0e0e0",
) -> None:
    """Generate an interactive HTML graph visualization.

    Args:
        graph: Knowledge graph to visualize.
        output_path: Where to write graph.html.
        height: Canvas height (CSS).
        width: Canvas width (CSS).
        bg_color: Background color.
        font_color: Label font color.
    """
    net = Network(
        height=height,
        width=width,
        bgcolor=bg_color,
        font_color=font_color,
        directed=True,
        notebook=False,
        select_menu=True,
        filter_menu=True,
    )

    # Physics: Barnes-Hut force-directed layout
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.05,
        damping=0.5,
    )

    # ── Add nodes ────────────────────────────────────────────────────
    for node_id, data in graph.g.nodes(data=True):
        kind = data.get("kind", "note")
        label = data.get("label", node_id)
        community_id = data.get("community_id")
        tokens = data.get("tokens", 0)
        degree = graph.degree(node_id)

        # Size: base 10, scale up with degree
        size = 10 + degree * 3
        # God nodes (degree > 10) get extra emphasis
        if degree > 10:
            size = 20 + degree * 4

        color = _community_color(community_id)
        shape = _KIND_SHAPES.get(kind, "dot")

        # Tooltip: show on hover
        source_file = data.get("source_file", "")
        title = (
            f"<b>{label}</b><br>"
            f"kind: {kind}<br>"
            f"degree: {degree}<br>"
            f"tokens: {tokens}<br>"
            f"community: {community_id or '—'}<br>"
        )
        if source_file:
            title += f"file: {source_file}<br>"

        # Skip content from tooltip (too large)
        net.add_node(
            node_id,
            label=label if kind == "note" else label[:20],
            title=title,
            size=size,
            color=color,
            shape=shape,
            font={"size": 12 if kind == "note" else 9},
        )

    # ── Add edges ────────────────────────────────────────────────────
    for source, target, data in graph.g.edges(data=True):
        relation = data.get("relation", "?")
        confidence = data.get("confidence", "EXTRACTED")
        score = float(data.get("confidence_score", 1.0))
        weight = float(data.get("weight", 1.0))

        edge_color = _EDGE_COLORS.get(confidence, "rgba(150,150,150,0.5)")
        edge_width = 0.5 + weight * 2

        title = f"{relation}<br>confidence: {confidence}<br>score: {score:.2f}"

        # Dashes for INFERRED edges
        dashes = confidence != "EXTRACTED"

        net.add_edge(
            source,
            target,
            title=title,
            width=edge_width,
            color=edge_color,
            dashes=dashes,
            arrows="to" if relation in ("links_to", "links_to_section", "links_to_block", "embeds") else "",
        )

    # ── Save ─────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))
    logger.info(f"[visualize] saved interactive graph to {output_path}")
