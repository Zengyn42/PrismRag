"""Generate an interactive HTML knowledge graph visualization using pyvis.

Produces a standalone graph.html file that can be opened in any browser.

Visual encoding:
- Node size: proportional to degree (god nodes are bigger)
- Node color: mapped to community_id (each community gets a distinct hue)
- Node shape: circle for notes, diamond for tags, triangle for categories
- Portal nodes (context_ref, cross-namespace): hexagon + orange (#F5A623)
- Edge width: proportional to weight (confidence_score)
- Edge color: EXTRACTED=solid bright, INFERRED=semi-transparent
- Edge style: dashed for INFERRED edges

Interaction:
- Drag to rearrange
- Scroll to zoom
- Hover to see node info
- Click to open in Obsidian (vault nodes) or navigate to portal target
- Physics simulation (Barnes-Hut) for force-directed layout
- URL hash (#NODE-ID) focuses that node after stabilization
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import quote

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
    # vault
    "note": "dot",
    "tag": "diamond",
    "category": "triangle",
    "knowledge": "dot",
    "image": "square",
    "pdf": "square",
    "audio": "square",
    "section": "dot",
    "block": "dot",
    # portal / cross-namespace
    "context_ref": "hexagon",
    # code
    "function": "dot",
    "class": "square",
    "module": "triangle",
    "flow": "hexagon",
}

# Node colors by kind (code namespace gets distinct palette)
_KIND_COLORS = {
    "function": "#4363d8",   # blue
    "class":    "#911eb4",   # purple
    "module":   "#f58231",   # orange
    "flow":     "#42d4f4",   # cyan
    # portal nodes
    "context_ref": "#F5A623",  # amber / portal orange
}

_PORTAL_COLOR = "#F5A623"   # shared portal color for cross-namespace nodes

# Edge colors by confidence
_EDGE_COLORS = {
    "EXTRACTED": "rgba(200, 200, 200, 0.8)",
    "INFERRED": "rgba(100, 180, 255, 0.4)",
    "AMBIGUOUS": "rgba(255, 200, 100, 0.3)",
}

# ---------------------------------------------------------------------------
# JavaScript injected just before </body>
# Handles:
#   1. Obsidian URI click-to-open for note/knowledge nodes
#   2. Portal navigation (portal_href) for context_ref / cross-namespace nodes
#   3. URL hash (#NODE-ID) focus after stabilization
# ---------------------------------------------------------------------------
_OBSIDIAN_JS_TEMPLATE = """\
<script type="text/javascript">
(function () {{
  /* Mapping: nodeId -> {{obsidian_uri: "..."}} or {{portal_href: "..."}} */
  var _prismNodeData = {node_data_json};

  function _attach() {{
    if (typeof network === 'undefined') {{
      setTimeout(_attach, 50);
      return;
    }}

    /* Click handler: open Obsidian or navigate to portal */
    network.on('click', function (params) {{
      if (!params.nodes.length) return;
      var nodeId = params.nodes[0];
      var nd = _prismNodeData[nodeId];
      if (!nd) return;
      if (nd.portal_href) {{
        window.location.href = nd.portal_href;
      }} else if (nd.obsidian_uri) {{
        window.location.href = nd.obsidian_uri;
      }}
    }});

    /* Hash-based focus: graph.html#NODE-ID */
    var hash = window.location.hash.slice(1);
    if (hash) {{
      var targetId = decodeURIComponent(hash);
      network.once('stabilizationIterationsDone', function () {{
        if (network.body.nodes[targetId]) {{
          network.focus(targetId, {{
            scale: 1.5,
            animation: {{ duration: 800, easingFunction: 'easeInOutQuad' }}
          }});
          network.selectNodes([targetId]);
        }}
      }});
    }}
  }}

  _attach();
}})();
</script>"""


def _community_color(community_id: str | None) -> str:
    """Map a community_id to a color."""
    if not community_id:
        return "#888888"
    try:
        idx = int(community_id.split("_")[-1])
        return _COMMUNITY_COLORS[idx % len(_COMMUNITY_COLORS)]
    except (ValueError, IndexError):
        return "#888888"


def _build_obsidian_uri(vault_name: str, file_path: str) -> str:
    """Construct an obsidian:// URI for a vault file."""
    # Strip leading slash if present
    clean_path = file_path.lstrip("/")
    return f"obsidian://open?vault={quote(vault_name)}&file={quote(clean_path)}"


def _is_portal_node(kind: str, data: dict) -> bool:
    """Return True if this node should be rendered as a cross-namespace portal."""
    if kind == "context_ref":
        return True
    # Check for cross_namespace marker (stored in metadata or as top-level attr)
    meta = data.get("metadata") or {}
    return bool(data.get("cross_namespace") or meta.get("cross_namespace"))


def generate_html(
    graph: KnowledgeGraph,
    output_path: Path,
    height: str = "900px",
    width: str = "100%",
    bg_color: str = "#1a1a2e",
    font_color: str = "#e0e0e0",
    vault_name: str | None = None,
) -> None:
    """Generate an interactive HTML graph visualization.

    Args:
        graph: Knowledge graph to visualize.
        output_path: Where to write graph.html.
        height: Canvas height (CSS).
        width: Canvas width (CSS).
        bg_color: Background color.
        font_color: Label font color.
        vault_name: Obsidian vault name for deep-link URIs.
            When provided, clicking a note/knowledge node opens it in Obsidian.
    """
    net = Network(
        height=height,
        width=width,
        bgcolor=bg_color,
        font_color=font_color,
        directed=True,
        notebook=False,
        select_menu=False,
        filter_menu=False,
        cdn_resources="remote",
    )

    # Physics: Barnes-Hut force-directed layout
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.05,
        damping=0.5,
    )

    # Collect node data for the JS bridge (obsidian_uri / portal_href)
    prism_node_data: dict[str, dict] = {}

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

        # ── Portal node override ─────────────────────────────────────
        is_portal = _is_portal_node(kind, data)
        if is_portal:
            color = _PORTAL_COLOR
            shape = "hexagon"
            label_display = f"⬡ {label}"

            # Build portal_href
            meta = data.get("metadata") or {}
            cross_ns = data.get("cross_namespace") or meta.get("cross_namespace", "")
            if isinstance(cross_ns, str) and "::" in cross_ns:
                # Format: "target_ns::target_id"
                target_ns, _, target_id = cross_ns.partition("::")
                portal_href = f"../{target_ns}/graph.html#{quote(target_id)}"
            elif kind == "context_ref":
                # Point to the source document node in the same graph
                source_file = data.get("source_file", "")
                portal_href = f"#{quote(source_file)}" if source_file else ""
            else:
                portal_href = ""

            if portal_href:
                prism_node_data[node_id] = {"portal_href": portal_href}

        else:
            # ── Regular node ─────────────────────────────────────────
            label_display = label
            if kind in _KIND_COLORS:
                color = _KIND_COLORS[kind]
            else:
                color = _community_color(community_id)
            shape = _KIND_SHAPES.get(kind, "dot")

            # Build Obsidian URI for note / knowledge nodes
            if vault_name and kind in ("note", "knowledge"):
                source_file = data.get("source_file", "")
                knowledge_id = data.get("knowledge_id")
                if knowledge_id and not source_file:
                    # Synthesize path for knowledge nodes that haven't been ingested yet
                    source_file = f"knowledge/{knowledge_id}.md"
                if source_file:
                    prism_node_data[node_id] = {
                        "obsidian_uri": _build_obsidian_uri(vault_name, source_file)
                    }

        # ── Tooltip ──────────────────────────────────────────────────
        source_file = data.get("source_file", "")
        meta = data.get("metadata") or {}
        title = (
            f"<b>{label}</b><br>"
            f"kind: {kind}<br>"
            f"degree: {degree}<br>"
            f"tokens: {tokens}<br>"
        )
        if source_file:
            ls, le = meta.get("line_start"), meta.get("line_end")
            loc = f":{ls}–{le}" if ls else ""
            title += f"<i>{source_file}{loc}</i><br>"
        sig = meta.get("signature")
        if sig:
            title += f"<code>{sig[:80]}</code><br>"
        doc = meta.get("docstring", "")
        if doc:
            title += f"{doc.split(chr(10))[0][:100]}<br>"
        content = data.get("content", "")
        if content and kind in ("function", "class"):
            lines = content.split("\n")[:8]
            snippet = "\n".join(lines)
            title += f"<pre style='font-size:11px;background:#222;color:#eee;padding:4px'>{snippet}</pre>"

        net.add_node(
            node_id,
            label=label_display[:28],
            title=title,
            size=size,
            color=color,
            shape=shape,
            font={"size": 12 if kind in ("note", "knowledge") else 9},
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
        dashes = confidence != "EXTRACTED"

        net.add_edge(
            source,
            target,
            title=title,
            width=edge_width,
            color=edge_color,
            dashes=dashes,
            arrows="to" if relation in (
                "links_to", "links_to_section", "links_to_block", "embeds"
            ) else "",
        )

    # ── Save + post-process ──────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))

    # Inject Obsidian + portal JS just before </body>
    obsidian_js = _OBSIDIAN_JS_TEMPLATE.format(
        node_data_json=json.dumps(prism_node_data, ensure_ascii=False)
    )
    html = output_path.read_text(encoding="utf-8")
    html = html.replace("</body>", obsidian_js + "\n</body>")
    output_path.write_text(html, encoding="utf-8")

    logger.info(
        f"[visualize] saved interactive graph to {output_path} "
        f"({len(prism_node_data)} nodes with Obsidian/portal URIs)"
    )
