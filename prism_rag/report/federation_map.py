"""Generate a federation meta-graph HTML visualization.

Produces a single graph.html showing all namespaces as box nodes with
cross-namespace edge counts. Clicking a namespace node opens that
namespace's own graph.html in a new tab.

Usage:
    from prism_rag.report.federation_map import generate_federation_html
    generate_federation_html(federated_graph, output_dir / "federation.html")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pyvis.network import Network

if TYPE_CHECKING:
    from prism_rag.store.federated import FederatedGraph

logger = logging.getLogger(__name__)

# Namespace box node color
_NS_BOX_COLOR = "#4363d8"    # blue
_CROSS_EDGE_COLOR = "rgba(245, 166, 35, 0.7)"  # amber — matches portal orange

# ---------------------------------------------------------------------------
# JavaScript injected just before </body>
# Clicking a namespace box opens that namespace's graph.html in a new tab.
# ---------------------------------------------------------------------------
_FEDERATION_JS_TEMPLATE = """\
<script type="text/javascript">
(function () {{
  /* Mapping: nodeId (namespace name) -> portal href */
  var _federationPortals = {portal_map_json};

  function _attach() {{
    if (typeof network === 'undefined') {{
      setTimeout(_attach, 50);
      return;
    }}
    network.on('click', function (params) {{
      if (!params.nodes.length) return;
      var nodeId = params.nodes[0];
      var href = _federationPortals[nodeId];
      if (href) window.open(href, '_blank');
    }});
  }}

  _attach();
}})();
</script>"""


def _count_cross_edges(
    federated: "FederatedGraph",
) -> dict[tuple[str, str], int]:
    """Count cross-namespace edge pairs.

    For every edge in every namespace graph, if the target node belongs to a
    different namespace (detected via 'namespace' attribute or '::' prefix),
    record a cross-namespace connection.

    Returns:
        dict mapping (source_ns, target_ns) -> count of cross-namespace edges.
    """
    counts: dict[tuple[str, str], int] = {}
    ns_set = set(federated.namespaces)

    for ns in federated.namespaces:
        graph = federated.get_graph(ns)
        if graph is None:
            continue
        for _src, target, data in graph.g.edges(data=True):
            # Detect target namespace from node attribute or '::' prefix
            target_data = graph.g.nodes.get(target, {})
            target_ns = target_data.get("namespace", "")
            if not target_ns and "::" in target:
                target_ns = target.split("::", 1)[0]
            if target_ns and target_ns != ns and target_ns in ns_set:
                key = (ns, target_ns)
                counts[key] = counts.get(key, 0) + 1
    return counts


def generate_federation_html(
    federated: "FederatedGraph",
    output: Path,
    height: str = "600px",
    width: str = "100%",
    bg_color: str = "#1a1a2e",
    font_color: str = "#e0e0e0",
) -> None:
    """Generate a federation meta-graph visualization.

    Each namespace becomes a box node. Edges represent cross-namespace
    connections (labeled with count). Clicking a node opens that namespace's
    graph.html in a new browser tab.

    Args:
        federated: Loaded FederatedGraph instance.
        output: Where to write the federation HTML file.
        height: Canvas height CSS string.
        width: Canvas width CSS string.
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
        select_menu=False,
        filter_menu=False,
        cdn_resources="remote",
    )

    # Light physics for small meta-graph
    net.repulsion(
        node_distance=200,
        central_gravity=0.1,
        spring_length=200,
        spring_strength=0.05,
        damping=0.5,
    )

    portal_map: dict[str, str] = {}

    # ── Namespace nodes ───────────────────────────────────────────────
    for ns in federated.namespaces:
        graph = federated.get_graph(ns)
        n_count = graph.node_count if graph is not None else 0
        e_count = graph.edge_count if graph is not None else 0
        label = f"{ns}\n{n_count}N / {e_count}E"
        portal_href = f"{ns}/graph.html"
        portal_map[ns] = portal_href

        net.add_node(
            ns,
            label=label,
            shape="box",
            color=_NS_BOX_COLOR,
            size=30,
            title=(
                f"<b>{ns}</b><br>"
                f"nodes: {n_count}<br>"
                f"edges: {e_count}<br>"
                f"<i>click to open {portal_href}</i>"
            ),
            font={"size": 14},
        )

    # ── Cross-namespace edges ─────────────────────────────────────────
    cross_edges = _count_cross_edges(federated)
    seen: set[tuple[str, str]] = set()
    for (src_ns, tgt_ns), count in cross_edges.items():
        # Deduplicate bidirectional pairs to avoid double edges in display
        pair = (min(src_ns, tgt_ns), max(src_ns, tgt_ns))
        if pair in seen:
            continue
        seen.add(pair)
        net.add_edge(
            src_ns,
            tgt_ns,
            title=f"{count} cross-namespace reference(s)",
            label=str(count),
            width=1 + min(count / 5, 4),   # max width 5
            color=_CROSS_EDGE_COLOR,
            dashes=False,
        )

    # ── Save + post-process ───────────────────────────────────────────
    output.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output))

    fed_js = _FEDERATION_JS_TEMPLATE.format(
        portal_map_json=json.dumps(portal_map, ensure_ascii=False)
    )
    html = output.read_text(encoding="utf-8")
    html = html.replace("</body>", fed_js + "\n</body>")
    output.write_text(html, encoding="utf-8")

    logger.info(
        f"[federation_map] saved federation graph to {output} "
        f"({len(federated.namespaces)} namespace(s), {len(cross_edges)} cross-ns edge pair(s))"
    )
