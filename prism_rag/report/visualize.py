"""Generate an interactive HTML knowledge graph visualization using force-graph.

Produces a standalone graph.html file that can be opened in any browser.

Visual encoding:
- Node size: proportional to structural degree (larger = more connected)
- Node color: mapped to community_id (distinct hue per cluster)
- Node glow: soft halo in node color (WebGL canvas)
- Edge particles: animated flow on wiki-link / cross-namespace edges
- Edge color: EXTRACTED=bright, INFERRED=semi-transparent
- On-demand edges (e.g. mentions_symbol): shown only on node click (toggle)

Interaction:
- Scroll/pinch: zoom (labels appear progressively as you zoom in — LOD)
- Drag background: pan
- Drag node: reposition
- Click node: toggle mentions_symbol edges + open in Obsidian / portal
- Click background: clear selection
- Search box: highlight matching nodes
- URL hash (#NODE-ID): auto-focuses that node after layout stabilizes

Renderer: vasturiano/force-graph (D3-force + HTML5 Canvas, CDN)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from urllib.parse import quote

from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# ── Community palette ──────────────────────────────────────────────────────────
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

# ── Kind → color overrides ────────────────────────────────────────────────────
_KIND_COLORS: dict[str, str] = {
    "function":    "#4363d8",  # blue        — standalone function
    "method":      "#6a8fd8",  # light blue  — class method
    "test":        "#2e8b57",  # green       — test function / test method
    "class":       "#911eb4",  # purple      — class
    "test_class":  "#7dbf7d",  # light green — Test class
    "module":      "#f58231",  # orange      — module
    "test_module": "#b5d89a",  # pale green  — test_*.py module
    "flow":        "#42d4f4",
    "context_ref": "#F5A623",
}

_PORTAL_COLOR = "#F5A623"

# ── Edge base colors by confidence ────────────────────────────────────────────
_EDGE_COLORS: dict[str, str] = {
    "EXTRACTED": "rgba(200,200,200,0.6)",
    "INFERRED":  "rgba(100,180,255,0.3)",
    "AMBIGUOUS": "rgba(255,200,100,0.25)",
}

# ── Relations that get animated particles ─────────────────────────────────────
_PARTICLE_RELATIONS = frozenset({
    "links_to",
    "links_to_section",
    "links_to_block",
    "cross_namespace",
})

# ── Standalone HTML template ──────────────────────────────────────────────────
# Note: all user-supplied content is set via textContent (DOM API) in JS,
# never via innerHTML, to prevent XSS.
_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: {bg_color}; overflow: hidden; font-family: monospace; color: {font_color}; }}
    #graph {{ width: 100vw; height: 100vh; display: block; }}
    #hud {{
      position: fixed; top: 12px; left: 12px; z-index: 20;
      display: flex; flex-direction: column; gap: 8px;
      pointer-events: none;
    }}
    #search {{
      pointer-events: all;
      background: rgba(20,20,36,0.88);
      border: 1px solid rgba(255,255,255,0.18);
      color: {font_color};
      padding: 6px 11px; border-radius: 7px;
      font-size: 13px; width: 230px; outline: none;
      transition: border-color .15s;
    }}
    #search:focus {{ border-color: rgba(255,255,255,0.45); }}
    #legend {{
      pointer-events: all;
      background: rgba(20,20,36,0.82);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 8px; font-size: 11px;
      color: rgba(255,255,255,0.55);
      min-width: 180px; overflow: hidden;
    }}
    #legend-header {{
      padding: 7px 12px; cursor: pointer;
      color: rgba(255,255,255,0.75); font-size: 11px;
      display: flex; justify-content: space-between; align-items: center;
      user-select: none;
    }}
    #legend-header:hover {{ color: #fff; }}
    #legend-body {{
      padding: 0 12px 10px; display: block;
      max-height: calc(100vh - 120px); overflow-y: auto;
      scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.15) transparent;
    }}
    #legend-body::-webkit-scrollbar {{ width: 4px; }}
    #legend-body::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.15); border-radius: 2px; }}
    #legend-body.collapsed {{ display: none; }}
    #btn-reset {{
      pointer-events: all;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.2);
      color: rgba(255,255,255,0.6); font-size: 11px;
      padding: 4px 10px; border-radius: 5px; cursor: pointer;
      transition: background .12s, color .12s;
      font-family: monospace; width: 100%;
    }}
    #btn-reset:hover {{ background: rgba(255,80,80,0.25); color: #fff; border-color: rgba(255,100,100,0.4); }}
    .leg-section {{
      font-size: 10px; color: rgba(255,255,255,0.3);
      letter-spacing: .08em; text-transform: uppercase;
      margin: 8px 0 4px;
    }}
    .leg-row {{
      display: flex; align-items: center; gap: 7px;
      line-height: 1.9; color: rgba(255,255,255,0.65);
    }}
    .leg-filter {{
      cursor: pointer; border-radius: 4px;
      padding: 0 4px; margin: 0 -4px;
      transition: background .12s, color .12s;
    }}
    .leg-filter:hover {{ background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.9); }}
    .leg-filter.active {{
      background: rgba(255,255,255,0.13);
      color: #fff;
      outline: 1px solid rgba(255,255,255,0.25);
    }}
    .swatch {{
      width: 10px; height: 10px; border-radius: 50%;
      flex-shrink: 0; display: inline-block;
    }}
    .swatch-line {{
      width: 20px; height: 2px; flex-shrink: 0; border-radius: 1px;
    }}
    .swatch-inferred {{
      background: repeating-linear-gradient(
        90deg, rgba(100,180,255,0.5) 0 4px, transparent 4px 7px);
      height: 2px;
    }}
    kbd {{
      background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
      border-radius: 3px; padding: 0 4px; font-size: 10px;
      font-family: monospace; color: rgba(255,255,255,0.8);
    }}
    #stats {{
      position: fixed; top: 12px; right: 12px; z-index: 20;
      font-size: 11px; color: rgba(255,255,255,0.28); text-align: right;
      line-height: 1.6;
    }}
    #info {{
      position: fixed; bottom: 14px; right: 14px; z-index: 20;
      background: rgba(20,20,36,0.92);
      border: 1px solid rgba(255,255,255,0.18);
      padding: 11px 15px; border-radius: 9px;
      font-size: 12px; max-width: 340px;
      display: none; line-height: 1.65;
    }}
    #info-title {{ font-size: 14px; color: #fff; margin-bottom: 5px; word-break: break-all; font-weight: bold; }}
    #info-meta {{ color: rgba(255,255,255,0.5); font-size: 11px; }}
    #info-file {{ color: rgba(120,200,255,0.7); margin-top: 3px; font-size: 11px; word-break: break-all; }}
    #info-sig {{ color: rgba(255,255,255,0.45); margin-top: 3px; font-size: 11px; font-style: italic; }}
    #info-hint {{ color: rgba(255,200,80,0.6); margin-top: 5px; font-size: 10px; }}
  </style>
</head>
<body>
  <div id="graph"></div>

  <div id="hud">
    <input id="search" type="text" placeholder="Search nodes..." autocomplete="off" spellcheck="false"/>
    <div id="legend">
      <div id="legend-header">
        <span>Legend</span><span id="legend-arrow">▾</span>
      </div>
      <div id="legend-body">

        <button id="btn-reset">⟳ Reset all</button>

        <div class="leg-section">Controls</div>
        <div class="leg-row"><kbd>click ×1</kbd>&nbsp;focus node</div>
        <div class="leg-row"><kbd>click ×2</kbd>&nbsp;+ select legend type</div>
        <div class="leg-row"><kbd>click ×3</kbd>&nbsp;clear node, keep legend</div>
        <div class="leg-row"><kbd>click bg</kbd>&nbsp;clear node focus</div>
        <div class="leg-row"><kbd>right-click</kbd>&nbsp;open Obsidian</div>
        <div class="leg-row"><kbd>Esc</kbd>&nbsp;reset everything</div>
        <div class="leg-row"><kbd>WASD</kbd>&nbsp;pan &nbsp;<kbd>+</kbd><kbd>-</kbd>&nbsp;zoom</div>

        <div class="leg-section">Node — Code</div>
        <div class="leg-row leg-filter" data-color="#4363d8"><span class="swatch" style="background:#4363d8"></span>function</div>
        <div class="leg-row leg-filter" data-color="#6a8fd8"><span class="swatch" style="background:#6a8fd8"></span>method</div>
        <div class="leg-row leg-filter" data-color="#911eb4"><span class="swatch" style="background:#911eb4"></span>class</div>
        <div class="leg-row leg-filter" data-color="#f58231"><span class="swatch" style="background:#f58231"></span>module</div>
        <div class="leg-row leg-filter" data-color="#42d4f4"><span class="swatch" style="background:#42d4f4"></span>flow</div>
        <div class="leg-row leg-filter" data-color="#5a7fa8"><span class="swatch" style="background:#5a7fa8"></span>external / stdlib</div>
        <div class="leg-section">Node — Test</div>
        <div class="leg-row leg-filter" data-color="#2e8b57"><span class="swatch" style="background:#2e8b57"></span>test function / method</div>
        <div class="leg-row leg-filter" data-color="#7dbf7d"><span class="swatch" style="background:#7dbf7d"></span>test class</div>
        <div class="leg-row leg-filter" data-color="#b5d89a"><span class="swatch" style="background:#b5d89a"></span>test module</div>

        <div class="leg-section">Node — Docs</div>
        {community_legend_html}
        <div class="leg-row" style="color:rgba(255,255,255,0.35);font-size:10px;margin-top:2px;line-height:1.6">
          Leiden: auto-detected topic clusters.<br>Same color = semantically related docs.
        </div>

        <div class="leg-section">Node — Portal</div>
        <div class="leg-row leg-filter" data-color="#F5A623"><span class="swatch" style="background:#F5A623"></span>cross-namespace ref</div>
        <div class="leg-row" style="color:rgba(255,255,255,0.35);font-size:10px;line-height:1.6">
          Links to a node in another graph<br>(only in federated multi-project views)
        </div>

        <div class="leg-section">Visual Encoding</div>
        <div class="leg-row" style="color:rgba(255,255,255,0.55);font-size:10px;line-height:1.7">
          <span style="color:rgba(255,255,255,0.8)">size</span> = connections (degree)<br>
          <span style="color:rgba(255,255,255,0.8)">tokens</span> = content length for RAG
        </div>

        <div class="leg-section">Edges</div>
        <div class="leg-row"><span class="swatch-line" style="background:rgba(200,200,200,0.6)"></span>structural</div>
        <div class="leg-row"><span class="swatch-inferred swatch-line"></span>inferred</div>
        <div class="leg-row"><span class="swatch-line" style="background:rgba(100,180,255,0.35)"></span>semantic similarity</div>
        <div class="leg-row" style="color:rgba(255,255,255,0.45);font-size:10px">&#9654; particles = wiki-link</div>
        <div class="leg-row" style="color:rgba(255,200,80,0.6);font-size:10px">click node = mentions_symbol</div>


      </div>
    </div>
  </div>

  <div id="stats">
    <span id="stat-nodes">{node_count}</span> nodes<br>
    <span id="stat-links">{link_count}</span> links<br>
    <span id="stat-od">{od_count}</span> on-demand
  </div>

  <div id="info">
    <div id="info-title"></div>
    <div id="info-meta"></div>
    <div id="info-file"></div>
    <div id="info-sig"></div>
    <div id="info-hint"></div>
  </div>

  <script src="https://unpkg.com/force-graph@1/dist/force-graph.min.js"></script>
  <script>
  (function () {{
    var GD = {graph_data_json};
    var _activeNode = null;    /* clicked node id (ego-graph focus) */
    var _focusSet = null;      /* computed visible set: null = show all */
    var _legendColors = {{}};  /* multi-select: {{ color: true }} */
    var _legendEls = {{}};     /* {{ color: domElement }} for .active management */
    var _nodeClickCount = 0;   /* consecutive clicks on _activeNode: 1=focus 2=+legend 3=clear node */
    var _origColors = {{}};
    GD.nodes.forEach(function (n) {{ _origColors[n.id] = n.color; }});

    /* -- Adjacency map (raw string ids, built before force-graph resolves them) */
    var _adj = {{}};
    GD.links.forEach(function (l) {{
      if (l._onDemand) return;
      var s = l.source, t = l.target;
      if (!_adj[s]) _adj[s] = {{}};
      if (!_adj[t]) _adj[t] = {{}};
      _adj[s][t] = true;
      _adj[t][s] = true;
    }});

    /* -- Compute focus set: union of legend colors + node ego-graph --------- */
    function _computeFocusSet() {{
      var hasLegend = Object.keys(_legendColors).length > 0;
      var hasNode   = !!_activeNode;
      if (!hasLegend && !hasNode) return null;  /* show everything */

      var set = {{}};
      if (hasLegend) {{
        GD.nodes.forEach(function (n) {{
          if (_legendColors[_origColors[n.id]]) set[n.id] = true;
        }});
      }}
      if (hasNode) {{
        set[_activeNode] = true;
        Object.keys(_adj[_activeNode] || {{}}).forEach(function (nid) {{
          set[nid] = true;
        }});
      }}
      return set;
    }}

    /* -- Apply computed focus to the graph ---------------------------------- */
    function _refresh() {{
      _focusSet = _computeFocusSet();
      GD.nodes.forEach(function (n) {{ n._dimmed = _focusSet ? !_focusSet[n.id] : false; }});
      Graph.nodeColor(function (n) {{ return _origColors[n.id] || '#888888'; }});
      Graph.linkVisibility(_linkVisible);
    }}

    /* -- Clear all legend selections ---------------------------------------- */
    function _clearAllLegend() {{
      Object.keys(_legendEls).forEach(function (c) {{
        _legendEls[c].classList.remove('active');
      }});
      _legendColors = {{}};
      _legendEls = {{}};
    }}

    /* -- Toggle a legend color on/off --------------------------------------- */
    function _toggleLegend(color, el) {{
      if (_legendColors[color]) {{
        delete _legendColors[color];
        el.classList.remove('active');
        delete _legendEls[color];
      }} else {{
        _legendColors[color] = true;
        el.classList.add('active');
        _legendEls[color] = el;
      }}
      _refresh();
    }}

    /* Legend click delegation */
    document.getElementById('legend-body').addEventListener('click', function (e) {{
      var row = e.target.closest('.leg-filter');
      if (!row) return;
      var color = row.getAttribute('data-color');
      if (color) _toggleLegend(color, row);
    }});

    /* -- Link visibility callback ----------------------------------------- */
    function _linkVisible(link) {{
      var s = typeof link.source === 'object' ? link.source.id : link.source;
      var t = typeof link.target === 'object' ? link.target.id : link.target;

      /* Focus mode: hide edges where either endpoint is dimmed */
      if (_focusSet && (!_focusSet[s] || !_focusSet[t])) return false;

      /* On-demand edges: only when their node is the active node */
      if (link._onDemand) {{
        if (!_activeNode) return false;
        return s === _activeNode || t === _activeNode;
      }}
      return true;
    }}

    /* -- Info panel (DOM-safe: textContent only) ---------------------------- */
    function _setText(id, val) {{
      var el = document.getElementById(id);
      if (el) el.textContent = val || '';
    }}
    function _setVisible(id, show) {{
      var el = document.getElementById(id);
      if (el) el.style.display = show ? '' : 'none';
    }}

    function _showInfo(node) {{
      var hasMentions = GD.links.some(function (l) {{
        if (!l._onDemand) return false;
        var s = typeof l.source === 'object' ? l.source.id : l.source;
        var t = typeof l.target === 'object' ? l.target.id : l.target;
        return s === node.id || t === node.id;
      }});
      _setText('info-title', node.label || node.id);
      _setText('info-meta',
        'kind: ' + (node.kind || '?') +
        '  degree: ' + (node.degree || 0) +
        (node.community ? '  cluster: ' + node.community : '')
      );
      _setText('info-file', node.source_file || '');
      _setText('info-sig', node.sig || '');
      var hint = '';
      if (hasMentions) {{
        hint = (_activeNode === node.id)
          ? 'click again to hide mentions'
          : 'click to show symbol mentions';
      }} else if (node.obsidian_uri) {{
        hint = 'opens in Obsidian';
      }} else if (node.portal_href) {{
        hint = 'portal link';
      }}
      _setText('info-hint', hint);
      document.getElementById('info').style.display = 'block';
    }}

    /* -- Build graph -------------------------------------------------------- */
    var Graph = ForceGraph()(document.getElementById('graph'))
      .backgroundColor('{bg_color}')
      .width(window.innerWidth)
      .height(window.innerHeight)
      .graphData(GD)
      .nodeId('id')
      .nodeVal('val')
      .linkColor('color')
      .linkWidth('width')
      .linkVisibility(_linkVisible)
      .linkDirectionalParticles('particles')
      .linkDirectionalParticleSpeed(0.005)
      .linkDirectionalParticleWidth(2.5)
      .linkDirectionalParticleColor('color')
      .nodeCanvasObjectMode(function () {{ return 'replace'; }})
      .nodeCanvasObject(function (node, ctx, globalScale) {{
        var r = Math.sqrt(Math.max(1, node.val)) * 4;
        var baseCol = _origColors[node.id] || '#888888';
        var col = (node._dimmed) ? baseCol + '18' : baseCol;

        /* Glow halo */
        ctx.beginPath();
        ctx.arc(node.x, node.y, r * 2.2, 0, 2 * Math.PI);
        ctx.fillStyle = baseCol + (node._dimmed ? '08' : '28');
        ctx.fill();

        /* Node circle */
        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = col;
        ctx.fill();

        /* Active ring */
        if (_activeNode === node.id) {{
          ctx.beginPath();
          ctx.arc(node.x, node.y, r + 3, 0, 2 * Math.PI);
          ctx.strokeStyle = 'rgba(255,255,255,0.55)';
          ctx.lineWidth = 1.5 / globalScale;
          ctx.stroke();
        }}

        /* LOD label — only for focused/visible nodes, hidden when dimmed */
        if (!node._dimmed && globalScale > 0.45) {{
          var fontSize = Math.min(14, Math.max(8, 11 / globalScale));
          ctx.font = fontSize + 'px monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          ctx.fillStyle = '{font_color}';
          ctx.globalAlpha = Math.min(1, (globalScale - 0.45) / 0.3);
          ctx.fillText((node.label || '').substring(0, 24), node.x, node.y + r + 2);
          ctx.globalAlpha = 1;
        }}
      }})
      .nodePointerAreaPaint(function (node, color, ctx) {{
        var r = Math.sqrt(Math.max(1, node.val)) * 4;
        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
      }})
      .nodeLabel(function (node) {{ return node.tooltip || node.label || node.id; }})
      .onNodeClick(function (node) {{
        if (_activeNode !== node.id) {{
          /* Different node: click 1 — ego focus */
          _activeNode = node.id;
          _nodeClickCount = 1;
          _refresh();
          _showInfo(node);
        }} else {{
          _nodeClickCount++;
          if (_nodeClickCount === 2) {{
            /* Click 2: also select this node's legend type */
            var color = _origColors[node.id];
            if (color) {{
              var el = document.querySelector('.leg-filter[data-color="' + color + '"]');
              if (el && !_legendColors[color]) _toggleLegend(color, el);
            }}
            _showInfo(node);
          }} else {{
            /* Click 3+: deselect node, keep legend */
            _activeNode = null;
            _nodeClickCount = 0;
            _refresh();
            document.getElementById('info').style.display = 'none';
          }}
        }}
      }})
      .onNodeRightClick(function (node) {{
        if (node.portal_href) window.location.href = node.portal_href;
        else if (node.obsidian_uri) window.open(node.obsidian_uri, '_blank');
      }})
      .onBackgroundClick(function () {{
        /* Background click: clear node focus only, keep legend selections */
        _activeNode = null;
        _nodeClickCount = 0;
        _refresh();
        document.getElementById('info').style.display = 'none';
      }});

    /* -- Responsive --------------------------------------------------------- */
    window.addEventListener('resize', function () {{
      Graph.width(window.innerWidth).height(window.innerHeight);
    }});

    /* -- Search: highlights matching nodes, clears all focus/legend -------- */
    document.getElementById('search').addEventListener('input', function (e) {{
      var q = e.target.value.toLowerCase().trim();
      _activeNode = null;
      _clearAllLegend();
      document.getElementById('info').style.display = 'none';
      if (!q) {{
        _focusSet = null;
        GD.nodes.forEach(function (n) {{ n._dimmed = false; }});
      }} else {{
        _focusSet = {{}};
        GD.nodes.forEach(function (n) {{
          var match = (n.label || '').toLowerCase().indexOf(q) !== -1 ||
                      (n.id || '').toLowerCase().indexOf(q) !== -1;
          if (match) _focusSet[n.id] = true;
          n._dimmed = !match;
        }});
      }}
      Graph.nodeColor(function (n) {{ return _origColors[n.id] || '#888888'; }});
      Graph.linkVisibility(_linkVisible);
    }});

    /* -- Reset button ------------------------------------------------------- */
    document.getElementById('btn-reset').addEventListener('click', function (e) {{
      e.stopPropagation();  /* don't trigger legend collapse */
      _activeNode = null;
      _clearAllLegend();
      _refresh();
      document.getElementById('info').style.display = 'none';
      document.getElementById('search').value = '';
    }});

    /* -- Legend toggle ------------------------------------------------------ */
    document.getElementById('legend-header').addEventListener('click', function () {{
      var body = document.getElementById('legend-body');
      var arrow = document.getElementById('legend-arrow');
      var collapsed = body.classList.toggle('collapsed');
      arrow.textContent = collapsed ? '▸' : '▾';
    }});

    /* -- Keyboard controls -------------------------------------------------- */
    /* WASD / arrow keys: pan   |   +/=/-: zoom   |   Escape: reset focus     */
    var _keys = {{}};
    var _rafId = null;

    var _MOVE_KEYS = ['w','W','a','A','s','S','d','D',
                      'ArrowUp','ArrowDown','ArrowLeft','ArrowRight'];

    function _keyLoop() {{
      var speed = 6 / (Graph.zoom() || 1);  /* faster when zoomed out */
      var dx = 0, dy = 0;
      if (_keys['w'] || _keys['W'] || _keys['ArrowUp'])    dy -= speed;
      if (_keys['s'] || _keys['S'] || _keys['ArrowDown'])  dy += speed;
      if (_keys['a'] || _keys['A'] || _keys['ArrowLeft'])  dx -= speed;
      if (_keys['d'] || _keys['D'] || _keys['ArrowRight']) dx += speed;
      if (dx !== 0 || dy !== 0) {{
        var c = Graph.centerAt();
        Graph.centerAt(c.x + dx, c.y + dy);
      }}
      var anyHeld = _MOVE_KEYS.some(function (k) {{ return _keys[k]; }});
      _rafId = anyHeld ? requestAnimationFrame(_keyLoop) : null;
    }}

    window.addEventListener('keydown', function (e) {{
      /* Ignore keyboard shortcuts when typing in search box */
      if (document.activeElement === document.getElementById('search')) return;

      /* Zoom */
      if (e.key === '+' || e.key === '=' || e.key === 'NumpadAdd') {{
        e.preventDefault();
        Graph.zoom(Graph.zoom() * 1.25, 150);
        return;
      }}
      if (e.key === '-' || e.key === '_' || e.key === 'NumpadSubtract') {{
        e.preventDefault();
        Graph.zoom(Graph.zoom() / 1.25, 150);
        return;
      }}

      /* Escape: clear everything — node focus + all legend selections */
      if (e.key === 'Escape') {{
        _activeNode = null;
        _nodeClickCount = 0;
        _clearAllLegend();
        _refresh();
        document.getElementById('info').style.display = 'none';
        return;
      }}

      /* Pan */
      if (_MOVE_KEYS.indexOf(e.key) !== -1) {{
        e.preventDefault();
        _keys[e.key] = true;
        if (!_rafId) _rafId = requestAnimationFrame(_keyLoop);
      }}
    }});

    window.addEventListener('keyup', function (e) {{
      delete _keys[e.key];
      /* Stop loop if no move keys held */
      if (_rafId && !_MOVE_KEYS.some(function (k) {{ return _keys[k]; }})) {{
        cancelAnimationFrame(_rafId);
        _rafId = null;
      }}
    }});

    /* -- URL hash focus: graph.html#NODE-ID --------------------------------- */
    var hashId = decodeURIComponent(window.location.hash.slice(1));
    if (hashId) {{
      var _focused = false;
      Graph.onEngineStop(function () {{
        if (_focused) return;
        var node = GD.nodes.find(function (n) {{ return n.id === hashId; }});
        if (node && node.x !== undefined) {{
          _focused = true;
          Graph.centerAt(node.x, node.y, 800);
          Graph.zoom(5, 800);
        }}
      }});
    }}
  }})();
  </script>
</body>
</html>
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _community_color(community_id: str | None) -> str:
    if not community_id:
        return "#888888"
    try:
        idx = int(community_id.split("_")[-1])
        return _COMMUNITY_COLORS[idx % len(_COMMUNITY_COLORS)]
    except (ValueError, IndexError):
        return "#888888"


def _build_obsidian_uri(vault_name: str, file_path: str) -> str:
    clean_path = file_path.lstrip("/")
    return f"obsidian://open?vault={quote(vault_name)}&file={quote(clean_path)}"


def _is_portal_node(kind: str, data: dict) -> bool:
    if kind == "context_ref":
        return True
    meta = data.get("metadata") or {}
    return bool(data.get("cross_namespace") or meta.get("cross_namespace"))


def _node_val(degree: int) -> float:
    """Map structural degree to force-graph node val (size + collision radius)."""
    return max(1.0, math.sqrt(degree) * 3)


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_html(
    graph: KnowledgeGraph,
    output_path: Path,
    height: str = "900px",   # kept for API compat — ignored (canvas is fullscreen)
    width: str = "100%",     # kept for API compat — ignored
    bg_color: str = "#1a1a2e",
    font_color: str = "#e0e0e0",
    vault_name: str | None = None,
    on_demand_relations: set[str] | None = None,
    min_degree: int = 0,
) -> None:
    """Generate an interactive HTML graph visualization using force-graph.

    Args:
        graph: Knowledge graph to visualize.
        output_path: Where to write graph.html.
        height: Ignored (canvas is always fullscreen). Kept for API compat.
        width: Ignored. Kept for API compat.
        bg_color: Page background color.
        font_color: Label font color.
        vault_name: Obsidian vault name for deep-link URIs.
        on_demand_relations: Edge relation types shown only on node click.
            Defaults to {"mentions_symbol"}.
        min_degree: Only render nodes with structural degree >= N (0 = all).
    """
    if on_demand_relations is None:
        on_demand_relations = {"mentions_symbol"}

    # ── Structural degree (excludes on-demand relations) ─────────────────────
    degree_map: dict[str, int] = {}
    for src, tgt, edata in graph.g.edges(data=True):
        if edata.get("relation") in on_demand_relations:
            continue
        degree_map[src] = degree_map.get(src, 0) + 1
        degree_map[tgt] = degree_map.get(tgt, 0) + 1

    if min_degree > 0:
        visible_nodes: set[str] = {
            n for n, d in degree_map.items() if d >= min_degree
        }
    else:
        visible_nodes = set(graph.g.nodes())

    # ── Build nodes ──────────────────────────────────────────────────────────
    nodes: list[dict] = []

    for node_id, data in graph.g.nodes(data=True):
        if node_id not in visible_nodes:
            continue

        kind = data.get("kind", "note")
        label = data.get("label", node_id)
        community_id = data.get("community_id")
        tokens = data.get("tokens", 0)
        degree = degree_map.get(node_id, 0)

        is_portal = _is_portal_node(kind, data)

        if is_portal:
            color = _PORTAL_COLOR
            label_display = f"P {label}"
            meta = data.get("metadata") or {}
            cross_ns = data.get("cross_namespace") or meta.get("cross_namespace", "")
            if isinstance(cross_ns, str) and "::" in cross_ns:
                target_ns, _, target_id = cross_ns.partition("::")
                portal_href = f"../{target_ns}/graph.html#{quote(target_id)}"
            elif kind == "context_ref":
                source_file = data.get("source_file", "")
                portal_href = f"#{quote(source_file)}" if source_file else ""
            else:
                portal_href = ""
            obsidian_uri = ""
        elif data.get("namespace") == "code" or node_id.startswith("code::"):
            # Code node (real or stub). namespace="code" is set by graph.py for new
            # ingests; node_id prefix is the fallback for graphs ingested before fix.
            # Slate-blue for unknown/external code kinds (logging, pathlib, typing etc.)
            label_display = label
            color = _KIND_COLORS.get(kind, "#5a7fa8")
            portal_href = ""
            obsidian_uri = ""
        else:
            # Doc node: community-based color
            label_display = label
            color = _community_color(community_id)
            portal_href = ""
            obsidian_uri = ""
            if vault_name and kind in ("note", "knowledge"):
                source_file = data.get("source_file", "")
                knowledge_id = data.get("knowledge_id")
                if knowledge_id and not source_file:
                    source_file = f"knowledge/{knowledge_id}.md"
                if source_file:
                    obsidian_uri = _build_obsidian_uri(vault_name, source_file)

        source_file = data.get("source_file", "")
        meta_d = data.get("metadata") or {}
        ls, le = meta_d.get("line_start"), meta_d.get("line_end")
        loc = f":{ls}-{le}" if ls else ""
        sig = meta_d.get("signature", "")
        doc = meta_d.get("docstring", "")

        tooltip_lines = [
            f"{label} [{kind}]",
            f"degree: {degree}  tokens: {tokens}",
        ]
        if source_file:
            tooltip_lines.append(f"{source_file}{loc}")
        if sig:
            tooltip_lines.append(sig[:80])
        if doc:
            tooltip_lines.append(doc.split("\n")[0][:100])

        node_entry: dict = {
            "id": node_id,
            "label": label_display[:32],
            "val": _node_val(degree),
            "color": color,
            "kind": kind,
            "degree": degree,
            "community": community_id or "",
            "source_file": (source_file + loc) if source_file else "",
            "sig": sig[:80] if sig else "",
            "tooltip": "\n".join(tooltip_lines),
        }
        if obsidian_uri:
            node_entry["obsidian_uri"] = obsidian_uri
        if portal_href:
            node_entry["portal_href"] = portal_href

        nodes.append(node_entry)

    # ── Build links ──────────────────────────────────────────────────────────
    links: list[dict] = []
    od_links: list[dict] = []

    for source, target, data in graph.g.edges(data=True):
        if source not in visible_nodes or target not in visible_nodes:
            continue

        relation = data.get("relation", "?")
        confidence = data.get("confidence", "EXTRACTED")
        score = float(data.get("confidence_score", 1.0))
        weight = float(data.get("weight", 1.0))

        edge_color = _EDGE_COLORS.get(confidence, "rgba(150,150,150,0.4)")
        edge_width = max(0.5, weight * 1.5)
        particles = 2 if relation in _PARTICLE_RELATIONS else 0

        entry: dict = {
            "source": source,
            "target": target,
            "color": edge_color,
            "width": edge_width,
            "particles": particles,
            "title": f"{relation} | {confidence} | {score:.2f}",
            "relation": relation,
        }

        if relation in on_demand_relations:
            entry["_onDemand"] = True
            od_links.append(entry)
        else:
            links.append(entry)

    all_links = links + od_links
    graph_data = {"nodes": nodes, "links": all_links}

    title = vault_name or output_path.parent.name or "PrismRag Graph"

    # ── Community legend: group doc nodes by rendered color ──────────────────
    # Multiple community_ids may map to the same color (12-color palette wraps
    # over 95 communities). Count by actual color so numbers reflect real sizes.
    # For each color we pick the hub node (highest degree doc node) as the cluster name.
    # Only "note" and "knowledge" kinds qualify — tag nodes are excluded to prevent
    # high-degree tags (e.g. tag:Ollama) from stealing the label.
    _DOC_KINDS = {"note", "knowledge"}
    color_counts: dict[str, int] = {}
    color_hub: dict[str, tuple[int, str]] = {}  # color → (max_degree, hub_label)
    for node_entry in nodes:
        kind_ = node_entry.get("kind", "")
        nid_ = node_entry.get("id", "")
        # Only count genuine doc nodes; skip code, tags, portals
        if kind_ not in _DOC_KINDS:
            continue
        if kind_ in _KIND_COLORS or kind_ == "context_ref" or nid_.startswith("code::"):
            continue
        c = node_entry.get("color", "")
        if c and c != "#888888":
            color_counts[c] = color_counts.get(c, 0) + 1
            deg = node_entry.get("degree", 1)  # raw degree for hub selection
            label = node_entry.get("label") or node_entry.get("id", "")
            # Strip namespace prefix for display (e.g. "nimbus::Foo" → "Foo")
            if "::" in label:
                label = label.split("::")[-1]
            prev_deg, _ = color_hub.get(c, (-1, ""))
            if deg > prev_deg:
                color_hub[c] = (deg, label)

    # Top 8 colors by doc node count
    top_colors = sorted(color_counts, key=lambda c: -color_counts[c])[:8]
    community_legend_rows = []
    for color in top_colors:
        count = color_counts[color]
        hub_label = color_hub.get(color, (0, "doc group"))[1] or "doc group"
        community_legend_rows.append(
            f'<div class="leg-row leg-filter" data-color="{color}">'
            f'<span class="swatch" style="background:{color}"></span>'
            f'{hub_label} ({count})'
            f'</div>'
        )
    community_legend_html = "\n        ".join(community_legend_rows) if community_legend_rows else ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = _HTML_TEMPLATE.format(
        title=title,
        bg_color=bg_color,
        font_color=font_color,
        node_count=len(nodes),
        link_count=len(links),
        od_count=len(od_links),
        community_legend_html=community_legend_html,
        graph_data_json=json.dumps(graph_data, ensure_ascii=False, separators=(",", ":")),
    )
    output_path.write_text(html, encoding="utf-8")

    logger.info(
        "[visualize] saved force-graph HTML to %s "
        "(%d nodes, %d structural + %d on-demand links)",
        output_path, len(nodes), len(links), len(od_links),
    )
