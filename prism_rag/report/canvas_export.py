"""Export a KnowledgeGraph subgraph to an Obsidian Canvas (.canvas) file.

Layout strategy: file-group grid
  - Nodes grouped by source_file, sorted by kind (module→class→function)
  - Files arranged in a column-major grid, sorted by directory path
  - Each file's nodes stacked vertically within their column
  - Call/inherits/imports edges drawn between nodes

Card content:
  - module   : filename header + path
  - class    : class name + signature line
  - function : function name + signature + first docstring line

Usage:
    from prism_rag.report.canvas_export import generate_canvas
    generate_canvas(kg, Path("output.canvas"), filter_prefix="framework/")
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# ── Layout constants ─────────────────────────────────────────────────────────
CARD_W        = 420     # px — wide enough for a signature line
CARD_H_MOD    = 80      # module card height
CARD_H_CLASS  = 130     # class card height
CARD_H_FN     = 120     # function card height
NODE_GAP      = 16      # vertical gap between cards in a file column
FILE_GAP_X    = 80      # horizontal gap between file columns
FILE_GAP_Y    = 120     # vertical gap between rows of file columns
FILES_PER_ROW = 6       # tune this to taste

# Obsidian card color codes (1-6 or "")
_KIND_COLOR = {"module": "6", "class": "3", "function": ""}

# Edge colors
_REL_COLOR = {
    "calls":    "#4363d8",
    "inherits": "#911eb4",
    "imports":  "#f58231",
}

CODE_KINDS = frozenset({"function", "class", "module"})
EDGE_RELS   = frozenset({"calls", "inherits", "imports"})


def _card_height(kind: str) -> int:
    return {"module": CARD_H_MOD, "class": CARD_H_CLASS}.get(kind, CARD_H_FN)


def _card_text(nid: str, data: dict) -> str:
    kind  = data.get("kind", "function")
    label = data.get("label", nid.split("::")[-1])
    meta  = data.get("metadata") or {}
    sig   = (meta.get("signature") or "").strip()
    doc   = (meta.get("docstring") or "").split("\n")[0].strip()
    sf    = data.get("source_file", "")
    ls, le = meta.get("line_start"), meta.get("line_end")
    loc   = f"L{ls}–{le}" if ls else ""

    if kind == "module":
        fname = sf.split("/")[-1]
        return f"## 📄 `{fname}`\n`{sf}`"

    if kind == "class":
        sig_short = sig[:72] + "…" if len(sig) > 72 else sig
        text = f"### 🏛 `{label}`\n```python\n{sig_short}\n```"
        return text

    # function
    sig_short = sig[:72] + "…" if len(sig) > 72 else sig
    lines = [f"### `{label}`", f"```python\n{sig_short}\n```"]
    if doc:
        lines.append(f"*{doc[:80]}*")
    if loc:
        lines.append(f"<sub>{sf} {loc}</sub>")
    return "\n".join(lines)


def generate_canvas(
    kg: KnowledgeGraph,
    output_path: Path,
    filter_prefix: str = "framework/",
    files_per_row: int = FILES_PER_ROW,
) -> tuple[int, int]:
    """Generate an Obsidian Canvas file from a filtered subgraph.

    Args:
        kg: The knowledge graph to export.
        output_path: Destination .canvas file path.
        filter_prefix: Only include nodes whose source_file starts with this.
        files_per_row: How many file-columns per row in the grid layout.

    Returns:
        (node_count, edge_count) written to the canvas.
    """
    # ── Group nodes by source file ──────────────────────────────────────────
    by_file: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for nid, data in kg.g.nodes(data=True):
        sf = data.get("source_file", "")
        if data.get("kind") in CODE_KINDS and sf.startswith(filter_prefix):
            by_file[sf].append((nid, data))

    if not by_file:
        logger.warning(f"[canvas] no nodes found under prefix {filter_prefix!r}")
        return 0, 0

    # Sort files: by directory path so siblings are grouped
    files = sorted(by_file.keys())
    for sf in files:
        # module → class → function; within kind sort by label
        by_file[sf].sort(key=lambda x: (
            {"module": 0, "class": 1, "function": 2}.get(x[1].get("kind", "function"), 2),
            x[1].get("label", ""),
        ))

    # ── Compute file-group layout positions ─────────────────────────────────
    # Each file occupies a column slot in a grid. First compute the max column
    # height so rows don't overlap.
    max_col_h = 0
    for sf in files:
        nodes = by_file[sf]
        col_h = sum(_card_height(d.get("kind", "function")) + NODE_GAP for _, d in nodes)
        max_col_h = max(max_col_h, col_h)

    # x/y origin per file index
    def _file_origin(fi: int) -> tuple[int, int]:
        col = fi % files_per_row
        row = fi // files_per_row
        x = col * (CARD_W + FILE_GAP_X)
        y = row * (max_col_h + FILE_GAP_Y)
        return x, y

    # ── Build canvas nodes ───────────────────────────────────────────────────
    canvas_nodes: list[dict] = []
    canvas_edges: list[dict] = []
    nid_to_cid:   dict[str, str] = {}  # knowledge-graph node id → canvas node id

    for fi, sf in enumerate(files):
        fx, fy = _file_origin(fi)
        cy = fy
        for nid, data in by_file[sf]:
            kind = data.get("kind", "function")
            h    = _card_height(kind)
            cid  = f"n{len(canvas_nodes)}"
            nid_to_cid[nid] = cid

            canvas_nodes.append({
                "id":     cid,
                "type":   "text",
                "text":   _card_text(nid, data),
                "x":      fx,
                "y":      cy,
                "width":  CARD_W,
                "height": h,
                "color":  _KIND_COLOR.get(kind, ""),
            })
            cy += h + NODE_GAP

    # ── Build canvas edges ───────────────────────────────────────────────────
    seen_edges: set[tuple[str, str]] = set()
    for nid in nid_to_cid:
        for src, dst, edata in kg.g.out_edges(nid, data=True):
            rel = edata.get("relation", "")
            if rel not in EDGE_RELS:
                continue
            if dst not in nid_to_cid:
                continue
            pair = (nid_to_cid[src], nid_to_cid[dst])
            if pair in seen_edges:
                continue
            seen_edges.add(pair)
            canvas_edges.append({
                "id":       f"e{len(canvas_edges)}",
                "fromNode": pair[0],
                "toNode":   pair[1],
                "label":    rel,
                "color":    _REL_COLOR.get(rel, ""),
            })

    # ── Write ────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"nodes": canvas_nodes, "edges": canvas_edges},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        f"[canvas] wrote {len(canvas_nodes)} nodes, {len(canvas_edges)} edges → {output_path}"
    )
    return len(canvas_nodes), len(canvas_edges)
