"""Impact analysis: directional BFS with confidence filtering.

Answers the question: "If I change node X, what else is affected?"

    downstream  — X depends on what? (follow out-edges: X → …)
    upstream    — who depends on X? (follow in-edges: … → X)
    both        — union of downstream + upstream

Results are grouped by traversal depth so callers can triage:

    Depth 1 → DIRECTLY AFFECTED  (immediate neighbours)
    Depth 2 → LIKELY AFFECTED
    Depth 3+→ MAY BE AFFECTED
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

Direction = Literal["downstream", "upstream", "both"]


def impact_bfs(
    graph: "KnowledgeGraph",
    target_id: str,
    direction: Direction = "upstream",
    max_depth: int = 3,
    min_confidence: float = 0.7,
) -> dict[int, list[str]]:
    """Compute impact graph via directional BFS.

    Args:
        graph: The KnowledgeGraph to traverse.
        target_id: Starting node (the one being changed).
        direction:
            "upstream"   — who calls / references / depends on target?
            "downstream" — what does target call / reference / depend on?
            "both"       — union of both directions.
        max_depth: Maximum traversal hops (default 3).
        min_confidence: Skip edges with confidence_score below this value.

    Returns:
        dict mapping depth (int ≥ 1) → list of affected node IDs.
        Depth 0 (the target itself) is excluded.

    Example::

        result = impact_bfs(graph, "LlmNode", direction="upstream")
        # {1: ["ClaudeSDKNode", "GeminiCLINode"], 2: [...], ...}
    """
    if target_id not in graph.g:
        logger.debug(f"[impact] target not found: {target_id!r}")
        return {}

    visited: set[str] = {target_id}
    queue: deque[tuple[str, int]] = deque([(target_id, 0)])
    result: dict[int, list[str]] = {}

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue

        neighbours = _neighbours(graph, current, direction)
        for neighbour, confidence in neighbours:
            if confidence < min_confidence:
                continue
            if neighbour in visited:
                continue
            visited.add(neighbour)
            result.setdefault(depth + 1, []).append(neighbour)
            queue.append((neighbour, depth + 1))

    return result


def _neighbours(
    graph: "KnowledgeGraph",
    node_id: str,
    direction: Direction,
) -> list[tuple[str, float]]:
    """Return (neighbour_id, confidence_score) pairs for the given direction."""
    pairs: list[tuple[str, float]] = []

    if direction in ("downstream", "both"):
        for successor in graph.g.successors(node_id):
            data = graph.g.edges[node_id, successor]
            conf = float(data.get("confidence_score", 1.0))
            pairs.append((successor, conf))

    if direction in ("upstream", "both"):
        for predecessor in graph.g.predecessors(node_id):
            data = graph.g.edges[predecessor, node_id]
            conf = float(data.get("confidence_score", 1.0))
            pairs.append((predecessor, conf))

    return pairs


def format_impact_report(
    graph: "KnowledgeGraph",
    target_id: str,
    impact: dict[int, list[str]],
    direction: Direction,
) -> str:
    """Format an impact result as a human-readable string."""
    if not impact:
        return f"No impact found for `{target_id}` (direction={direction})."

    labels = {
        1: "DIRECTLY AFFECTED",
        2: "LIKELY AFFECTED",
    }

    lines = [f"Impact analysis for **{target_id}** (direction={direction})\n"]
    total = sum(len(v) for v in impact.values())
    lines.append(f"Total affected nodes: {total}\n")

    for depth in sorted(impact):
        tag = labels.get(depth, f"MAY BE AFFECTED (depth {depth})")
        lines.append(f"\n### Depth {depth} — {tag}")
        for nid in impact[depth]:
            data = graph.g.nodes.get(nid, {})
            label = data.get("label", nid)
            kind = data.get("kind", "?")
            ns = data.get("namespace", "nimbus")
            lines.append(f"  - [{ns}] **{label}** (`{nid}`, kind={kind})")

    return "\n".join(lines)
