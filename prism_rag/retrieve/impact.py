"""Impact analysis: directional BFS with confidence and tier filtering.

Answers the question: "If I change node X, what else is affected?"

    downstream  — X depends on what? (follow out-edges: X → …)
    upstream    — who depends on X? (follow in-edges: … → X)
    both        — union of downstream + upstream

Results are grouped by traversal depth so callers can triage:

    Depth 1 → DIRECTLY AFFECTED  (immediate neighbours)
    Depth 2 → LIKELY AFFECTED
    Depth 3+→ MAY BE AFFECTED

Each node is paired with its path_score (float 0.0–1.0). Two scoring modes:
    "weakest_link"     — min(edge.confidence_score) along the path
    "cumulative_decay" — product of TIER_DECAY[tier] for each edge on the path
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

Direction = Literal["downstream", "upstream", "both"]
PathScoreFn = Literal["weakest_link", "cumulative_decay"]

_DEFAULT_TIERS: frozenset[str] = frozenset({"EXTRACTED", "INFERRED"})

_TIER_DECAY: dict[str, float] = {
    "EXTRACTED": 1.0,
    "INFERRED": 0.6,
    "AMBIGUOUS": 0.2,
}


def impact_bfs(
    graph: "KnowledgeGraph",
    target_id: str,
    direction: Direction = "upstream",
    max_depth: int = 3,
    min_confidence: float = 0.7,
    allowed_tiers: frozenset[str] | None = _DEFAULT_TIERS,
    allowed_edge_kinds: frozenset[str] | None = None,
    path_score_fn: PathScoreFn = "weakest_link",
    tier_decay: dict[str, float] | None = None,
) -> dict[int, list[tuple[str, float]]]:
    """Compute impact graph via directional BFS with tier and kind filtering.

    Args:
        graph: The KnowledgeGraph to traverse.
        target_id: Starting node (the one being changed).
        direction:
            "upstream"   — who calls / references / depends on target?
            "downstream" — what does target call / reference / depend on?
            "both"       — union of both directions.
        max_depth: Maximum traversal hops (default 3).
        min_confidence: Skip edges with confidence_score below this value.
        allowed_tiers: Only follow edges whose confidence tier is in this set.
            Default: {"EXTRACTED", "INFERRED"} — excludes AMBIGUOUS.
            Pass None to allow all tiers.
        allowed_edge_kinds: If set, only follow edges whose 'relation' field is
            in this set (e.g. frozenset({"calls", "imports"})).
            None means all edge kinds are allowed.
        path_score_fn:
            "weakest_link"     — path score = min(edge.confidence_score) on path
            "cumulative_decay" — path score = product of tier_decay[tier] on path
        tier_decay: Override for tier decay factors used with "cumulative_decay".
            Defaults to {"EXTRACTED": 1.0, "INFERRED": 0.6, "AMBIGUOUS": 0.2}.

    Returns:
        dict mapping depth (int ≥ 1) → list of (node_id, path_score) tuples,
        sorted by path_score descending within each depth.
        Depth 0 (the target itself) is excluded.

    Example::

        result = impact_bfs(graph, "LlmNode", direction="upstream")
        # {1: [("ClaudeSDKNode", 0.95), ("GeminiCLINode", 0.80)], 2: [...]}
    """
    if target_id not in graph.g:
        logger.debug(f"[impact] target not found: {target_id!r}")
        return {}

    decay = tier_decay if tier_decay is not None else _TIER_DECAY
    score_fn = _make_score_fn(path_score_fn, decay)

    visited: set[str] = {target_id}
    # queue: (node_id, depth, path_score_so_far)
    queue: deque[tuple[str, int, float]] = deque([(target_id, 0, 1.0)])
    result: dict[int, list[tuple[str, float]]] = {}

    while queue:
        current, depth, path_score = queue.popleft()
        if depth >= max_depth:
            continue

        for neighbour, edge_data in _neighbours(graph, current, direction):
            tier = edge_data.get("confidence", "EXTRACTED")
            conf = float(edge_data.get("confidence_score", 1.0))
            kind = edge_data.get("relation", "")

            if allowed_tiers is not None and tier not in allowed_tiers:
                continue
            if conf < min_confidence:
                continue
            if allowed_edge_kinds is not None and kind not in allowed_edge_kinds:
                continue
            if neighbour in visited:
                continue

            visited.add(neighbour)
            new_score = score_fn(path_score, conf, tier)
            result.setdefault(depth + 1, []).append((neighbour, new_score))
            queue.append((neighbour, depth + 1, new_score))

    # Sort each depth bucket by score descending
    for depth_list in result.values():
        depth_list.sort(key=lambda pair: pair[1], reverse=True)

    return result


def _make_score_fn(
    name: PathScoreFn,
    decay: dict[str, float],
) -> Callable[[float, float, str], float]:
    if name == "weakest_link":
        return lambda path_score, conf, tier: min(path_score, conf)
    else:  # cumulative_decay
        return lambda path_score, conf, tier: path_score * decay.get(tier, 1.0)


def _neighbours(
    graph: "KnowledgeGraph",
    node_id: str,
    direction: Direction,
) -> list[tuple[str, dict]]:
    """Return (neighbour_id, edge_data) pairs for the given direction."""
    pairs: list[tuple[str, dict]] = []

    if direction in ("downstream", "both"):
        for successor in graph.g.successors(node_id):
            data = graph.g.edges[node_id, successor]
            pairs.append((successor, data))

    if direction in ("upstream", "both"):
        for predecessor in graph.g.predecessors(node_id):
            data = graph.g.edges[predecessor, node_id]
            pairs.append((predecessor, data))

    return pairs


def format_impact_report(
    graph: "KnowledgeGraph",
    target_id: str,
    impact: dict[int, list[tuple[str, float]]],
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
        for nid, score in impact[depth]:
            data = graph.g.nodes.get(nid, {})
            label = data.get("label", nid)
            kind = data.get("kind", "?")
            ns = data.get("namespace", "nimbus")
            lines.append(
                f"  - [{ns}] **{label}** (`{nid}`, kind={kind}, score={score:.2f})"
            )

    return "\n".join(lines)
