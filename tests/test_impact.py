"""Tests for impact_bfs() — Phase 6 tier filtering and path scoring."""

from __future__ import annotations

import pytest

from prism_rag.retrieve.impact import _DEFAULT_TIERS, format_impact_report, impact_bfs
from prism_rag.store.graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _graph_with_edges(edges: list[tuple[str, str, dict]]) -> KnowledgeGraph:
    """Build a KnowledgeGraph from a list of (src, dst, edge_data) triples."""
    kg = KnowledgeGraph()
    all_nodes: set[str] = set()
    for src, dst, _ in edges:
        all_nodes.add(src)
        all_nodes.add(dst)
    for n in all_nodes:
        kg.g.add_node(n, label=n, kind="test", tokens=10)
    for src, dst, data in edges:
        kg.g.add_edge(src, dst, **data)
    return kg


def _edge(tier: str = "EXTRACTED", score: float = 1.0, kind: str = "calls") -> dict:
    return {"confidence": tier, "confidence_score": score, "relation": kind}


# ---------------------------------------------------------------------------
# Basic traversal
# ---------------------------------------------------------------------------

class TestBasicTraversal:
    def test_empty_result_for_missing_node(self):
        kg = _graph_with_edges([("A", "B", _edge())])
        assert impact_bfs(kg, "Z") == {}

    def test_upstream_single_hop(self):
        # A → B: who depends on B? → A
        kg = _graph_with_edges([("A", "B", _edge())])
        result = impact_bfs(kg, "B", direction="upstream")
        assert 1 in result
        ids = [nid for nid, _ in result[1]]
        assert "A" in ids

    def test_downstream_single_hop(self):
        # A → B: what does A depend on? → B
        kg = _graph_with_edges([("A", "B", _edge())])
        result = impact_bfs(kg, "A", direction="downstream")
        assert 1 in result
        ids = [nid for nid, _ in result[1]]
        assert "B" in ids

    def test_both_direction(self):
        # C → A → B
        kg = _graph_with_edges([
            ("C", "A", _edge()),
            ("A", "B", _edge()),
        ])
        result = impact_bfs(kg, "A", direction="both")
        all_ids = {nid for nodes in result.values() for nid, _ in nodes}
        assert "B" in all_ids
        assert "C" in all_ids

    def test_max_depth_respected(self):
        # A → B → C → D
        kg = _graph_with_edges([
            ("A", "B", _edge()),
            ("B", "C", _edge()),
            ("C", "D", _edge()),
        ])
        result = impact_bfs(kg, "A", direction="downstream", max_depth=2)
        all_ids = {nid for nodes in result.values() for nid, _ in nodes}
        assert "B" in all_ids
        assert "C" in all_ids
        assert "D" not in all_ids


# ---------------------------------------------------------------------------
# Tier filtering
# ---------------------------------------------------------------------------

class TestTierFiltering:
    def test_default_tiers_exclude_ambiguous(self):
        # A -[AMBIGUOUS]-> B; default tiers = {EXTRACTED, INFERRED}
        kg = _graph_with_edges([
            ("A", "B", _edge(tier="AMBIGUOUS", score=0.1)),
        ])
        result = impact_bfs(kg, "A", direction="downstream", min_confidence=0.0)
        assert result == {}

    def test_extracted_inferred_both_pass_by_default(self):
        kg = _graph_with_edges([
            ("A", "B", _edge(tier="EXTRACTED", score=0.95)),
            ("A", "C", _edge(tier="INFERRED", score=0.60)),
        ])
        result = impact_bfs(kg, "A", direction="downstream", min_confidence=0.0)
        ids = {nid for nodes in result.values() for nid, _ in nodes}
        assert "B" in ids
        assert "C" in ids

    def test_allowed_tiers_none_passes_all(self):
        kg = _graph_with_edges([
            ("A", "B", _edge(tier="AMBIGUOUS", score=0.1)),
        ])
        result = impact_bfs(
            kg, "A", direction="downstream",
            min_confidence=0.0,
            allowed_tiers=None,
        )
        ids = {nid for nodes in result.values() for nid, _ in nodes}
        assert "B" in ids

    def test_custom_tiers_extracted_only(self):
        kg = _graph_with_edges([
            ("A", "B", _edge(tier="EXTRACTED")),
            ("A", "C", _edge(tier="INFERRED")),
        ])
        result = impact_bfs(
            kg, "A", direction="downstream",
            min_confidence=0.0,
            allowed_tiers=frozenset({"EXTRACTED"}),
        )
        ids = {nid for nodes in result.values() for nid, _ in nodes}
        assert "B" in ids
        assert "C" not in ids


# ---------------------------------------------------------------------------
# Confidence score filtering
# ---------------------------------------------------------------------------

class TestConfidenceFiltering:
    def test_min_confidence_blocks_low_score(self):
        kg = _graph_with_edges([
            ("A", "B", _edge(score=0.5)),
        ])
        result = impact_bfs(kg, "A", direction="downstream", min_confidence=0.7)
        assert result == {}

    def test_min_confidence_passes_high_score(self):
        kg = _graph_with_edges([
            ("A", "B", _edge(score=0.95)),
        ])
        result = impact_bfs(kg, "A", direction="downstream", min_confidence=0.7)
        assert 1 in result


# ---------------------------------------------------------------------------
# Edge kind filtering
# ---------------------------------------------------------------------------

class TestEdgeKindFiltering:
    def test_allowed_edge_kinds_restricts_traversal(self):
        kg = _graph_with_edges([
            ("A", "B", _edge(kind="calls")),
            ("A", "C", _edge(kind="imports")),
        ])
        result = impact_bfs(
            kg, "A", direction="downstream",
            min_confidence=0.0,
            allowed_edge_kinds=frozenset({"calls"}),
        )
        ids = {nid for nodes in result.values() for nid, _ in nodes}
        assert "B" in ids
        assert "C" not in ids

    def test_allowed_edge_kinds_none_passes_all(self):
        kg = _graph_with_edges([
            ("A", "B", _edge(kind="calls")),
            ("A", "C", _edge(kind="imports")),
        ])
        result = impact_bfs(
            kg, "A", direction="downstream",
            min_confidence=0.0,
            allowed_edge_kinds=None,
        )
        ids = {nid for nodes in result.values() for nid, _ in nodes}
        assert "B" in ids
        assert "C" in ids


# ---------------------------------------------------------------------------
# Path scoring
# ---------------------------------------------------------------------------

class TestPathScoring:
    def test_weakest_link_two_hops(self):
        # A -[0.9]-> B -[0.6]-> C
        kg = _graph_with_edges([
            ("A", "B", _edge(score=0.9)),
            ("B", "C", _edge(score=0.6)),
        ])
        result = impact_bfs(
            kg, "A", direction="downstream",
            min_confidence=0.0,
            path_score_fn="weakest_link",
        )
        b_score = dict(result[1])["B"]
        c_score = dict(result[2])["C"]
        assert abs(b_score - 0.9) < 1e-9
        assert abs(c_score - 0.6) < 1e-9  # weakest = min(0.9, 0.6)

    def test_cumulative_decay_extracted_hops(self):
        # A -[EXTRACTED]-> B -[EXTRACTED]-> C → decay = 1.0 * 1.0 = 1.0
        kg = _graph_with_edges([
            ("A", "B", _edge(tier="EXTRACTED", score=1.0)),
            ("B", "C", _edge(tier="EXTRACTED", score=1.0)),
        ])
        result = impact_bfs(
            kg, "A", direction="downstream",
            min_confidence=0.0,
            path_score_fn="cumulative_decay",
        )
        c_score = dict(result[2])["C"]
        assert abs(c_score - 1.0) < 1e-9

    def test_cumulative_decay_inferred_hop(self):
        # A -[INFERRED]-> B  → decay = 0.6
        kg = _graph_with_edges([
            ("A", "B", _edge(tier="INFERRED", score=1.0)),
        ])
        result = impact_bfs(
            kg, "A", direction="downstream",
            min_confidence=0.0,
            path_score_fn="cumulative_decay",
        )
        b_score = dict(result[1])["B"]
        assert abs(b_score - 0.6) < 1e-9

    def test_results_sorted_by_score_descending(self):
        kg = _graph_with_edges([
            ("A", "B", _edge(score=0.5)),
            ("A", "C", _edge(score=0.9)),
            ("A", "D", _edge(score=0.7)),
        ])
        result = impact_bfs(
            kg, "A", direction="downstream",
            min_confidence=0.0,
            path_score_fn="weakest_link",
        )
        scores = [s for _, s in result[1]]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# format_impact_report
# ---------------------------------------------------------------------------

class TestFormatImpactReport:
    def test_empty_impact(self):
        kg = _graph_with_edges([("A", "B", _edge())])
        report = format_impact_report(kg, "A", {}, "upstream")
        assert "No impact found" in report

    def test_report_contains_score(self):
        kg = _graph_with_edges([("A", "B", _edge(score=0.95))])
        result = impact_bfs(kg, "A", direction="downstream", min_confidence=0.0)
        report = format_impact_report(kg, "A", result, "downstream")
        assert "score=" in report
        assert "0.95" in report

    def test_report_labels_depth_1_as_directly_affected(self):
        kg = _graph_with_edges([("A", "B", _edge())])
        result = impact_bfs(kg, "A", direction="downstream", min_confidence=0.0)
        report = format_impact_report(kg, "A", result, "downstream")
        assert "DIRECTLY AFFECTED" in report


# ---------------------------------------------------------------------------
# _DEFAULT_TIERS constant
# ---------------------------------------------------------------------------

def test_default_tiers_constant():
    assert "EXTRACTED" in _DEFAULT_TIERS
    assert "INFERRED" in _DEFAULT_TIERS
    assert "AMBIGUOUS" not in _DEFAULT_TIERS
