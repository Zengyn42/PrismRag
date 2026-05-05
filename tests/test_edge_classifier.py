from __future__ import annotations

import pytest

from prism_rag.ingest.edge_classifier import select_tier2_candidates
from prism_rag.store.cross_namespace_probe import CrossEdgeEntry


def _e(conf: float, eid: str = "x") -> CrossEdgeEntry:
    return CrossEdgeEntry(
        edge_id=eid, source_node="s", target_node="t",
        edge_kind="embedding_similar", confidence_tier="INFERRED",
        confidence=conf, first_seen_at="x",
    )


def test_select_within_margin():
    ranked = [_e(0.80, "a"), _e(0.74, "b"), _e(0.72, "c"), _e(0.55, "d"), _e(0.30, "e")]
    out = select_tier2_candidates(ranked, margin=0.25, hard_cap=5)
    # threshold = 0.80 * (1-0.25) = 0.60 → [a, b, c] (d=0.55, e=0.30 cut)
    assert [x.edge_id for x in out] == ["a", "b", "c"]


def test_select_hard_cap():
    ranked = [_e(0.80 - 0.001 * i, str(i)) for i in range(20)]
    out = select_tier2_candidates(ranked, margin=0.25, hard_cap=5)
    assert len(out) == 5


def test_select_empty():
    assert select_tier2_candidates([], margin=0.25, hard_cap=5) == []
