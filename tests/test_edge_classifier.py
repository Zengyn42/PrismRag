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


from prism_rag.config import ClassifierProfile
from prism_rag.ingest.edge_classifier import classify_one, TIER_1, TIER_2, TIER_3
from prism_rag.store.cross_namespace_probe import MIGRATION_PENDING
from prism_rag.store.graph import LifecycleClass


_PROFILE = ClassifierProfile(
    tier1_min_conf=0.75, tier1_top_k=1, tier1_min_consecutive=2,
    tier2_min_conf=0.70, tier2_margin=0.25, tier2_hard_cap=5, tier2_min_consecutive=2,
)


def _probe_entry(conf=0.80, consec=2, parsed_at="t1", anchored=False):
    e = _e(conf, "x")
    e.consecutive_seen = consec
    e.last_seen_parsed_at = parsed_at
    if anchored:
        e.lifecycle_class = LifecycleClass.ANCHORED
    return e


def test_classify_migration_pending_short_circuits():
    e = _probe_entry(conf=0.99, consec=10)
    e.last_seen_parsed_at = MIGRATION_PENDING
    assert classify_one(e, is_top_1=True, profile=_PROFILE) == TIER_3


def test_classify_anchored_skips():
    e = _probe_entry(anchored=True)
    assert classify_one(e, is_top_1=True, profile=_PROFILE) == TIER_3


def test_classify_tier1():
    e = _probe_entry(conf=0.80, consec=2)
    assert classify_one(e, is_top_1=True, profile=_PROFILE) == TIER_1


def test_classify_tier1_requires_top_1():
    e = _probe_entry(conf=0.80, consec=2)
    assert classify_one(e, is_top_1=False, profile=_PROFILE) == TIER_2


def test_classify_tier1_requires_consecutive():
    e = _probe_entry(conf=0.80, consec=1)
    assert classify_one(e, is_top_1=True, profile=_PROFILE) == TIER_2


def test_classify_tier2_low_consec_top1():
    # conf >= tier2_min, top-1 OR consec >= 2 → Tier 2
    e = _probe_entry(conf=0.71, consec=1)
    assert classify_one(e, is_top_1=True, profile=_PROFILE) == TIER_2


def test_classify_tier3_below_min_conf():
    e = _probe_entry(conf=0.60, consec=2)
    assert classify_one(e, is_top_1=True, profile=_PROFILE) == TIER_3
