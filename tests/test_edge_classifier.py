from __future__ import annotations

import pytest
from pathlib import Path

from prism_rag.config import ClassifierProfile, GraphSource, PrismRagSettings, get_classifier_profile
from prism_rag.ingest.edge_classifier import (
    classify_and_route,
    classify_one,
    select_tier2_candidates,
    ClassifyReport,
    TIER_1,
    TIER_2,
    TIER_3,
)
from prism_rag.inbox.store import InboxStore
from prism_rag.store.cross_namespace_probe import CrossEdgeEntry, CrossNamespaceProbe, MIGRATION_PENDING
from prism_rag.store.federated import FederatedGraph
from prism_rag.store.graph import KnowledgeGraph, LifecycleClass, Node


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


def test_classify_and_route_first_run_all_noop_due_to_migration_pending(tmp_path):
    src = GraphSource(namespace="nimbus", vault_path=tmp_path, data_dir=tmp_path)
    g = KnowledgeGraph()
    g.add_node(Node(id="doc", label="doc"))
    g.save(src.graph_path)
    fg = FederatedGraph.load([src])

    probe = CrossNamespaceProbe(model_id="bge-m3")
    # Inject a "migration" entry directly: consecutive=1, last_seen_parsed_at=MIGRATION_PENDING
    eid = "code::a.py::Foo→nimbus::doc"
    probe._index[eid] = CrossEdgeEntry(
        edge_id=eid, source_node="code::a.py::Foo", target_node="nimbus::doc",
        edge_kind="embedding_similar", confidence_tier="INFERRED",
        confidence=0.99, first_seen_at="x",
        last_seen_parsed_at=MIGRATION_PENDING, source_file="a.py",
        consecutive_seen=1, model_id="bge-m3",
    )
    inbox_path = tmp_path / "inbox.jsonl"
    inbox = InboxStore(inbox_path)
    settings = PrismRagSettings()
    profile = get_classifier_profile(settings, "bge-m3")
    report = classify_and_route(fg, probe, inbox, src, profile)
    assert report.promoted == 0
    assert report.queued == 0


def test_classify_and_route_promotes_tier1(tmp_path):
    src = GraphSource(namespace="nimbus", vault_path=tmp_path, data_dir=tmp_path)
    g = KnowledgeGraph()
    g.add_node(Node(id="doc", label="doc"))
    g.save(src.graph_path)
    fg = FederatedGraph.load([src])
    probe = CrossNamespaceProbe(model_id="bge-m3")
    eid = "code::a.py::Foo→nimbus::doc"
    probe._index[eid] = CrossEdgeEntry(
        edge_id=eid, source_node="code::a.py::Foo", target_node="nimbus::doc",
        edge_kind="embedding_similar", confidence_tier="INFERRED",
        confidence=0.80, first_seen_at="x",
        last_seen_parsed_at="t1", source_file="a.py",
        consecutive_seen=2, model_id="bge-m3",
    )
    inbox = InboxStore(tmp_path / "inbox.jsonl")
    settings = PrismRagSettings()
    profile = get_classifier_profile(settings, "bge-m3")
    report = classify_and_route(fg, probe, inbox, src, profile)
    assert report.promoted == 1
    nimbus = fg.get_graph("nimbus")
    assert nimbus.g.has_edge("doc", "code::a.py::Foo")
    edge_data = nimbus.g.edges["doc", "code::a.py::Foo"]
    assert edge_data["lifecycle_class"] == LifecycleClass.ANCHORED
    # probe entry should now be ANCHORED
    assert probe._index[eid].lifecycle_class == LifecycleClass.ANCHORED
