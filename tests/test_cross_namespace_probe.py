from prism_rag.store.cross_namespace_probe import (
    CrossEdgeEntry, CrossNamespaceProbe, MIGRATION_PENDING,
)
from prism_rag.store.graph import LifecycleClass


def test_entry_has_new_v52_fields():
    e = CrossEdgeEntry(
        edge_id="x", source_node="s", target_node="t",
        edge_kind="embedding_similar", confidence_tier="INFERRED",
        confidence=0.7, first_seen_at="2026-05-04T00:00:00Z",
    )
    assert e.consecutive_seen == 1
    assert e.last_seen_parsed_at == ""
    assert e.source_file == ""
    assert e.model_id == ""
    assert e.lifecycle_class == LifecycleClass.PROBABILISTIC


def test_migration_pending_constant():
    assert MIGRATION_PENDING == "MIGRATION_PENDING"


def _bridge(src, tgt, weight=0.74, source_file="a.py"):
    return {
        "source_ns": "code", "source_id": src,
        "target_ns": "nimbus", "target_id": tgt,
        "relation": "embedding_similar", "confidence": "INFERRED",
        "weight": weight, "source_file": source_file,
    }


def test_record_new_entry():
    p = CrossNamespaceProbe(model_id="bge-m3")
    p.record(_bridge("a.py::Foo", "doc"), scan_timestamp="t1")
    e = list(p._index.values())[0]
    assert e.consecutive_seen == 1
    assert e.last_seen_parsed_at == "t1"
    assert e.model_id == "bge-m3"
    assert e.lifecycle_class == LifecycleClass.PROBABILISTIC


def test_record_same_scan_is_noop():
    p = CrossNamespaceProbe(model_id="bge-m3")
    p.record(_bridge("a.py::Foo", "doc"), scan_timestamp="t1")
    p.record(_bridge("a.py::Foo", "doc"), scan_timestamp="t1")
    e = list(p._index.values())[0]
    assert e.consecutive_seen == 1


def test_record_new_scan_increments_consecutive():
    p = CrossNamespaceProbe(model_id="bge-m3")
    p.record(_bridge("a.py::Foo", "doc"), scan_timestamp="t1")
    p.record(_bridge("a.py::Foo", "doc"), scan_timestamp="t2")
    e = list(p._index.values())[0]
    assert e.consecutive_seen == 2
    assert e.last_seen_parsed_at == "t2"


def test_record_model_change_resets():
    p = CrossNamespaceProbe(model_id="bge-m3")
    p.record(_bridge("a.py::Foo", "doc", weight=0.74), scan_timestamp="t1")
    p.record(_bridge("a.py::Foo", "doc", weight=0.74), scan_timestamp="t2")
    p._model_id = "qwen3-embedding-8b"
    p.record(_bridge("a.py::Foo", "doc", weight=0.85), scan_timestamp="t3")
    e = list(p._index.values())[0]
    assert e.consecutive_seen == 1
    assert e.confidence == 0.85
    assert e.model_id == "qwen3-embedding-8b"


def test_record_anchored_short_circuits():
    p = CrossNamespaceProbe(model_id="bge-m3")
    p.record(_bridge("a.py::Foo", "doc"), scan_timestamp="t1")
    eid = list(p._index.keys())[0]
    p._index[eid].lifecycle_class = LifecycleClass.ANCHORED
    p._index[eid].consecutive_seen = 99
    p.record(_bridge("a.py::Foo", "doc"), scan_timestamp="t2")
    assert p._index[eid].consecutive_seen == 99
    assert p._index[eid].last_seen_parsed_at == "t1"


def test_record_migration_pending_overwrite():
    p = CrossNamespaceProbe(model_id="bge-m3")
    eid = "code::a.py::Foo→nimbus::doc"
    p._index[eid] = CrossEdgeEntry(
        edge_id=eid, source_node="code::a.py::Foo", target_node="nimbus::doc",
        edge_kind="embedding_similar", confidence_tier="INFERRED",
        confidence=0.5, first_seen_at="2026-04-01T00:00:00Z",
        last_seen_parsed_at=MIGRATION_PENDING, source_file="a.py",
        consecutive_seen=1, model_id="bge-m3",
        lifecycle_class=LifecycleClass.PROBABILISTIC,
    )
    p.record(_bridge("a.py::Foo", "doc", weight=0.74), scan_timestamp="t-now")
    e = p._index[eid]
    assert e.last_seen_parsed_at == "t-now"
    assert e.consecutive_seen == 1
    assert e.confidence == 0.74
