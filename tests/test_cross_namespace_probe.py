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
