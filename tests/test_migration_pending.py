from __future__ import annotations

import json
from pathlib import Path

from prism_rag.store.cross_namespace_probe import (
    CrossNamespaceProbe, MIGRATION_PENDING,
)


def test_legacy_entries_default_migration_pending(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({
        "edge_id": "code::a.py::F→nimbus::d",
        "source_node": "code::a.py::F",
        "target_node": "nimbus::d",
        "edge_kind": "embedding_similar",
        "confidence_tier": "INFERRED",
        "confidence": 0.74,
        "first_seen_at": "2026-04-01T00:00:00Z",
        "evidence": [],
    }) + "\n")
    p = CrossNamespaceProbe(log_path=log, model_id="bge-m3")
    e = list(p._index.values())[0]
    assert e.last_seen_parsed_at == MIGRATION_PENDING
    assert e.consecutive_seen == 1
    assert e.source_file == "a.py"   # back-fill from source_node
    assert e.model_id == "bge-m3"


def test_load_does_not_infer_source_file_for_nimbus_namespace(tmp_path):
    """nimbus::doc/path source_node must NOT be back-filled as source_file."""
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({
        "edge_id": "nimbus::doc/path→code::x.py::Y",
        "source_node": "nimbus::doc/path",
        "target_node": "code::x.py::Y",
        "edge_kind": "embedding_similar",
        "confidence_tier": "INFERRED",
        "confidence": 0.7,
        "first_seen_at": "2026-04-01T00:00:00Z",
        "evidence": [],
    }) + "\n")
    p = CrossNamespaceProbe(log_path=log, model_id="bge-m3")
    e = list(p._index.values())[0]
    assert e.source_file == ""   # NOT "doc/path"


def test_load_infers_source_file_for_code_namespace(tmp_path):
    """code::path::Symbol source_node SHOULD back-fill source_file."""
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({
        "edge_id": "code::pkg/sub.py::Foo→nimbus::d",
        "source_node": "code::pkg/sub.py::Foo",
        "target_node": "nimbus::d",
        "edge_kind": "embedding_similar",
        "confidence_tier": "INFERRED",
        "confidence": 0.7,
        "first_seen_at": "2026-04-01T00:00:00Z",
        "evidence": [],
    }) + "\n")
    p = CrossNamespaceProbe(log_path=log, model_id="bge-m3")
    e = list(p._index.values())[0]
    assert e.source_file == "pkg/sub.py"
