"""Tests for LifecycleClass enum + Edge.lifecycle_class field (Sprint 1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_rag.store.graph import (
    Edge,
    KnowledgeGraph,
    LifecycleClass,
    Node,
)


def test_lifecycle_class_values():
    assert LifecycleClass.PROBABILISTIC == "probabilistic"
    assert LifecycleClass.DETERMINISTIC == "deterministic"
    assert LifecycleClass.ANCHORED == "anchored"


def test_edge_default_lifecycle_class():
    e = Edge(source="a", target="b", relation="r")
    assert e.lifecycle_class == LifecycleClass.PROBABILISTIC


def test_edge_explicit_lifecycle_class():
    e = Edge(source="a", target="b", relation="mentions_symbol",
             lifecycle_class=LifecycleClass.ANCHORED)
    assert e.lifecycle_class == LifecycleClass.ANCHORED
    assert e.to_dict()["lifecycle_class"] == "anchored"


def test_graph_save_and_load_preserves_lifecycle(tmp_path):
    g = KnowledgeGraph()
    g.add_node(Node(id="n1", label="N1"))
    g.add_node(Node(id="n2", label="N2"))
    g.add_edge(Edge(source="n1", target="n2", relation="mentions_symbol",
                    lifecycle_class=LifecycleClass.ANCHORED))
    out = tmp_path / "graph.json"
    g.save(out)
    g2 = KnowledgeGraph.load(out)
    edge_data = g2.g.edges["n1", "n2"]
    assert edge_data["lifecycle_class"] == "anchored"


def test_legacy_edge_load_defaults_probabilistic(tmp_path):
    """Edges in older graph.json without lifecycle_class field load as PROBABILISTIC."""
    out = tmp_path / "graph.json"
    out.write_text(json.dumps({
        "metadata": {"version": "v4.0", "node_count": 2, "edge_count": 1, "community_count": 0},
        "nodes": [
            {"id": "n1", "label": "N1", "kind": "note"},
            {"id": "n2", "label": "N2", "kind": "note"},
        ],
        "edges": [
            {"source": "n1", "target": "n2", "relation": "r", "confidence": "EXTRACTED",
             "confidence_score": 1.0, "weight": 1.0, "source_pass": "ast"}
        ],
        "communities": [],
    }))
    g = KnowledgeGraph.load(out)
    edge_data = g.g.edges["n1", "n2"]
    assert edge_data.get("lifecycle_class", "probabilistic") == "probabilistic"


def test_graph_save_is_atomic(tmp_path):
    """KnowledgeGraph.save must use atomic_write — concurrent readers never see torn JSON.

    Uses a ~110KB payload so that on Linux ext4/tmpfs a non-atomic write_text
    would empirically interleave with reads. The torn-detect uses json.loads,
    which catches any structural break (truncated, mid-write, mixed old+new).
    """
    import json as _json
    import threading

    g = KnowledgeGraph()
    for i in range(200):
        g.add_node(Node(id=f"n{i}", label=f"N{i}",
                        content="x" * 200, source_file=f"file{i}.md"))
    out = tmp_path / "graph.json"
    g.save(out)

    stop = threading.Event()
    saw_partial: list[str] = []

    def reader() -> None:
        while not stop.is_set():
            try:
                txt = out.read_text()
            except FileNotFoundError:
                continue
            if not txt:
                continue
            try:
                _json.loads(txt)
            except _json.JSONDecodeError as exc:
                saw_partial.append(f"{type(exc).__name__}: {str(exc)[:60]}")

    rt = threading.Thread(target=reader)
    rt.start()
    try:
        for _ in range(30):
            g.save(out)
    finally:
        stop.set()
        rt.join()

    assert saw_partial == [], f"saw {len(saw_partial)} torn reads: {saw_partial[:3]}"
