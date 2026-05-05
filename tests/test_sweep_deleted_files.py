"""Tests for incremental.sweep_deleted_files (Sprint 1 protocol)."""
from __future__ import annotations

from pathlib import Path

from prism_rag.ingest.incremental import sweep_deleted_files
from prism_rag.store.graph import Edge, KnowledgeGraph, LifecycleClass, Node


def test_sweep_removes_node_for_missing_file(tmp_path):
    g = KnowledgeGraph()
    g.add_node(Node(id="alive", label="alive", source_file="alive.md"))
    g.add_node(Node(id="dead", label="dead", source_file="dead.md"))
    (tmp_path / "alive.md").write_text("hi")
    # dead.md is intentionally absent
    removed = sweep_deleted_files(g, tmp_path)
    assert removed == 1
    assert "alive" in g.g
    assert "dead" not in g.g


def test_sweep_skips_nodes_with_no_source_file(tmp_path):
    g = KnowledgeGraph()
    g.add_node(Node(id="anonymous", label="anonymous", source_file=""))
    removed = sweep_deleted_files(g, tmp_path)
    assert removed == 0
    assert "anonymous" in g.g


def test_sweep_is_idempotent(tmp_path):
    g = KnowledgeGraph()
    g.add_node(Node(id="dead", label="dead", source_file="dead.md"))
    sweep_deleted_files(g, tmp_path)
    second = sweep_deleted_files(g, tmp_path)
    assert second == 0
