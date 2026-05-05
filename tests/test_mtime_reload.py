"""Tests for FederatedGraph 500ms mtime reload (Sprint 1 protocol)."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from prism_rag.config import GraphSource
from prism_rag.store.federated import FederatedGraph
from prism_rag.store.graph import Edge, KnowledgeGraph, Node


def _build_simple_graph(path: Path) -> None:
    g = KnowledgeGraph()
    g.add_node(Node(id="a", label="A"))
    g.save(path)


def test_maybe_reload_throttles_within_500ms(tmp_path):
    src = GraphSource(namespace="ns", vault_path=tmp_path, data_dir=tmp_path)
    _build_simple_graph(src.graph_path)
    fg = FederatedGraph.load([src])
    # Force the throttle window to elapse so the first _maybe_reload would
    # otherwise re-stat — this isolates the throttle effect across the loop.
    fg._last_check_at = 0.0

    original_stat = Path.stat
    call_count = 0

    def counting_stat(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_stat(self, *args, **kwargs)

    with patch.object(Path, "stat", counting_stat):
        for _ in range(50):
            fg._maybe_reload()
    # First call resets _last_check_at and stats N namespaces (here N=1).
    # Subsequent 49 calls are inside the 500ms window → no stat.
    # So total stats == 1 (one namespace × one allowed window entry).
    assert call_count == 1, f"expected exactly 1 stat call, got {call_count}"


def test_maybe_reload_picks_up_mtime_change(tmp_path):
    src = GraphSource(namespace="ns", vault_path=tmp_path, data_dir=tmp_path)
    _build_simple_graph(src.graph_path)
    fg = FederatedGraph.load([src])
    fg._maybe_reload()
    initial_count = fg.get_graph("ns").node_count

    # Modify the graph on disk
    g2 = KnowledgeGraph.load(src.graph_path)
    g2.add_node(Node(id="b", label="B"))
    g2.save(src.graph_path)
    # Bump mtime explicitly in case clock granularity collides
    import os
    future = time.time() + 1.0
    os.utime(src.graph_path, (future, future))

    # Force the throttle window to elapse
    fg._last_check_at = 0.0
    fg._maybe_reload()
    assert fg.get_graph("ns").node_count == initial_count + 1
