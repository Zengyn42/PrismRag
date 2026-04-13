"""Tests for federated multi-graph functionality."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_rag.config import GraphSource, PrismRagSettings


class TestGraphSource:
    def test_graph_source_basic(self, tmp_path):
        src = GraphSource(
            namespace="nimbus",
            vault_path=tmp_path / "vault",
            data_dir=tmp_path / "data" / "nimbus",
        )
        assert src.namespace == "nimbus"
        assert src.writable is False  # default

    def test_graph_source_graph_path(self, tmp_path):
        src = GraphSource(
            namespace="work",
            vault_path=tmp_path / "vault",
            data_dir=tmp_path / "data" / "work",
        )
        assert src.graph_path == tmp_path / "data" / "work" / "graph.json"


class TestSettingsBackwardCompat:
    def test_single_vault_path_still_works(self, tmp_path, monkeypatch):
        """Old-style PRISM_VAULT_PATH + PRISM_DATA_DIR still works."""
        monkeypatch.setenv("PRISM_VAULT_PATH", str(tmp_path / "vault"))
        monkeypatch.setenv("PRISM_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.delenv("PRISM_GRAPHS", raising=False)
        s = PrismRagSettings()
        graphs = s.resolved_graphs
        assert len(graphs) == 1
        assert graphs[0].namespace == "default"
        assert graphs[0].vault_path == tmp_path / "vault"
        assert graphs[0].data_dir == tmp_path / "data"

    def test_explicit_graphs_env(self, tmp_path, monkeypatch):
        """PRISM_GRAPHS JSON overrides vault_path."""
        graphs_json = json.dumps([
            {"namespace": "a", "vault_path": str(tmp_path / "va"), "data_dir": str(tmp_path / "da")},
            {"namespace": "b", "vault_path": str(tmp_path / "vb"), "data_dir": str(tmp_path / "db"), "writable": True},
        ])
        monkeypatch.setenv("PRISM_GRAPHS", graphs_json)
        s = PrismRagSettings()
        graphs = s.resolved_graphs
        assert len(graphs) == 2
        assert graphs[0].namespace == "a"
        assert graphs[1].writable is True
