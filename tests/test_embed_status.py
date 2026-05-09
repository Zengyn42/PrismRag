# tests/test_embed_status.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from prism_rag.cli import app


runner = CliRunner()


@pytest.fixture
def embed_env(tmp_path):
    """Vault + data dir with a small graph and partial embed cache."""
    from prism_rag.store.graph import KnowledgeGraph, Node
    from prism_rag.ingest.embedder import _append_cache_entry

    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()

    # 3 embeddable nodes
    kg = KnowledgeGraph()
    for i in range(3):
        node = Node(id=f"n{i}", label=f"n{i}", kind="note",
                    content=f"content {i}", content_hash=f"sha:{i}")
        kg.add_node(node)
    kg.save(data / "graph.json")

    # Only n0 and n1 are cached
    cache = data / "embed_cache.jsonl"
    _append_cache_entry(cache, "n0", "sha:0", [0.1])
    _append_cache_entry(cache, "n1", "sha:1", [0.2])

    return data


def test_embed_status_shows_progress(embed_env, monkeypatch):
    data = embed_env

    # Patch PrismRagSettings to use our tmp dirs
    from prism_rag.config import PrismRagSettings, GraphSource
    fake_settings = PrismRagSettings(
        graphs=[GraphSource(namespace="nimbus", vault_path=data.parent / "vault", data_dir=data)]
    )

    with patch("prism_rag.cli.PrismRagSettings", return_value=fake_settings), \
         patch("prism_rag.cli.detect_model_device", return_value="gpu"):
        result = runner.invoke(app, ["embed-status"])

    assert result.exit_code == 0, result.output
    output = result.output
    assert "nimbus" in output
    # 2 embedded, 1 pending
    assert "2" in output
    assert "1" in output


def test_embed_status_shows_device(embed_env, monkeypatch):
    data = embed_env

    from prism_rag.config import PrismRagSettings, GraphSource
    fake_settings = PrismRagSettings(
        graphs=[GraphSource(namespace="nimbus", vault_path=data.parent / "vault", data_dir=data)]
    )

    with patch("prism_rag.cli.PrismRagSettings", return_value=fake_settings), \
         patch("prism_rag.cli.detect_model_device", return_value="cpu"):
        result = runner.invoke(app, ["embed-status"])

    assert result.exit_code == 0, result.output
    assert "cpu" in result.output
