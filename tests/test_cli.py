"""CLI integration tests (Sections 1 and 7)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


# Discover the installed prism-rag script location
VENV_BIN = Path(sys.executable).parent


def _prism_rag_cmd() -> list[str]:
    """Return the command to invoke the PrismRag CLI.

    Prefer the console-script entry point if on PATH; fall back to
    `python -m prism_rag.cli` which always works with an editable install.
    """
    prism_rag_path = VENV_BIN / "prism-rag"
    if prism_rag_path.exists():
        return [str(prism_rag_path)]
    return [sys.executable, "-m", "prism_rag.cli"]


@pytest.fixture
def tiny_vault(tmp_path):
    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()
    (vault / "note.md").write_text("# Hello\n\n[[other]]\n")
    # Minimal graph.json so serve doesn't abort
    from prism_rag.store.graph import KnowledgeGraph, Node
    g = KnowledgeGraph()
    g.add_node(Node(id="note", label="note", kind="note", tokens=10))
    g.save(data / "graph.json")
    return vault, data


def test_serve_stdio_starts_and_exits(tiny_vault):
    """prism-rag serve --transport stdio starts and responds to SIGTERM."""
    vault, data = tiny_vault
    env = os.environ.copy()
    env["PRISM_VAULT_PATH"] = str(vault)
    env["PRISM_DATA_DIR"] = str(data)
    env["PRISM_GEMINI_API_KEY"] = ""

    proc = subprocess.Popen(
        _prism_rag_cmd() + ["serve", "--transport", "stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        time.sleep(1.5)
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode(errors="replace")
            pytest.fail(f"serve exited early with code {proc.returncode}: {stderr}")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
            pytest.fail("serve did not exit within 5s of SIGTERM")
        assert proc.returncode in (0, 143, -15, 1, -2)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)


def test_serve_fails_gracefully_when_no_graph(tmp_path):
    """serve without a graph.json exits non-zero with a clear error."""
    empty_vault = tmp_path / "empty-vault"
    empty_data = tmp_path / "empty-data"
    empty_vault.mkdir()
    empty_data.mkdir()

    env = os.environ.copy()
    env["PRISM_VAULT_PATH"] = str(empty_vault)
    env["PRISM_DATA_DIR"] = str(empty_data)
    env["PRISM_GEMINI_API_KEY"] = ""

    result = subprocess.run(
        _prism_rag_cmd() + ["serve", "--transport", "stdio"],
        capture_output=True, text=True, env=env, timeout=10,
    )
    assert result.returncode != 0
    assert "No graphs loaded" in result.stderr or "graph" in result.stderr.lower()


def test_mock_embedder_deterministic():
    """Mock embedder returns the same 768-dim vector for identical content."""
    from prism_rag.ingest.mock_embedder import mock_embed_text
    v1 = mock_embed_text("hello world")
    v2 = mock_embed_text("hello world")
    v3 = mock_embed_text("different content")
    assert len(v1) == 768
    assert v1 == v2
    assert v1 != v3


def test_ingest_with_no_embedding_flag(tmp_path):
    """prism-rag ingest --no-embedding skips Pass 3 entirely."""
    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()
    (vault / "a.md").write_text("# A\n\n[[B]]")
    (vault / "b.md").write_text("# B")

    env = os.environ.copy()
    env["PRISM_VAULT_PATH"] = str(vault)
    env["PRISM_DATA_DIR"] = str(data)
    env["PRISM_GEMINI_API_KEY"] = ""

    result = subprocess.run(
        _prism_rag_cmd() + [
            "ingest",
            "--vault", str(vault),
            "--output", str(data),
            "--no-embedding",
        ],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert (data / "graph.json").exists()
