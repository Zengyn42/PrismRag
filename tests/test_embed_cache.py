# tests/test_embed_cache.py
from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prism_rag.ingest.embedder import (
    _load_embed_cache,
    _append_cache_entry,
    _compute_embeddings_ollama,
)
from prism_rag.store.graph import KnowledgeGraph, Node


def test_load_embed_cache_empty(tmp_path):
    cache_file = tmp_path / "embed_cache.jsonl"
    result = _load_embed_cache(cache_file)
    assert result == {}


def test_load_embed_cache_returns_last_wins_on_duplicate_node_id(tmp_path):
    cache_file = tmp_path / "embed_cache.jsonl"
    cache_file.write_text(
        '{"node_id": "n1", "sha": "sha1", "vec": [0.1]}\n'
        '{"node_id": "n1", "sha": "sha2", "vec": [0.9]}\n',
        encoding="utf-8",
    )
    result = _load_embed_cache(cache_file)
    assert result == {"n1": ("sha2", [0.9])}


def test_load_embed_cache_returns_node_id_to_sha_vec(tmp_path):
    cache_file = tmp_path / "embed_cache.jsonl"
    cache_file.write_text(
        '{"node_id": "n1", "sha": "abc", "vec": [0.1, 0.2]}\n'
        '{"node_id": "n2", "sha": "def", "vec": [0.3, 0.4]}\n',
        encoding="utf-8",
    )
    result = _load_embed_cache(cache_file)
    assert result["n1"] == ("abc", [0.1, 0.2])
    assert result["n2"] == ("def", [0.3, 0.4])


def test_load_embed_cache_skips_malformed_lines(tmp_path):
    cache_file = tmp_path / "embed_cache.jsonl"
    cache_file.write_text(
        'not-json\n'
        '{"node_id": "n1", "sha": "abc", "vec": [0.1]}\n',
        encoding="utf-8",
    )
    result = _load_embed_cache(cache_file)
    assert "n1" in result


def test_append_cache_entry_creates_file_if_missing(tmp_path):
    cache_file = tmp_path / "embed_cache.jsonl"
    _append_cache_entry(cache_file, "n1", "sha1", [0.1, 0.2])
    lines = cache_file.read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry == {"node_id": "n1", "sha": "sha1", "vec": [0.1, 0.2]}


def test_append_cache_entry_appends_not_overwrites(tmp_path):
    cache_file = tmp_path / "embed_cache.jsonl"
    _append_cache_entry(cache_file, "n1", "sha1", [0.1])
    _append_cache_entry(cache_file, "n2", "sha2", [0.2])
    lines = cache_file.read_text().strip().split("\n")
    assert len(lines) == 2


def test_append_cache_entry_concurrent_writes_safe(tmp_path):
    """Verify concurrent threads don't corrupt the file."""
    cache_file = tmp_path / "embed_cache.jsonl"
    DIM = 4096
    errors = []

    def write_entry(node_id: str):
        try:
            _append_cache_entry(cache_file, node_id, "sha", [0.1] * DIM)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write_entry, args=(f"n{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    lines = [l for l in cache_file.read_text().strip().split("\n") if l]
    assert len(lines) == 10
    for line in lines:
        entry = json.loads(line)
        assert len(entry["vec"]) == DIM


def _make_graph_with_node(node_id: str, content: str, sha: str) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    node = Node(id=node_id, label=node_id, kind="note", content=content, content_hash=sha)
    kg.add_node(node)
    return kg


def test_compute_embeddings_ollama_skips_cache_hits(tmp_path):
    cache_file = tmp_path / "embed_cache.jsonl"
    cached_vec = [0.42] * 4
    _append_cache_entry(cache_file, "n1", "sha:abc", cached_vec)

    kg = _make_graph_with_node("n1", "content", "sha:abc")

    with patch("prism_rag.ingest.embedder.OllamaEmbedder") as mock_cls:
        mock_embedder = MagicMock()
        mock_cls.return_value = mock_embedder
        result = _compute_embeddings_ollama(kg, settings=None, cache_path=cache_file)

    mock_embedder.embed_batch.assert_not_called()
    assert result["n1"] == cached_vec


def test_compute_embeddings_ollama_reembeds_on_sha_mismatch(tmp_path):
    cache_file = tmp_path / "embed_cache.jsonl"
    _append_cache_entry(cache_file, "n1", "sha:OLD", [0.1])

    kg = _make_graph_with_node("n1", "new content", "sha:NEW")
    new_vec = [0.99, 0.99]

    with patch("prism_rag.ingest.embedder.OllamaEmbedder") as mock_cls:
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [new_vec]
        mock_cls.return_value = mock_embedder
        result = _compute_embeddings_ollama(kg, settings=None, cache_path=cache_file)

    assert result["n1"] == new_vec
    lines = [json.loads(l) for l in cache_file.read_text().strip().split("\n") if l]
    shas = {l["sha"] for l in lines}
    assert "sha:NEW" in shas
