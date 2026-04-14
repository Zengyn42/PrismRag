"""Tests for EmbeddingStore (LanceDB wrapper)."""
from __future__ import annotations

import pytest
from pathlib import Path

from prism_rag.store.embedding_store import EmbeddingStore


class TestEmbeddingStore:
    def test_upsert_and_get(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        vec = [0.1] * 768
        store.upsert("node_a", vec)
        result = store.get("node_a")
        assert result is not None
        assert len(result) == 768
        assert abs(result[0] - 0.1) < 1e-6

    def test_get_missing_returns_none(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        assert store.get("nonexistent") is None

    def test_delete(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        store.upsert("node_a", [0.1] * 768)
        store.delete("node_a")
        assert store.get("node_a") is None

    def test_upsert_overwrites(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        store.upsert("node_a", [0.1] * 768)
        store.upsert("node_a", [0.9] * 768)
        result = store.get("node_a")
        assert abs(result[0] - 0.9) < 1e-6

    def test_all_embeddings(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        store.upsert("a", [0.1] * 768)
        store.upsert("b", [0.2] * 768)
        all_vecs = store.all_embeddings()
        assert len(all_vecs) == 2
        assert "a" in all_vecs
        assert "b" in all_vecs

    def test_search(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        # Insert 3 vectors: a and b are similar, c is different
        store.upsert("a", [1.0, 0.0] + [0.0] * 766)
        store.upsert("b", [0.9, 0.1] + [0.0] * 766)
        store.upsert("c", [0.0, 1.0] + [0.0] * 766)
        results = store.search([1.0, 0.0] + [0.0] * 766, top_k=2)
        # a should be first (exact match), b second
        assert len(results) == 2
        assert results[0][0] == "a"
        assert results[1][0] == "b"

    def test_count(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        assert store.count() == 0
        store.upsert("a", [0.1] * 768)
        assert store.count() == 1
        store.upsert("b", [0.2] * 768)
        assert store.count() == 2

    def test_empty_store_all_embeddings(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        assert store.all_embeddings() == {}

    def test_empty_store_search(self, tmp_path):
        store = EmbeddingStore(tmp_path / "lance")
        results = store.search([0.1] * 768, top_k=5)
        assert results == []
