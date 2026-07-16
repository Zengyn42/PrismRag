"""T1.7 — tests for OllamaEmbedder and compute_embeddings (Ollama backend).

Mocks urllib.request.urlopen so no real Ollama server is needed.
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prism_rag.config import PrismRagSettings
from prism_rag.ingest.embedder import (
    OllamaEmbedder,
    _get_embeddable_nodes,
    compute_embeddings,
    persist_embeddings,
)
from prism_rag.store.graph import Edge, KnowledgeGraph, Node


# ── Helpers ───────────────────────────────────────────────────────────────────

_DIM = 4


def _fake_response(embeddings: list[list[float]]) -> MagicMock:
    """Build a mock HTTP response that returns the given embeddings."""
    body = json.dumps({"embeddings": embeddings}).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _vec(n: int = _DIM) -> list[float]:
    return [0.1] * n


def _kg_with_notes(*ids) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for nid in ids:
        kg.add_node(Node(id=nid, label=nid, kind="note", content=f"content of {nid}"))
    return kg


# ── OllamaEmbedder.embed_query ────────────────────────────────────────────────


def test_embed_query_returns_vector():
    with patch("urllib.request.urlopen", return_value=_fake_response([_vec()])):
        emb = OllamaEmbedder(model="test-model")
        result = emb.embed_query("hello world")
    assert isinstance(result, list)
    assert result == _vec()


def test_embed_query_sends_correct_payload():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["payload"] = json.loads(req.data)
        return _fake_response([_vec()])

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        emb = OllamaEmbedder(model="bge-m3", base_url="http://localhost:11434")
        emb.embed_query("test text")

    assert captured["payload"]["model"] == "bge-m3"
    assert captured["payload"]["input"] == ["test text"]


def test_embed_query_uses_configured_host():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _fake_response([_vec()])

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        emb = OllamaEmbedder(model="bge-m3", base_url="http://myhost:9999")
        emb.embed_query("x")

    assert "myhost:9999" in captured["url"]


# ── OllamaEmbedder.embed_batch ────────────────────────────────────────────────


def test_embed_batch_returns_one_per_input():
    vecs = [_vec(), _vec(), _vec()]
    with patch("urllib.request.urlopen", return_value=_fake_response(vecs)):
        emb = OllamaEmbedder(model="bge-m3")
        result = emb.embed_batch(["a", "b", "c"])
    assert len(result) == 3
    assert result[0] == _vec()


def test_embed_batch_empty_input():
    emb = OllamaEmbedder(model="bge-m3")
    result = emb.embed_batch([])
    assert result == []


def test_embed_batch_count_mismatch_raises():
    with patch("urllib.request.urlopen", return_value=_fake_response([_vec()])):
        emb = OllamaEmbedder(model="bge-m3")
        with pytest.raises(ValueError, match="2 inputs"):
            emb.embed_batch(["a", "b"])


# ── _get_embeddable_nodes ─────────────────────────────────────────────────────


def test_get_embeddable_nodes_includes_note_and_knowledge():
    kg = KnowledgeGraph()
    kg.add_node(Node(id="n1", label="n1", kind="note", content="text"))
    kg.add_node(Node(id="n2", label="n2", kind="knowledge", content="text"))
    kg.add_node(Node(id="t1", label="t1", kind="tag", content="tag"))
    pairs = _get_embeddable_nodes(kg)
    ids = {p[0] for p in pairs}
    assert "n1" in ids
    assert "n2" in ids
    assert "t1" not in ids


def test_get_embeddable_nodes_skips_empty_content():
    kg = KnowledgeGraph()
    kg.add_node(Node(id="n1", label="n1", kind="note", content=""))
    kg.add_node(Node(id="n2", label="n2", kind="note", content="real content"))
    pairs = _get_embeddable_nodes(kg)
    ids = {p[0] for p in pairs}
    assert "n1" not in ids
    assert "n2" in ids


def test_get_embeddable_nodes_respects_embed_false():
    kg = KnowledgeGraph()
    kg.add_node(Node(id="n1", label="n1", kind="note", content="text",
                     frontmatter={"embed": False}))
    kg.add_node(Node(id="n2", label="n2", kind="note", content="text"))
    pairs = _get_embeddable_nodes(kg)
    ids = {p[0] for p in pairs}
    assert "n1" not in ids
    assert "n2" in ids


# ── compute_embeddings (ollama backend) ───────────────────────────────────────


def test_compute_embeddings_ollama_returns_dict():
    kg = _kg_with_notes("a", "b")
    settings = PrismRagSettings(embed_backend="ollama", ollama_model="bge-m3")

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data)
        # Return one vector per input text
        return _fake_response([_vec()] * len(payload["input"]))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = compute_embeddings(kg, settings)

    assert set(result.keys()) == {"a", "b"}
    assert len(result["a"]) == _DIM


def test_compute_embeddings_ollama_skips_failed_nodes(caplog):
    kg = _kg_with_notes("good", "bad")
    settings = PrismRagSettings(embed_backend="ollama", ollama_model="bge-m3")

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data)
        texts = payload["input"]
        # Batch fails when "bad" is included; single "good" call succeeds
        if any("bad" in t for t in texts):
            raise OSError("connection refused")
        return _fake_response([_vec()] * len(texts))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = compute_embeddings(kg, settings)

    # "good" succeeds via single fallback; "bad" fails and is skipped
    assert "good" in result
    assert "bad" not in result


def test_compute_embeddings_empty_graph():
    kg = KnowledgeGraph()
    settings = PrismRagSettings(embed_backend="ollama")
    result = compute_embeddings(kg, settings)
    assert result == {}


# ── persist_embeddings ────────────────────────────────────────────────────────


def test_persist_embeddings_writes_to_lance():
    vectors = {"node_a": [0.1, 0.2, 0.3, 0.4], "node_b": [0.5, 0.6, 0.7, 0.8]}
    with tempfile.TemporaryDirectory() as tmp:
        count = persist_embeddings(vectors, Path(tmp) / "lance", dim=4)
    assert count == 2


def test_persist_embeddings_empty_vectors():
    with tempfile.TemporaryDirectory() as tmp:
        count = persist_embeddings({}, Path(tmp) / "lance", dim=4)
    assert count == 0


def test_persist_embeddings_roundtrip():
    vectors = {"x": [1.0, 0.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0, 0.0]}
    with tempfile.TemporaryDirectory() as tmp:
        lance_path = Path(tmp) / "lance"
        persist_embeddings(vectors, lance_path, dim=4)

        from prism_rag.store.embedding_store import EmbeddingStore
        store = EmbeddingStore(lance_path, dim=4)
        assert store.count() == 2
        vec_x = store.get("x")
        assert vec_x is not None
        assert abs(vec_x[0] - 1.0) < 1e-5
