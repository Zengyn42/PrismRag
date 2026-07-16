"""Tests for embedding model configuration refactoring.

Covers:
  1. get_embedder() factory routes by backend
  2. OllamaEmbedder() bare construction → TypeError (model is required)
  3. OpenAICompatEmbedder sends correct /v1/embeddings request body
  4. Per-namespace embed_model override via get_embedder()
  5. Per-namespace fallback to global when embed_model is empty
  6. GraphSource new embed fields with correct defaults
  7. embed_meta.json written by compute_embeddings
  8. embed_meta.json mismatch → warning logged, no crash
  9. Missing embed_meta.json → no error at query time
  10. EmbedBackend Literal includes 'openai'
  11. PrismRagSettings openai_* fields exist
  12. embedding_dim resolves openai backend correctly
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prism_rag.config import GraphSource, PrismRagSettings


# ── Helpers ──────────────────────────────────────────────────────────────────


def _settings(**kwargs):
    """Build a PrismRagSettings with safe defaults for tests."""
    defaults = dict(
        vault_path=Path("/tmp/vault"),
        data_dir=Path("/tmp/data"),
        embed_backend="ollama",
        ollama_model="bge-m3",
        ollama_host="http://localhost:11434",
    )
    defaults.update(kwargs)
    return PrismRagSettings(**defaults)


def _graph_with_note(node_id="n1", content="hello world"):
    """Minimal KnowledgeGraph with one note node."""
    from prism_rag.store.graph import KnowledgeGraph, Node
    g = KnowledgeGraph()
    g.add_node(Node(id=node_id, label=node_id, kind="note", content=content, tokens=10))
    return g


# ── 1. get_embedder factory: ollama backend ──────────────────────────────────


def test_get_embedder_factory_ollama_backend():
    from prism_rag.ingest.embedder import get_embedder, OllamaEmbedder

    settings = _settings(embed_backend="ollama", ollama_model="bge-m3")
    embedder = get_embedder(settings)
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model == "bge-m3"


# ── 2. get_embedder factory: openai backend ──────────────────────────────────


def test_get_embedder_factory_openai_backend():
    from prism_rag.ingest.embedder import get_embedder, OpenAICompatEmbedder

    settings = _settings(
        embed_backend="openai",
        openai_base_url="http://localhost:1234",
        openai_api_key="test-key",
        openai_embed_model="text-embedding-3-small",
        openai_embed_dim=1536,
    )
    embedder = get_embedder(settings)
    assert isinstance(embedder, OpenAICompatEmbedder)
    assert embedder.model == "text-embedding-3-small"


# ── 3. OllamaEmbedder bare construction → TypeError ─────────────────────────


def test_ollama_embedder_requires_model():
    from prism_rag.ingest.embedder import OllamaEmbedder

    with pytest.raises(TypeError):
        OllamaEmbedder()  # model has no default


# ── 4. OpenAICompatEmbedder sends correct request body ───────────────────────


def test_openai_compat_embedder_request_body():
    from prism_rag.ingest.embedder import OpenAICompatEmbedder

    embedder = OpenAICompatEmbedder(
        model="my-embed-model",
        base_url="http://localhost:1234",
        api_key="sk-test",
    )

    captured_bodies = []
    fake_response_data = {
        "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]
    }
    fake_response = MagicMock()
    fake_response.__enter__ = lambda s: s
    fake_response.__exit__ = MagicMock(return_value=False)
    fake_response.read.return_value = json.dumps(fake_response_data).encode()

    def fake_urlopen(request, **kwargs):
        import urllib.request
        if isinstance(request, urllib.request.Request):
            captured_bodies.append(json.loads(request.data.decode()))
        return fake_response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = embedder.embed_query("hello world")

    assert len(captured_bodies) == 1
    body = captured_bodies[0]
    assert body["model"] == "my-embed-model"
    assert body["input"] == ["hello world"]
    assert result == [0.1, 0.2, 0.3]


def test_openai_compat_embedder_sends_auth_header():
    from prism_rag.ingest.embedder import OpenAICompatEmbedder

    embedder = OpenAICompatEmbedder(
        model="test-model",
        base_url="http://localhost:1234",
        api_key="sk-secret-123",
    )

    captured_headers = {}
    fake_response_data = {
        "data": [{"embedding": [0.1], "index": 0}]
    }
    fake_response = MagicMock()
    fake_response.__enter__ = lambda s: s
    fake_response.__exit__ = MagicMock(return_value=False)
    fake_response.read.return_value = json.dumps(fake_response_data).encode()

    def fake_urlopen(request, **kwargs):
        import urllib.request
        if isinstance(request, urllib.request.Request):
            captured_headers.update(dict(request.headers))
        return fake_response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        embedder.embed_query("test")

    assert "Authorization" in captured_headers
    assert captured_headers["Authorization"] == "Bearer sk-secret-123"


def test_openai_compat_embedder_batch():
    from prism_rag.ingest.embedder import OpenAICompatEmbedder

    embedder = OpenAICompatEmbedder(
        model="test-model",
        base_url="http://localhost:1234",
    )

    fake_response_data = {
        "data": [
            {"embedding": [0.1, 0.2], "index": 0},
            {"embedding": [0.3, 0.4], "index": 1},
        ]
    }
    fake_response = MagicMock()
    fake_response.__enter__ = lambda s: s
    fake_response.__exit__ = MagicMock(return_value=False)
    fake_response.read.return_value = json.dumps(fake_response_data).encode()

    with patch("urllib.request.urlopen", return_value=fake_response):
        result = embedder.embed_batch(["a", "b"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2]
    assert result[1] == [0.3, 0.4]


# ── 5. Per-namespace embed_model override ────────────────────────────────────


def test_get_embedder_per_namespace_override():
    from prism_rag.ingest.embedder import get_embedder, OllamaEmbedder

    ns1 = GraphSource(
        namespace="ns1",
        vault_path=Path("/tmp/vault1"),
        data_dir=Path("/tmp/data1"),
        embed_backend="ollama",
        embed_model="qwen3-embedding",
        embed_dim=4096,
    )
    settings = _settings(ollama_model="bge-m3", graphs=[ns1])

    embedder = get_embedder(settings, namespace="ns1")
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model == "qwen3-embedding"


# ── 6. Per-namespace fallback to global ──────────────────────────────────────


def test_get_embedder_per_namespace_fallback_to_global():
    from prism_rag.ingest.embedder import get_embedder, OllamaEmbedder

    ns2 = GraphSource(
        namespace="ns2",
        vault_path=Path("/tmp/vault2"),
        data_dir=Path("/tmp/data2"),
        # embed_model intentionally left empty
    )
    settings = _settings(ollama_model="snowflake-arctic-embed2", graphs=[ns2])

    embedder = get_embedder(settings, namespace="ns2")
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model == "snowflake-arctic-embed2"


# ── 7. GraphSource new embed fields defaults ─────────────────────────────────


def test_graphsource_embed_fields_defaults():
    gs = GraphSource(
        namespace="default",
        vault_path=Path("/tmp/v"),
        data_dir=Path("/tmp/d"),
    )
    assert gs.embed_backend == ""
    assert gs.embed_model == ""
    assert gs.embed_dim == 0


def test_graphsource_embed_fields_custom():
    gs = GraphSource(
        namespace="code",
        vault_path=Path("/tmp/v"),
        data_dir=Path("/tmp/d"),
        embed_backend="openai",
        embed_model="text-embedding-3-large",
        embed_dim=3072,
    )
    assert gs.embed_backend == "openai"
    assert gs.embed_model == "text-embedding-3-large"
    assert gs.embed_dim == 3072


# ── 8. embed_meta.json written by compute_embeddings ─────────────────────────


def test_embed_meta_written_after_compute_embeddings(tmp_path):
    from prism_rag.ingest.embedder import compute_embeddings

    graph = _graph_with_note()
    settings = _settings(
        embed_backend="ollama",
        ollama_model="bge-m3",
        data_dir=tmp_path,
    )

    fake_vec = [0.1] * 1024
    with patch(
        "prism_rag.ingest.embedder._compute_embeddings_ollama",
        return_value={"n1": fake_vec},
    ):
        compute_embeddings(graph, settings, cache_path=None)

    meta_path = tmp_path / "embed_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["backend"] == "ollama"
    assert meta["model"] == "bge-m3"
    assert isinstance(meta["dim"], int) and meta["dim"] > 0


def test_embed_meta_not_written_for_empty_graph(tmp_path):
    """No embed_meta.json when no nodes are embedded."""
    from prism_rag.ingest.embedder import compute_embeddings
    from prism_rag.store.graph import KnowledgeGraph

    graph = KnowledgeGraph()  # empty
    settings = _settings(data_dir=tmp_path)

    compute_embeddings(graph, settings)
    meta_path = tmp_path / "embed_meta.json"
    assert not meta_path.exists()


# ── 9. embed_meta.json mismatch → warning ────────────────────────────────────


def test_embed_meta_mismatch_logs_warning(tmp_path, caplog):
    from prism_rag.ingest.embedder import check_embed_consistency

    meta_path = tmp_path / "embed_meta.json"
    meta_path.write_text(json.dumps({
        "backend": "ollama",
        "model": "bge-m3",
        "dim": 1024,
    }))

    settings = _settings(
        embed_backend="ollama",
        ollama_model="qwen3-embedding:8b",
        data_dir=tmp_path,
    )

    with caplog.at_level(logging.WARNING):
        check_embed_consistency(settings, data_dir=tmp_path)

    warning_text = "\n".join(caplog.messages).lower()
    assert "mismatch" in warning_text
    assert "bge-m3" in warning_text


def test_embed_meta_mismatch_does_not_raise(tmp_path):
    """Mismatch must log but not raise."""
    from prism_rag.ingest.embedder import check_embed_consistency

    meta_path = tmp_path / "embed_meta.json"
    meta_path.write_text(json.dumps({
        "backend": "ollama",
        "model": "bge-m3",
        "dim": 1024,
    }))

    settings = _settings(
        embed_backend="openai",
        ollama_model="different-model",
        data_dir=tmp_path,
    )

    # Must not raise
    check_embed_consistency(settings, data_dir=tmp_path)


# ── 10. Missing embed_meta.json → no error ──────────────────────────────────


def test_embed_meta_missing_no_error(tmp_path):
    from prism_rag.ingest.embedder import check_embed_consistency

    settings = _settings(data_dir=tmp_path)
    # tmp_path has no embed_meta.json — must not raise
    check_embed_consistency(settings, data_dir=tmp_path)


# ── 11. EmbedBackend includes 'openai' ──────────────────────────────────────


def test_embed_backend_includes_openai():
    """PrismRagSettings must accept embed_backend='openai'."""
    settings = _settings(embed_backend="openai")
    assert settings.embed_backend == "openai"


# ── 12. PrismRagSettings openai fields ──────────────────────────────────────


def test_settings_openai_fields_exist():
    settings = _settings()
    assert hasattr(settings, "openai_base_url")
    assert hasattr(settings, "openai_api_key")
    assert hasattr(settings, "openai_embed_model")
    assert hasattr(settings, "openai_embed_dim")


def test_settings_embedding_dim_openai():
    settings = _settings(embed_backend="openai", openai_embed_dim=3072)
    assert settings.embedding_dim == 3072


def test_settings_get_embed_model_name():
    s_ollama = _settings(embed_backend="ollama", ollama_model="bge-m3")
    assert s_ollama.get_embed_model_name() == "bge-m3"

    s_openai = _settings(embed_backend="openai", openai_embed_model="text-embedding-3-large")
    assert s_openai.get_embed_model_name() == "text-embedding-3-large"


# ── 13. Old config backward compat ──────────────────────────────────────────


def test_old_graphsource_json_no_embed_fields():
    """GraphSource from JSON without embed fields works (PRISM_GRAPHS env var)."""
    old_json = json.dumps({
        "namespace": "nimbus",
        "vault_path": "/tmp/vault",
        "data_dir": "/tmp/data",
    })
    gs = GraphSource.model_validate_json(old_json)
    assert gs.embed_backend == ""
    assert gs.embed_model == ""
    assert gs.embed_dim == 0


def test_get_embedder_no_namespace_match():
    """get_embedder with unknown namespace falls back to global."""
    from prism_rag.ingest.embedder import get_embedder, OllamaEmbedder

    ns1 = GraphSource(
        namespace="ns1",
        vault_path=Path("/tmp/v"),
        data_dir=Path("/tmp/d"),
        embed_model="special-model",
    )
    settings = _settings(ollama_model="bge-m3", graphs=[ns1])

    # Request namespace "nonexistent" — should fall back to global bge-m3
    embedder = get_embedder(settings, namespace="nonexistent")
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model == "bge-m3"
