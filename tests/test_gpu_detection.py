# tests/test_gpu_detection.py
from __future__ import annotations

from unittest.mock import patch, MagicMock
import json

import pytest

from prism_rag.ingest.embedder import detect_model_device


def _mock_ps_response(models: list[dict]):
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = json.dumps({"models": models}).encode()
    return mock_resp


def test_detect_model_device_gpu(monkeypatch):
    resp = _mock_ps_response([{"name": "qwen3-embedding:8b", "size_vram": 8_000_000_000}])
    with patch("urllib.request.urlopen", return_value=resp):
        result = detect_model_device("qwen3-embedding:8b", "http://localhost:11434")
    assert result == "gpu"


def test_detect_model_device_cpu(monkeypatch):
    resp = _mock_ps_response([{"name": "qwen3-embedding:8b", "size_vram": 0}])
    with patch("urllib.request.urlopen", return_value=resp):
        result = detect_model_device("qwen3-embedding:8b", "http://localhost:11434")
    assert result == "cpu"


def test_detect_model_device_unknown_on_exception():
    with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
        result = detect_model_device("qwen3-embedding:8b", "http://localhost:11434")
    assert result == "unknown"


def test_detect_model_device_unknown_when_model_not_loaded():
    resp = _mock_ps_response([])
    with patch("urllib.request.urlopen", return_value=resp):
        result = detect_model_device("qwen3-embedding:8b", "http://localhost:11434")
    assert result == "unknown"
