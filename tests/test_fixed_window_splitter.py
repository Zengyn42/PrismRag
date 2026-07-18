"""Tests for FixedWindowSplitter."""

from __future__ import annotations

import pytest

from prism_rag.ingest.splitters.fixed_window import FixedWindowSplitter


def _splitter(window_size=400, overlap=50):
    return FixedWindowSplitter(window_size=window_size, overlap=overlap)


def test_name():
    assert _splitter().name == "fixed_window"


def test_short_text_single_chunk():
    claims = _splitter(window_size=200, overlap=20).split("Hello world.")
    assert len(claims) == 1
    assert claims[0].text == "Hello world."


def test_long_text_multiple_chunks():
    text = "x" * 500
    claims = _splitter(window_size=100, overlap=10).split(text)
    assert len(claims) > 1
    for c in claims:
        assert len(c.text) <= 100


def test_overlap_content_matches():
    text = "A" * 100 + "B" * 100 + "C" * 100  # 300 chars
    claims = _splitter(window_size=120, overlap=30).split(text)
    assert len(claims) >= 2
    for i in range(len(claims) - 1):
        assert claims[i].text[-30:] == claims[i + 1].text[:30]


def test_overlap_ge_window_raises():
    with pytest.raises(ValueError):
        FixedWindowSplitter(window_size=100, overlap=150)


def test_overlap_eq_window_raises():
    with pytest.raises(ValueError):
        FixedWindowSplitter(window_size=100, overlap=100)


def test_empty_input():
    assert _splitter().split("") == []


def test_whitespace_only():
    assert _splitter().split("  \n\t  ") == []


def test_window_index_metadata():
    claims = _splitter(window_size=50, overlap=5).split("a" * 200)
    indices = [c.metadata["window_index"] for c in claims]
    assert indices == list(range(len(claims)))


def test_registry_integration():
    from prism_rag.ingest.splitters.registry import get_splitter, list_splitters

    assert "fixed_window" in list_splitters()
    s = get_splitter("fixed_window")
    assert isinstance(s, FixedWindowSplitter)


def test_package_export():
    from prism_rag.ingest.splitters import FixedWindowSplitter as Exported
    assert Exported is FixedWindowSplitter
