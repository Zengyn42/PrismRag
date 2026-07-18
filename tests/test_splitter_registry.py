"""Tests for prism_rag/ingest/splitters/registry.py."""

from __future__ import annotations

import pytest

from prism_rag.ingest.splitters.base import AtomicClaim, PassthroughSplitter, Splitter
from prism_rag.ingest.splitters.registry import (
    SPLITTER_REGISTRY,
    get_splitter,
    list_splitters,
    register_splitter,
)
from prism_rag.ingest.splitters.sentence import SentenceSplitter


# ── get_splitter ─────────────────────────────────────────────────────

def test_get_splitter_passthrough():
    s = get_splitter("passthrough")
    assert isinstance(s, PassthroughSplitter)
    assert s.name == "passthrough"


def test_get_splitter_sentence():
    s = get_splitter("sentence")
    assert isinstance(s, SentenceSplitter)
    assert s.name == "sentence"


def test_get_splitter_unknown_raises():
    with pytest.raises(ValueError, match="Unknown splitter"):
        get_splitter("nonexistent")


def test_get_splitter_error_lists_available():
    with pytest.raises(ValueError, match="passthrough"):
        get_splitter("bad")


# ── list_splitters ───────────────────────────────────────────────────

def test_list_splitters_sorted():
    names = list_splitters()
    assert names == sorted(names)
    assert "passthrough" in names
    assert "sentence" in names


# ── register_splitter ────────────────────────────────────────────────

def test_register_and_retrieve():
    class _Tmp(Splitter):
        @property
        def name(self) -> str:
            return "_reg_test_tmp"

        def split(self, section_text, *, doc_context=None):
            return [AtomicClaim(text=section_text)]

    try:
        register_splitter(_Tmp)
        s = get_splitter("_reg_test_tmp")
        assert isinstance(s, _Tmp)
    finally:
        SPLITTER_REGISTRY.pop("_reg_test_tmp", None)


def test_register_duplicate_raises():
    class _Dup(Splitter):
        @property
        def name(self) -> str:
            return "_reg_test_dup"

        def split(self, section_text, *, doc_context=None):
            return []

    try:
        register_splitter(_Dup)
        with pytest.raises(ValueError, match="already registered"):
            register_splitter(_Dup)
    finally:
        SPLITTER_REGISTRY.pop("_reg_test_dup", None)


def test_cleanup_no_leak():
    assert "_reg_test_tmp" not in SPLITTER_REGISTRY
    assert "_reg_test_dup" not in SPLITTER_REGISTRY


# ── package exports ──────────────────────────────────────────────────

def test_package_exports():
    from prism_rag.ingest import splitters

    assert hasattr(splitters, "get_splitter")
    assert hasattr(splitters, "list_splitters")
    assert hasattr(splitters, "register_splitter")
    assert hasattr(splitters, "SPLITTER_REGISTRY")
