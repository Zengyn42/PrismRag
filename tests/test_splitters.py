"""Tests for the prism_rag.ingest.splitters interface and reference implementation."""

from __future__ import annotations

import pytest

from prism_rag.ingest.splitters.base import AtomicClaim, PassthroughSplitter, Splitter


# ── AtomicClaim construction ────────────────────────────────────────────────


def test_atomic_claim_defaults():
    claim = AtomicClaim(text="Python uses indentation for blocks.")
    assert claim.text == "Python uses indentation for blocks."
    assert claim.source_section_id is None
    assert claim.context_note is None
    assert claim.metadata == {}


def test_atomic_claim_all_fields():
    claim = AtomicClaim(
        text="LanceDB stores embeddings.",
        source_section_id="sec-3",
        context_note="In the context of PrismRag's embedding persistence layer",
        metadata={"confidence": 0.95, "method": "rule-based"},
    )
    assert claim.source_section_id == "sec-3"
    assert claim.context_note.startswith("In the context")
    assert claim.metadata["confidence"] == 0.95


def test_atomic_claim_metadata_independence():
    """Each claim must get its own metadata dict (no shared mutable default)."""
    a = AtomicClaim(text="one")
    b = AtomicClaim(text="two")
    a.metadata["key"] = "val"
    assert "key" not in b.metadata


# ── Splitter ABC ─────────────────────────────────────────────────────────────


def test_splitter_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Splitter()  # type: ignore[abstract]


def test_splitter_is_abstract_base_class():
    assert issubclass(Splitter, Splitter)
    # Incomplete subclass should also fail
    class Incomplete(Splitter):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


# ── PassthroughSplitter ─────────────────────────────────────────────────────


def test_passthrough_name():
    s = PassthroughSplitter()
    assert s.name == "passthrough"


def test_passthrough_returns_single_claim():
    s = PassthroughSplitter()
    result = s.split("Leiden clustering detects communities.")
    assert len(result) == 1
    assert result[0].text == "Leiden clustering detects communities."


def test_passthrough_preserves_full_text():
    text = "Line 1.\nLine 2.\nLine 3."
    s = PassthroughSplitter()
    result = s.split(text)
    assert len(result) == 1
    assert result[0].text == text


def test_passthrough_empty_input():
    s = PassthroughSplitter()
    assert s.split("") == []
    assert s.split("   ") == []
    assert s.split("\n\t") == []


def test_passthrough_ignores_doc_context():
    s = PassthroughSplitter()
    result = s.split("claim text", doc_context="some document title")
    assert len(result) == 1
    assert result[0].text == "claim text"


def test_passthrough_is_splitter_subclass():
    assert issubclass(PassthroughSplitter, Splitter)
    assert isinstance(PassthroughSplitter(), Splitter)


# ── Package import ───────────────────────────────────────────────────────────


def test_package_exports():
    from prism_rag.ingest.splitters import AtomicClaim, PassthroughSplitter, Splitter
    assert AtomicClaim is not None
    assert Splitter is not None
    assert PassthroughSplitter is not None
