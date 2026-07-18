"""Tests for ParagraphSplitter."""

from __future__ import annotations

from prism_rag.ingest.splitters.paragraph import ParagraphSplitter


def _splitter() -> ParagraphSplitter:
    return ParagraphSplitter()


# ── name property ────────────────────────────────────────────────────

def test_name():
    assert _splitter().name == "paragraph"


# ── normal multi-paragraph splitting ─────────────────────────────────

def test_multiple_paragraphs():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    claims = _splitter().split(text)
    assert len(claims) == 3
    assert claims[0].text == "First paragraph."
    assert claims[1].text == "Second paragraph."
    assert claims[2].text == "Third paragraph."


# ── single paragraph (no blank lines) ───────────────────────────────

def test_single_paragraph():
    text = "Just one paragraph with multiple lines.\nStill the same paragraph."
    claims = _splitter().split(text)
    assert len(claims) == 1
    assert "Just one paragraph" in claims[0].text
    assert "Still the same paragraph" in claims[0].text


# ── empty / whitespace-only input ────────────────────────────────────

def test_empty_input():
    assert _splitter().split("") == []


def test_whitespace_only():
    assert _splitter().split("   \n\n  \t  \n") == []


# ── excessive blank lines collapsed ─────────────────────────────────

def test_multiple_blank_lines():
    text = "Para A.\n\n\n\n\nPara B."
    claims = _splitter().split(text)
    assert len(claims) == 2
    assert claims[0].text == "Para A."
    assert claims[1].text == "Para B."


# ── CRLF line endings ───────────────────────────────────────────────

def test_crlf_line_endings():
    text = "Windows para 1.\r\n\r\nWindows para 2.\r\n\r\nWindows para 3."
    claims = _splitter().split(text)
    assert len(claims) == 3
    assert claims[0].text == "Windows para 1."
    assert claims[2].text == "Windows para 3."


def test_mixed_line_endings():
    text = "Mixed A.\r\n\r\nMixed B.\n\nMixed C."
    claims = _splitter().split(text)
    assert len(claims) == 3


# ── leading/trailing whitespace stripped ─────────────────────────────

def test_leading_trailing_whitespace():
    text = "\n\n  Hello world.  \n\n  Goodbye.  \n\n"
    claims = _splitter().split(text)
    assert len(claims) == 2
    assert claims[0].text == "Hello world."
    assert claims[1].text == "Goodbye."


# ── AtomicClaim defaults ────────────────────────────────────────────

def test_claim_defaults():
    claims = _splitter().split("One paragraph.")
    c = claims[0]
    assert c.source_section_id is None
    assert c.context_note is None
    assert c.metadata == {}


# ── registry integration ────────────────────────────────────────────

def test_registry_has_paragraph():
    from prism_rag.ingest.splitters.registry import get_splitter, list_splitters

    assert "paragraph" in list_splitters()
    s = get_splitter("paragraph")
    assert isinstance(s, ParagraphSplitter)


# ── package export ──────────────────────────────────────────────────

def test_package_export():
    from prism_rag.ingest.splitters import ParagraphSplitter as Exported

    assert Exported is ParagraphSplitter
