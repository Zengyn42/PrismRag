"""
QA tests for SentenceSplitter (prism_rag/ingest/splitters/sentence.py).

Verified behaviors (10 tests):
  1. name property equals 'sentence'
  2. Basic English splitting by . ? !
  3. Chinese/mixed language splitting (。？！)
  4. Empty input → empty list
  5. Whitespace-only input → empty list
  6. Blank sentences within text are skipped (no empty-text claims)
  7. metadata['index'] is 0-based and consecutive across all claims
  8. Pure code block → single claim with metadata['is_code'] == True, text intact
  9. Code block mixed with surrounding prose → block intact, prose split normally
 10. Consecutive internal whitespace is collapsed in claim text
"""
from __future__ import annotations

import pytest


def _split(text: str, **kw):
    """Helper: import SentenceSplitter and split text."""
    from prism_rag.ingest.splitters.sentence import SentenceSplitter
    return SentenceSplitter().split(text, **kw)


# ---------------------------------------------------------------------------
# 1. name property
# ---------------------------------------------------------------------------

def test_sentence_splitter_name():
    from prism_rag.ingest.splitters.sentence import SentenceSplitter
    assert SentenceSplitter().name == "sentence"


# ---------------------------------------------------------------------------
# 2. Basic English sentence splitting
# ---------------------------------------------------------------------------

def test_english_sentence_splitting():
    text = "The cat sat on the mat. Dogs are loyal! Are cats independent?"
    claims = _split(text)
    assert len(claims) == 3, f"Expected 3 claims, got {len(claims)}: {[c.text for c in claims]}"
    assert "cat sat" in claims[0].text
    assert "loyal" in claims[1].text
    assert "independent" in claims[2].text


# ---------------------------------------------------------------------------
# 3. Chinese and mixed-language splitting
# ---------------------------------------------------------------------------

def test_chinese_and_mixed_splitting():
    text = "机器学习是人工智能的子领域。它使计算机能够从数据中学习！You can mix languages?"
    claims = _split(text)
    assert len(claims) == 3, f"Expected 3 claims, got {len(claims)}: {[c.text for c in claims]}"
    texts = [c.text for c in claims]
    assert any("机器学习" in t for t in texts)
    assert any("学习" in t and "数据" in t for t in texts)
    assert any("mix" in t for t in texts)


# ---------------------------------------------------------------------------
# 4. Empty input → empty list
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty_list():
    assert _split("") == []


# ---------------------------------------------------------------------------
# 5. Whitespace-only input → empty list
# ---------------------------------------------------------------------------

def test_whitespace_only_returns_empty_list():
    assert _split("   \n\n\t  ") == []


# ---------------------------------------------------------------------------
# 6. Blank / whitespace-only sentences are skipped
# ---------------------------------------------------------------------------

def test_blank_sentences_skipped():
    # Multiple punctuation in a row or trailing punctuation can produce blank candidates
    text = "First sentence. \n\n Second sentence! "
    claims = _split(text)
    for c in claims:
        assert c.text.strip(), f"Got empty-text claim: {c!r}"
    assert len(claims) == 2, f"Expected 2 non-blank claims, got {len(claims)}: {[c.text for c in claims]}"


# ---------------------------------------------------------------------------
# 7. metadata['index'] is 0-based and consecutive
# ---------------------------------------------------------------------------

def test_metadata_index_consecutive():
    text = "Alpha. Beta! Gamma? Delta."
    claims = _split(text)
    assert len(claims) == 4
    indices = [c.metadata.get("index") for c in claims]
    assert indices == list(range(4)), f"Expected [0,1,2,3] indices, got {indices}"


# ---------------------------------------------------------------------------
# 8. Pure code block → single claim, metadata['is_code'] == True
# ---------------------------------------------------------------------------

def test_pure_code_block_single_claim():
    text = "```python\ndef greet(name):\n    return f'Hello, {name}'\n```"
    claims = _split(text)
    assert len(claims) == 1, f"Code block must produce exactly 1 claim, got {len(claims)}"
    assert claims[0].metadata.get("is_code") is True, (
        f"Code block claim must have metadata['is_code']=True, got {claims[0].metadata}"
    )
    # Code content must be preserved intact (no sentence splitting inside)
    assert "def greet" in claims[0].text
    assert "return" in claims[0].text


# ---------------------------------------------------------------------------
# 9. Code block mixed with prose: block intact, prose split normally
# ---------------------------------------------------------------------------

def test_code_block_mixed_with_prose():
    text = (
        "Here is an example function.\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
        "The function adds two numbers. Use it wisely!"
    )
    claims = _split(text)

    code_claims = [c for c in claims if c.metadata.get("is_code")]
    prose_claims = [c for c in claims if not c.metadata.get("is_code")]

    assert len(code_claims) == 1, f"Expected 1 code claim, got {len(code_claims)}"
    assert "def add" in code_claims[0].text, "Code block content must be preserved"

    assert len(prose_claims) >= 2, (
        f"Expected ≥2 prose claims (2 sentences), got {len(prose_claims)}: "
        f"{[c.text for c in prose_claims]}"
    )
    prose_text = " ".join(c.text for c in prose_claims)
    assert "example function" in prose_text
    assert "adds two numbers" in prose_text


# ---------------------------------------------------------------------------
# 10. Consecutive internal whitespace collapsed in claim text
# ---------------------------------------------------------------------------

def test_consecutive_whitespace_collapsed():
    text = "This   has   extra   spaces.  So   does   this!"
    claims = _split(text)
    for c in claims:
        assert "  " not in c.text, (
            f"Claim text must have collapsed whitespace, got: {c.text!r}"
        )
