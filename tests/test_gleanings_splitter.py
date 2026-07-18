"""Tests for GleaningsSplitter and the new Knot status/payload fields."""

from __future__ import annotations

import json

import pytest

from prism_rag.ingest.splitters import (
    GleaningsSplitter,
    Knot,
    get_splitter,
    list_splitters,
)

_BASE = [{"title": "a", "body": "atom A", "ontology_type": "fact", "context_note": ""}]
_GLEANED = [{"title": "b", "body": "atom B (missed)", "ontology_type": "decision", "context_note": ""}]


class _SeqLlm:
    """llm_fn returning canned responses in sequence."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else "[]"


# ── Knot status / payload ────────────────────────────────────────────────────

def test_knot_default_status_confirmed():
    k = Knot(text="x")
    assert k.status == "confirmed"
    assert k.payload is None


def test_knot_valid_statuses():
    for s in ("confirmed", "suspected", "superseded"):
        assert Knot(text="x", status=s).status == s


def test_knot_invalid_status_raises():
    with pytest.raises(ValueError, match="Invalid Knot status"):
        Knot(text="x", status="banana")


def test_knot_payload_roundtrip():
    k = Knot(text="run pytest", ontology_type="procedure",
             payload={"commands": ["python3 -m pytest"]})
    assert k.payload["commands"] == ["python3 -m pytest"]


# ── GleaningsSplitter ────────────────────────────────────────────────────────

def test_name_and_registry():
    assert "llm_gleanings" in list_splitters()
    s = get_splitter("llm_gleanings", llm_fn=lambda p: "[]")
    assert isinstance(s, GleaningsSplitter)
    assert s.name == "llm_gleanings"


def test_base_plus_one_gleaning_round():
    llm = _SeqLlm([json.dumps(_BASE), json.dumps(_GLEANED), "[]"])
    s = GleaningsSplitter(llm_fn=llm, max_gleanings=2)
    knots = s.split("source text")
    assert [k.text for k in knots] == ["atom A", "atom B (missed)"]
    assert all(k.method == "llm_gleanings" for k in knots)
    assert knots[1].metadata["gleaning_round"] == 1
    # 3 calls: base + round1 (found) + round2 (empty → stop)
    assert len(llm.prompts) == 3
    # gleaning prompt must contain source and already-extracted atoms
    assert "source text" in llm.prompts[1]
    assert "atom A" in llm.prompts[1]


def test_stops_early_when_round_empty():
    llm = _SeqLlm([json.dumps(_BASE), "[]"])
    s = GleaningsSplitter(llm_fn=llm, max_gleanings=5)
    knots = s.split("text")
    assert len(knots) == 1
    assert len(llm.prompts) == 2  # base + 1 round, no further rounds


def test_exact_duplicate_from_gleaning_dropped():
    dup = [{"title": "a2", "body": "atom A", "ontology_type": "fact"}]
    llm = _SeqLlm([json.dumps(_BASE), json.dumps(dup)])
    knots = GleaningsSplitter(llm_fn=llm, max_gleanings=3).split("text")
    assert len(knots) == 1  # duplicate text not re-added; loop stopped (no new)


def test_max_gleanings_zero_equals_plain_llm():
    llm = _SeqLlm([json.dumps(_BASE)])
    knots = GleaningsSplitter(llm_fn=llm, max_gleanings=0).split("text")
    assert len(knots) == 1
    assert len(llm.prompts) == 1


def test_unparseable_gleaning_round_stops_gracefully():
    llm = _SeqLlm([json.dumps(_BASE), "no json here at all"])
    knots = GleaningsSplitter(llm_fn=llm, max_gleanings=3).split("text")
    assert len(knots) == 1  # base result kept, no crash


def test_empty_base_skips_gleaning():
    llm = _SeqLlm(["[]"])
    knots = GleaningsSplitter(llm_fn=llm, max_gleanings=3).split("text")
    assert knots == []
    assert len(llm.prompts) == 1


def test_negative_max_gleanings_raises():
    with pytest.raises(ValueError):
        GleaningsSplitter(llm_fn=lambda p: "[]", max_gleanings=-1)
