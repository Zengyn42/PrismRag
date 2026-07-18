"""Tests for LlmSplitter (prism_rag/ingest/splitters/llm.py).

All tests use an injected mock llm_fn — no Ollama required.
"""

from __future__ import annotations

import json

import pytest

from prism_rag.ingest.splitters import Knot, LlmSplitter, get_splitter, list_splitters
from prism_rag.ingest.splitters.llm import PROMPT_VERSION, _extract_json_array

_GOOD_ITEMS = [
    {
        "title": "checkpointer async setup",
        "body": "LangGraph 的 SqliteSaver checkpointer 需要在使用前执行 async setup。",
        "ontology_type": "fact",
        "context_note": "",
    },
    {
        "title": "use systemd restart",
        "body": "Hani 服务重启必须使用 systemctl --user restart hani，禁止手动 nohup。",
        "ontology_type": "procedure",
        "context_note": "决策记录于 feedback_hani_restart.md",
    },
]


def _mk(response: str) -> LlmSplitter:
    return LlmSplitter(llm_fn=lambda prompt: response)


def test_name():
    assert _mk("[]").name == "llm"


def test_basic_split_two_knots():
    s = _mk(json.dumps(_GOOD_ITEMS))
    knots = s.split("some section text")
    assert len(knots) == 2
    assert all(isinstance(k, Knot) for k in knots)
    assert knots[0].title == "checkpointer async setup"
    assert knots[0].ontology_type == "fact"
    assert knots[0].context_note is None  # empty string → None
    assert knots[1].ontology_type == "procedure"
    assert knots[1].context_note == "决策记录于 feedback_hani_restart.md"


def test_method_and_metadata():
    knots = _mk(json.dumps(_GOOD_ITEMS)).split("text")
    assert knots[0].method == "llm"
    assert knots[0].metadata["backend"] == "custom"
    assert knots[0].metadata["prompt_version"] == PROMPT_VERSION


def test_empty_input_returns_empty_no_llm_call():
    calls = []

    def fn(prompt):
        calls.append(prompt)
        return "[]"

    s = LlmSplitter(llm_fn=fn)
    assert s.split("") == []
    assert s.split("   \n  ") == []
    assert calls == []  # LLM must not be called for empty input


def test_empty_array_response():
    assert _mk("[]").split("greetings only") == []


def test_markdown_fenced_response():
    fenced = "Here you go:\n```json\n" + json.dumps(_GOOD_ITEMS) + "\n```\nDone."
    knots = _mk(fenced).split("text")
    assert len(knots) == 2


def test_prose_wrapped_array():
    wrapped = "Sure! The atoms are: " + json.dumps(_GOOD_ITEMS) + " — hope this helps."
    knots = _mk(wrapped).split("text")
    assert len(knots) == 2


def test_invalid_ontology_type_coerced_to_fact():
    items = [{"title": "t", "body": "b", "ontology_type": "banana"}]
    knots = _mk(json.dumps(items)).split("text")
    assert knots[0].ontology_type == "fact"


def test_items_without_body_skipped():
    items = [{"title": "no body"}, {"title": "ok", "body": "real content"}]
    knots = _mk(json.dumps(items)).split("text")
    assert len(knots) == 1
    assert knots[0].text == "real content"


def test_unparseable_output_raises():
    with pytest.raises(ValueError, match="No parseable JSON array"):
        _mk("I refuse to answer in JSON.").split("text")


def test_doc_context_injected_into_prompt():
    seen = {}

    def fn(prompt):
        seen["prompt"] = prompt
        return "[]"

    LlmSplitter(llm_fn=fn).split("body text", doc_context="Doc title: PrismRag v5.6")
    assert "Doc title: PrismRag v5.6" in seen["prompt"]
    assert "body text" in seen["prompt"]


def test_registry_integration():
    assert "llm" in list_splitters()
    s = get_splitter("llm", llm_fn=lambda p: "[]")
    assert isinstance(s, LlmSplitter)


def test_extract_json_array_nested():
    text = 'noise [1, [2, 3], {"a": [4]}] trailing'
    assert _extract_json_array(text) == [1, [2, 3], {"a": [4]}]
