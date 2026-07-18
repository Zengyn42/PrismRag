"""LlmSplitter — pure-LLM knowledge atomization behind the Splitter interface.

This wraps "let the LLM read the section and propose knowledge atoms"
(previously implicit in each calling agent's conversation prompt) into a
versioned, reproducible, benchmarkable component that outputs list[Knot]
like every other splitter.

Backend resolution order:
  1. ``llm_fn`` constructor arg — any ``Callable[[str], str]`` (prompt → text).
     Use this to plug in Claude/Gemini/vLLM or a mock in tests.
  2. Default: local Ollama ``/api/generate`` (host/model via env
     PRISM_OLLAMA_HOST / PRISM_OLLAMA_LLM_MODEL).

The atomization prompt is versioned (PROMPT_VERSION) so benchmark results
can be tied to an exact prompt revision.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable

from prism_rag.ingest.splitters.base import Knot, Splitter

logger = logging.getLogger(__name__)

_OLLAMA_DEFAULT_HOST = "http://localhost:11434"
_OLLAMA_DEFAULT_MODEL = "qwen3.5:9b"

PROMPT_VERSION = "v1"

ATOMIZE_PROMPT_V1 = """\
You are a knowledge atomization engine. Split the source text into atomic \
knowledge units (KNOTs). Each KNOT must be:

1. SELF-CONTAINED — understandable without reading the source text. Resolve \
pronouns and implicit references ("it", "this approach", "the above") into \
explicit subjects.
2. ATOMIC — exactly one fact, decision, procedure, or concept per KNOT. If a \
sentence bundles several claims, split them.
3. FAITHFUL — no invention, no speculation, no summarizing-away of specifics \
(keep numbers, names, versions, paths exactly as written).
4. WORTH KEEPING — skip filler, greetings, section headers with no content, \
and meta-commentary.

Classify each KNOT's ontology_type as one of:
  concept   — definition or explanation of a term/idea
  fact      — a verifiable statement about the world/system
  decision  — a choice that was made, with or without rationale
  procedure — how to do something (steps, commands, configuration)

If a KNOT needs extra context to stand alone, put that context in \
context_note instead of bloating the body.

{doc_context_block}## Source text

{section_text}

## Output format

Return ONLY a JSON array (no markdown fences, no commentary):
[
  {{"title": "<short label, <=10 words>",
    "body": "<the self-contained atomic statement>",
    "ontology_type": "<concept|fact|decision|procedure>",
    "context_note": "<optional, empty string if not needed>"}}
]
Return [] if the text contains nothing worth keeping.
"""


def _default_ollama_generate(prompt: str, *, model: str, host: str, timeout: int) -> str:
    """Minimal stdlib Ollama /api/generate call (non-streaming)."""
    import urllib.request

    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "")


def _extract_json_array(text: str) -> list:
    """Parse the first JSON array found in *text*.

    Tolerates markdown fences and leading/trailing prose — LLMs do not
    reliably obey "return ONLY JSON".
    """
    # Fast path: whole output is the array
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    # Fenced block
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # First bracket-balanced array anywhere in the text
    start = text.find("[")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("[", start + 1)
    raise ValueError(f"No parseable JSON array in LLM output (head: {text[:200]!r})")


_VALID_ONTOLOGY_TYPES = {"concept", "fact", "decision", "procedure"}


class LlmSplitter(Splitter):
    """Pure-LLM atomization: section text → LLM → list[Knot].

    Args:
        llm_fn: ``Callable[[str], str]`` mapping a prompt to raw LLM output.
            When None, a local Ollama /api/generate backend is used.
        model: Ollama model name (ignored when llm_fn given). Defaults to
            env PRISM_OLLAMA_LLM_MODEL or "qwen3.5:9b".
        host: Ollama host (ignored when llm_fn given). Defaults to env
            PRISM_OLLAMA_HOST or "http://localhost:11434".
        timeout: per-call timeout in seconds for the default backend.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str] | None = None,
        *,
        model: str | None = None,
        host: str | None = None,
        timeout: int = 300,
    ):
        self._model = model or os.environ.get("PRISM_OLLAMA_LLM_MODEL", _OLLAMA_DEFAULT_MODEL)
        self._host = host or os.environ.get("PRISM_OLLAMA_HOST", _OLLAMA_DEFAULT_HOST)
        self._timeout = timeout
        if llm_fn is not None:
            self._llm_fn = llm_fn
            self._backend = "custom"
        else:
            self._llm_fn = lambda prompt: _default_ollama_generate(
                prompt, model=self._model, host=self._host, timeout=self._timeout
            )
            self._backend = f"ollama/{self._model}"

    @property
    def name(self) -> str:
        return "llm"

    def split(
        self,
        section_text: str,
        *,
        doc_context: str | None = None,
    ) -> list[Knot]:
        if not section_text.strip():
            return []

        doc_context_block = (
            f"## Document context (for disambiguation only — do NOT atomize it)\n\n"
            f"{doc_context}\n\n"
            if doc_context
            else ""
        )
        prompt = ATOMIZE_PROMPT_V1.format(
            doc_context_block=doc_context_block, section_text=section_text
        )
        raw = self._llm_fn(prompt)
        items = _extract_json_array(raw)

        knots: list[Knot] = []
        for item in items:
            if not isinstance(item, dict):
                logger.warning("[LlmSplitter] skipping non-dict item: %r", item)
                continue
            body = str(item.get("body", "")).strip()
            if not body:
                continue
            otype = str(item.get("ontology_type", "fact")).strip().lower()
            if otype not in _VALID_ONTOLOGY_TYPES:
                logger.warning(
                    "[LlmSplitter] invalid ontology_type %r → 'fact'", otype
                )
                otype = "fact"
            note = str(item.get("context_note", "") or "").strip() or None
            knots.append(
                Knot(
                    text=body,
                    title=str(item.get("title", "")).strip(),
                    ontology_type=otype,
                    context_note=note,
                    method="llm",
                    metadata={
                        "backend": self._backend,
                        "prompt_version": PROMPT_VERSION,
                    },
                )
            )
        return knots
