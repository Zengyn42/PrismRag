#!/usr/bin/env python3
"""Run LLM splitter benchmark with multiple prompt strategies.

Compares:
  1. V1 — current ATOMIZE_PROMPT_V1 (baseline)
  2. V2-decontext — adds explicit decontextualization + compound-sentence guard
  3. V2-propositions — Dense X Propositionizer-style prompt (proposition focused)
  4. sentence — rule-based sentence splitter (reference)

Against the Propositionizer-wiki-data gold standard (test split).
"""

import json
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism_rag.ingest.splitters.base import Knot, Splitter
from prism_rag.ingest.splitters.llm import LlmSplitter, _extract_json_array, _default_ollama_generate
from prism_rag.ingest.splitters.benchmark.propositionizer import load_propositionizer_dataset
from prism_rag.ingest.splitters.benchmark.scoring import score_splitter, SplitterScore
from prism_rag.ingest.splitters.benchmark.harness import format_report, BenchmarkReport
from prism_rag.ingest.splitters.registry import get_splitter
from datetime import datetime, timezone

# ── Prompt variants ──────────────────────────────────────────────────────────

PROMPT_V2_DECONTEXT = """\
You are a knowledge atomization engine. Your task has TWO phases:

## Phase 1: Decontextualize
For each sentence in the source text, resolve ALL:
- Pronouns ("it", "this", "they", "he", "she") → the actual named entity
- Implicit references ("the system", "the above approach") → explicit name
- Relative time ("last year", "recently") → absolute if available

## Phase 2: Atomize
Split the decontextualized text into atomic knowledge units. Each unit must be:

1. SELF-CONTAINED — understandable without the source text. Every statement \
must name its subject explicitly.
2. ATOMIC — exactly ONE fact per unit. If a sentence contains "and", "but", \
"while", or "although" joining two independent clauses, split them into \
separate units.
3. FAITHFUL — no invention. Keep numbers, names, versions exactly as written.
4. WORTH KEEPING — skip filler and meta-commentary.

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

PROMPT_V2_PROPOSITIONS = """\
Decompose the following text into clear, simple propositions. A proposition is:
- A single, atomic fact or assertion
- Self-contained — understandable without reading the original text
- Minimal — as concise as possible while remaining unambiguous

Rules:
1. Split compound sentences into simple ones
2. Resolve ALL coreferences — replace pronouns and implicit references with \
the actual entity names (e.g., "it" → "the Leaning Tower of Pisa")
3. Maintain the original meaning and phrasing as closely as possible
4. Keep specific numbers, dates, names, and measurements exactly as written
5. Each proposition should be one sentence

{doc_context_block}## Source text

{section_text}

## Output format

Return ONLY a JSON array (no markdown fences, no commentary):
[
  {{"title": "<2-5 word label>",
    "body": "<the self-contained proposition>",
    "ontology_type": "fact",
    "context_note": ""}}
]
Return [] if the text contains nothing worth keeping.
"""

PROMPT_V2_MOLECULAR = """\
You are a molecular fact extractor. Extract molecular facts from the source \
text. A molecular fact is the smallest unit of information that is STILL \
UNAMBIGUOUS when read alone.

## Process
1. For each sentence, identify what could be ambiguous if read standalone \
(who is "he"? which "system"? what "approach"?)
2. Add the MINIMUM context needed to resolve each ambiguity
3. Split compound claims into separate molecular facts
4. Drop filler and meta-commentary

Each molecular fact must:
- Name its subject explicitly (never use pronouns)
- Contain exactly ONE claim
- Be faithful to the source (no invention, keep specifics)

{doc_context_block}## Source text

{section_text}

## Output format

Return ONLY a JSON array (no markdown fences, no commentary):
[
  {{"title": "<short label>",
    "body": "<the molecular fact>",
    "ontology_type": "fact",
    "context_note": ""}}
]
Return [] if nothing worth keeping.
"""


# ── Custom splitter wrappers ─────────────────────────────────────────────────

class PromptVariantSplitter(Splitter):
    """LLM splitter with a custom prompt template."""

    def __init__(self, name: str, prompt_template: str, model: str = "holo35b"):
        self._name = name
        self._prompt_template = prompt_template
        self._model = model
        self._host = "http://localhost:11434"

    @property
    def name(self) -> str:
        return self._name

    def split(self, section_text: str, *, doc_context: str | None = None) -> list[Knot]:
        if not section_text.strip():
            return []

        doc_context_block = (
            f"## Document context (for disambiguation only)\n\n{doc_context}\n\n"
            if doc_context else ""
        )
        prompt = self._prompt_template.format(
            doc_context_block=doc_context_block,
            section_text=section_text,
        )

        raw = _default_ollama_generate(
            prompt, model=self._model, host=self._host, timeout=120
        )

        # Strip qwen3 thinking tags if present
        import re as _re
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()

        try:
            items = _extract_json_array(raw)
        except ValueError:
            return []

        knots = []
        for item in items:
            if not isinstance(item, dict):
                continue
            body = str(item.get("body", "")).strip()
            if not body:
                continue
            knots.append(Knot(text=body, title=str(item.get("title", "")), method=self._name))
        return knots


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM Splitter Benchmark")
    parser.add_argument("--cases", type=int, default=100, help="Number of cases to evaluate")
    parser.add_argument("--model", type=str, default="holo35b", help="Ollama model name")
    args = parser.parse_args()

    print(f"Loading Propositionizer dataset (test split, max {args.cases} cases)...")
    cases = load_propositionizer_dataset(split="test", max_cases=args.cases)
    print(f"Loaded {len(cases)} cases\n")

    # Define all splitter variants
    splitters = [
        get_splitter("sentence"),  # rule-based reference
        LlmSplitter(model=args.model, host="http://localhost:11434"),  # V1 baseline
        PromptVariantSplitter("v2_decontext", PROMPT_V2_DECONTEXT, model=args.model),
        PromptVariantSplitter("v2_propositions", PROMPT_V2_PROPOSITIONS, model=args.model),
        PromptVariantSplitter("v2_molecular", PROMPT_V2_MOLECULAR, model=args.model),
    ]

    scores = []
    for splitter in splitters:
        name = splitter.name
        print(f"Running: {name} ...", end=" ", flush=True)
        t0 = time.time()
        score = score_splitter(splitter, cases)
        elapsed = time.time() - t0
        scores.append(score)
        print(f"done ({elapsed:.1f}s) — overall={score.overall:.4f}, gold_f1={score.gold_alignment:.4f}")

    report = BenchmarkReport(
        scores=scores,
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_size=len(cases),
        dataset_sources=[c.source for c in cases[:5]],
    )

    print("\n" + format_report(report))

    # Save to file
    out_path = os.path.join(os.path.dirname(__file__), "..", "benchmark_results.md")
    with open(out_path, "w") as f:
        f.write(format_report(report))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
