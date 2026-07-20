#!/usr/bin/env python3
"""Multi-granularity retrieval benchmark.

Builds a 3-layer index (L0/L1/L2) from Propositionizer dataset texts,
generates QA pairs, and compares 5 retrieval strategies:

  flat_l0      -- L0 cosine search (baseline)
  flat_l1      -- L1 group cosine search
  parent_l0    -- L0 match, return parent L1 text
  multi_layer  -- L2 tag filter -> L1 cosine within scope
  collapsed    -- all L0+L1 vectors flat-searched (RAPTOR style)

Usage::

    OLLAMA_HOST=http://localhost:11434 python3 scripts/run_multi_granularity_eval.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prism_rag.ingest.splitters.base import Knot, Splitter
from prism_rag.ingest.splitters.benchmark.downstream import (
    QAPair,
    generate_qa_pairs,
    ollama_chat,
    ollama_embed,
    word_overlap,
    cosine_similarity,
    score_context_sufficiency,
    score_iou,
    score_boundary_clarity,
)
from prism_rag.ingest.splitters.benchmark.multi_granularity import (
    MultiGranularityIndex,
    build_multi_granularity_index,
    retrieve_flat_l0,
    retrieve_flat_l1,
    retrieve_parent_l0,
    retrieve_multi_layer,
    retrieve_collapsed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── V2 Propositions splitter (inline, same as run_llm_benchmark.py) ──────────

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

## Source text

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


class V2PropositionsSplitter(Splitter):
    """LLM splitter using v2_propositions prompt."""

    def __init__(self, llm_fn=None):
        self._llm_fn = llm_fn or ollama_chat

    @property
    def name(self) -> str:
        return "v2_propositions"

    def split(self, section_text: str, *, doc_context: str | None = None) -> list[Knot]:
        import json
        import re

        if not section_text.strip():
            return []

        prompt = PROMPT_V2_PROPOSITIONS.format(section_text=section_text)
        raw = self._llm_fn(prompt)

        # Strip thinking tags
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Parse JSON array
        try:
            # Try whole text
            text = raw.strip()
            items = None
            if text.startswith("["):
                try:
                    items = json.loads(text)
                except json.JSONDecodeError:
                    pass
            if items is None:
                match = re.search(r"\[.*\]", text, re.DOTALL)
                if match:
                    items = json.loads(match.group())
                else:
                    items = []
        except (json.JSONDecodeError, ValueError):
            items = []

        knots = []
        for item in items:
            if not isinstance(item, dict):
                continue
            body = str(item.get("body", "")).strip()
            if not body:
                continue
            knots.append(Knot(
                text=body,
                title=str(item.get("title", "")),
                method="v2_propositions",
            ))
        return knots


# ── Evaluation helpers ───────────────────────────────────────────────────────

def score_retrieval_strategy(
    strategy_name: str,
    retrieve_fn,
    qa_pairs: list[QAPair],
    index: MultiGranularityIndex,
    embedder_fn,
    llm_fn,
    top_k: int = 5,
) -> dict:
    """Score a retrieval strategy on QA pairs.

    Returns dict with: recall, mrr, iou, sufficiency, boundary.
    """
    questions = [qa.question for qa in qa_pairs]
    q_vecs = embedder_fn(questions)

    recall_hits = 0
    mrr_sum = 0.0
    iou_sum = 0.0

    for i, qa in enumerate(qa_pairs):
        # Retrieve chunks
        if strategy_name == "multi_layer":
            retrieved_texts = retrieve_fn(q_vecs[i], qa.question, index, top_k)
        else:
            retrieved_texts = retrieve_fn(q_vecs[i], index, top_k)

        # Context recall: does any retrieved chunk contain the answer?
        answer_found = any(
            word_overlap(chunk, qa.answer) >= 0.5 for chunk in retrieved_texts
        )
        if answer_found:
            recall_hits += 1

        # MRR: reciprocal rank of first relevant chunk
        for rank, chunk in enumerate(retrieved_texts, 1):
            if word_overlap(chunk, qa.source_text) >= 0.3:
                mrr_sum += 1.0 / rank
                break

        # IoU: overlap of retrieved vs relevant
        # relevant = source-text overlapping chunks from all L0+L1
        all_texts = index.l0_texts + index.l1_texts
        relevant_set = {
            t for t in all_texts
            if word_overlap(t, qa.source_text) >= 0.3
        }
        retrieved_set = set(retrieved_texts)
        intersection = retrieved_set & relevant_set
        union = retrieved_set | relevant_set
        if union:
            iou_sum += len(intersection) / len(union)

    n = len(qa_pairs)
    recall = recall_hits / n if n else 0.0
    mrr = mrr_sum / n if n else 0.0
    iou = iou_sum / n if n else 0.0

    # Boundary clarity: use L1 vectors (most strategies return L1-level chunks)
    boundary = score_boundary_clarity(index.l1_vectors)

    # Context sufficiency (LLM-as-judge) — sample first 10 for speed
    sufficiency = 0.0
    sample_n = min(10, n)
    if sample_n > 0:
        correct = 0
        for i in range(sample_n):
            qa = qa_pairs[i]
            if strategy_name == "multi_layer":
                retrieved = retrieve_fn(q_vecs[i], qa.question, index, top_k)
            else:
                retrieved = retrieve_fn(q_vecs[i], index, top_k)
            context = "\n\n".join(retrieved)
            prompt = (
                f"Given ONLY the following context (retrieved chunks), can you answer this question?\n"
                f"Question: {qa.question}\nContext: {context}\n\n"
                f'Answer with JSON: {{"answerable": true/false, "confidence": 0.0-1.0, "answer": "..."}}'
            )
            try:
                import json
                import re
                response = llm_fn(prompt)
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    if parsed.get("answerable"):
                        llm_answer = parsed.get("answer", "")
                        if llm_answer and word_overlap(llm_answer, qa.answer) >= 0.5:
                            correct += 1
            except Exception:
                pass
        sufficiency = correct / sample_n

    return {
        "strategy": strategy_name,
        "recall": recall,
        "mrr": mrr,
        "iou": iou,
        "sufficiency": sufficiency,
        "boundary": boundary,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-granularity retrieval benchmark")
    parser.add_argument("--texts", type=int, default=50, help="Number of texts to load")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval")
    parser.add_argument("--n-qa", type=int, default=3, help="QA pairs per text")
    parser.add_argument("--embed-model", default="qwen3-embedding:8b")
    parser.add_argument("--llm-model", default="gemma4:e4b")
    args = parser.parse_args()

    def embedder_fn(batch: list[str]) -> list[list[float]]:
        return ollama_embed(batch, model=args.embed_model)

    def llm_fn(prompt: str) -> str:
        return ollama_chat(prompt, model=args.llm_model)

    # Load texts from Propositionizer dataset
    print(f"Loading {args.texts} texts from Propositionizer dataset...")
    from prism_rag.ingest.splitters.benchmark.propositionizer import load_propositionizer_dataset
    cases = load_propositionizer_dataset(split="test", max_cases=args.texts)
    texts = [c.section_text for c in cases]
    print(f"Loaded {len(texts)} texts")

    # Build multi-granularity index
    print("\nBuilding multi-granularity index...")
    splitter = V2PropositionsSplitter(llm_fn=llm_fn)
    t0 = time.time()
    index = build_multi_granularity_index(
        texts, splitter, embedder_fn, llm_fn, l1_window=3
    )
    build_time = time.time() - t0
    print(f"Index built in {build_time:.1f}s")
    print(f"  L0: {len(index.l0_texts)} knots")
    print(f"  L1: {len(index.l1_texts)} groups")
    print(f"  L2: {len(index.l2_tags)} clusters")
    for i, tag in enumerate(index.l2_tags):
        members = index.l2_members[i]
        print(f"    [{i}] {tag!r} ({len(members)} L1 groups)")

    # Generate QA pairs
    print(f"\nGenerating QA pairs ({args.n_qa} per text)...")
    qa_pairs = generate_qa_pairs(texts, llm_fn, n_per_text=args.n_qa)
    print(f"Generated {len(qa_pairs)} QA pairs")

    if not qa_pairs:
        print("ERROR: No QA pairs generated, aborting")
        sys.exit(1)

    # Run all 5 retrieval strategies
    strategies = [
        ("flat_l0", retrieve_flat_l0),
        ("flat_l1", retrieve_flat_l1),
        ("parent_l0", retrieve_parent_l0),
        ("multi_layer", retrieve_multi_layer),
        ("collapsed", retrieve_collapsed),
    ]

    print(f"\nRunning {len(strategies)} retrieval strategies (top_k={args.top_k})...")
    results = []
    for name, fn in strategies:
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        scores = score_retrieval_strategy(
            name, fn, qa_pairs, index, embedder_fn, llm_fn, top_k=args.top_k
        )
        elapsed = time.time() - t0
        results.append(scores)
        print(f"done ({elapsed:.1f}s) recall={scores['recall']:.3f} mrr={scores['mrr']:.3f}")

    # Print comparison table
    print("\n" + "=" * 90)
    print("Multi-Granularity Retrieval Benchmark Results")
    print("=" * 90)
    print(f"Texts: {len(texts)} | QA pairs: {len(qa_pairs)} | "
          f"L0: {len(index.l0_texts)} | L1: {len(index.l1_texts)} | L2: {len(index.l2_tags)}")
    print("-" * 90)
    print(f"{'Strategy':<15} {'Recall':>8} {'MRR':>8} {'IoU':>8} {'Sufficiency':>12} {'Boundary':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['strategy']:<15} {r['recall']:>8.3f} {r['mrr']:>8.3f} "
              f"{r['iou']:>8.3f} {r['sufficiency']:>12.3f} {r['boundary']:>10.3f}")
    print("-" * 90)

    # Highlight best
    best_recall = max(results, key=lambda r: r["recall"])
    best_mrr = max(results, key=lambda r: r["mrr"])
    print(f"\nBest recall: {best_recall['strategy']} ({best_recall['recall']:.3f})")
    print(f"Best MRR:    {best_mrr['strategy']} ({best_mrr['mrr']:.3f})")


if __name__ == "__main__":
    main()
