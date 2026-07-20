#!/usr/bin/env python3
"""Entity-based vs window-based L1 grouping benchmark.

Compares 5 retrieval strategies:

  flat_l0            -- L0 cosine search (baseline)
  flat_l1_window     -- L1 fixed-window grouping (current, window=3)
  flat_l1_entity     -- L1 entity-based grouping (NEW)
  parent_l0          -- L0 search -> return L1(window) parent
  parent_l0_entity   -- L0 search -> return L1(entity) parent

Uses 50 texts from Propositionizer dataset, shared QA pairs,
gemma4:e4b (LLM, think:false), qwen3-embedding:8b (embedding).

Usage::

    OLLAMA_HOST=http://localhost:11434 python3 scripts/run_entity_grouping_eval.py
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
    score_boundary_clarity,
)
from prism_rag.ingest.splitters.benchmark.multi_granularity import (
    MultiGranularityIndex,
    build_multi_granularity_index,
    build_multi_granularity_index_entity,
    retrieve_flat_l0,
    retrieve_flat_l1,
    retrieve_parent_l0,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---- V2 Propositions splitter (same as run_multi_granularity_eval.py) ------

PROMPT_V2_PROPOSITIONS = """\
Decompose the following text into clear, simple propositions. A proposition is:
- A single, atomic fact or assertion
- Self-contained — understandable without reading the original text
- Minimal — as concise as possible while remaining unambiguous

Rules:
1. Split compound sentences into simple ones
2. Resolve ALL coreferences — replace pronouns and implicit references with \
the actual entity names (e.g., "it" -> "the Leaning Tower of Pisa")
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

        try:
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


# ---- Evaluation helpers ---------------------------------------------------

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

        # IoU
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

    boundary = score_boundary_clarity(index.l1_vectors)

    return {
        "strategy": strategy_name,
        "recall": recall,
        "mrr": mrr,
        "iou": iou,
        "boundary": boundary,
    }


# ---- Main -----------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Entity-based L1 grouping benchmark")
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

    # Build window-based index
    print("\nBuilding window-based multi-granularity index (window=3)...")
    splitter = V2PropositionsSplitter(llm_fn=llm_fn)
    t0 = time.time()
    index_window = build_multi_granularity_index(
        texts, splitter, embedder_fn, llm_fn, l1_window=3
    )
    build_window_time = time.time() - t0
    print(f"Window index built in {build_window_time:.1f}s")
    print(f"  L0: {len(index_window.l0_texts)} knots")
    print(f"  L1: {len(index_window.l1_texts)} groups")
    print(f"  L2: {len(index_window.l2_tags)} clusters")

    # Build entity-based index
    print("\nBuilding entity-based multi-granularity index...")
    t0 = time.time()
    index_entity = build_multi_granularity_index_entity(
        texts, embedder_fn, llm_fn, max_group_size=5
    )
    build_entity_time = time.time() - t0
    print(f"Entity index built in {build_entity_time:.1f}s")
    print(f"  L0: {len(index_entity.l0_texts)} knots")
    print(f"  L1: {len(index_entity.l1_texts)} groups")
    print(f"  L2: {len(index_entity.l2_tags)} clusters")

    # L1 group size statistics
    for label, idx in [("window", index_window), ("entity", index_entity)]:
        sizes = [len(m) for m in idx.l1_members]
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            print(f"  L1 ({label}) avg group size: {avg_size:.2f}, "
                  f"min={min(sizes)}, max={max(sizes)}, "
                  f"singletons={sum(1 for s in sizes if s == 1)}")

    # Generate QA pairs (shared across all strategies)
    print(f"\nGenerating QA pairs ({args.n_qa} per text)...")
    qa_pairs = generate_qa_pairs(texts, llm_fn, n_per_text=args.n_qa)
    print(f"Generated {len(qa_pairs)} QA pairs")

    if not qa_pairs:
        print("ERROR: No QA pairs generated, aborting")
        sys.exit(1)

    # Run strategies
    strategies = [
        ("flat_l0", retrieve_flat_l0, index_window),
        ("flat_l1_window", retrieve_flat_l1, index_window),
        ("flat_l1_entity", retrieve_flat_l1, index_entity),
        ("parent_l0", retrieve_parent_l0, index_window),
        ("parent_l0_entity", retrieve_parent_l0, index_entity),
    ]

    print(f"\nRunning {len(strategies)} retrieval strategies (top_k={args.top_k})...")
    results = []
    for name, fn, index in strategies:
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        scores = score_retrieval_strategy(
            name, fn, qa_pairs, index, embedder_fn, llm_fn, top_k=args.top_k
        )
        elapsed = time.time() - t0
        results.append(scores)
        print(f"done ({elapsed:.1f}s) recall={scores['recall']:.3f} mrr={scores['mrr']:.3f}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("Entity-Based L1 Grouping Benchmark Results")
    print("=" * 80)
    print(f"Texts: {len(texts)} | QA pairs: {len(qa_pairs)}")
    print(f"Window index — L0: {len(index_window.l0_texts)} | "
          f"L1: {len(index_window.l1_texts)}")
    print(f"Entity index — L0: {len(index_entity.l0_texts)} | "
          f"L1: {len(index_entity.l1_texts)}")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Recall':>8} {'MRR':>8} {'IoU':>8} {'Boundary':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['strategy']:<20} {r['recall']:>8.3f} {r['mrr']:>8.3f} "
              f"{r['iou']:>8.3f} {r['boundary']:>10.3f}")
    print("-" * 80)

    # Highlight best
    best_recall = max(results, key=lambda r: r["recall"])
    best_mrr = max(results, key=lambda r: r["mrr"])
    print(f"\nBest recall: {best_recall['strategy']} ({best_recall['recall']:.3f})")
    print(f"Best MRR:    {best_mrr['strategy']} ({best_mrr['mrr']:.3f})")

    # Direct comparison
    window_result = next(r for r in results if r["strategy"] == "flat_l1_window")
    entity_result = next(r for r in results if r["strategy"] == "flat_l1_entity")
    mrr_delta = entity_result["mrr"] - window_result["mrr"]
    recall_delta = entity_result["recall"] - window_result["recall"]
    print(f"\nEntity vs Window L1:")
    print(f"  MRR delta:    {mrr_delta:+.3f}")
    print(f"  Recall delta: {recall_delta:+.3f}")


if __name__ == "__main__":
    main()
