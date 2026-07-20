#!/usr/bin/env python3
"""HotpotQA multi-hop retrieval benchmark.

Compares multi-granularity retrieval strategies (flat L1, parent L0,
PPR, PPR->L1 hybrid) on HotpotQA distractor setting.

Usage::

    OLLAMA_HOST=http://localhost:11434 python3 scripts/run_hotpotqa_eval.py --cases 50

HotpotQA is multi-hop: answers require info from 2 different paragraphs.
PPR should excel here because graph traversal connects entities across paragraphs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prism_rag.ingest.splitters.benchmark.downstream import (
    ollama_chat,
    ollama_embed,
    word_overlap,
)
from prism_rag.ingest.splitters.benchmark.hotpotqa import (
    HotpotQACase,
    load_hotpotqa,
)
from prism_rag.ingest.splitters.benchmark.multi_granularity import (
    _cosine,
    _find_top_k,
    atomize_with_entities,
    build_l1_groups_by_entity,
)
from prism_rag.ingest.splitters.benchmark.ppr_retrieval import (
    AtomEntityGraph,
    build_atom_entity_graph,
    extract_query_entities,
    retrieve_ppr,
    retrieve_ppr_l1,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _answer_in_chunks(answer: str, chunks: list[str], threshold: float = 0.5) -> bool:
    """Check if answer words are found in any retrieved chunk."""
    combined = " ".join(chunks)
    return word_overlap(combined, answer) >= threshold


def _find_first_relevant_rank(
    chunks: list[str],
    gold_titles: list[str],
    paragraphs: list[str],
    titles: list[str],
) -> int | None:
    """Find rank (1-based) of first chunk overlapping with a gold paragraph."""
    # Build gold paragraph texts
    gold_paras = set()
    for gt in gold_titles:
        for i, t in enumerate(titles):
            if t == gt:
                gold_paras.add(paragraphs[i])
                break

    for rank, chunk in enumerate(chunks, 1):
        for gp in gold_paras:
            if word_overlap(chunk, gp) >= 0.3:
                return rank
    return None


@dataclass
class StrategyResult:
    """Results for one retrieval strategy across all cases."""
    name: str
    recall_sum: float = 0.0
    mrr_sum: float = 0.0
    answer_coverage_sum: float = 0.0
    multi_hop_recall_sum: float = 0.0
    count: int = 0

    @property
    def recall(self) -> float:
        return self.recall_sum / max(self.count, 1)

    @property
    def mrr(self) -> float:
        return self.mrr_sum / max(self.count, 1)

    @property
    def answer_coverage(self) -> float:
        return self.answer_coverage_sum / max(self.count, 1)

    @property
    def multi_hop_recall(self) -> float:
        return self.multi_hop_recall_sum / max(self.count, 1)


def _score_chunks(
    chunks: list[str],
    case: HotpotQACase,
    result: StrategyResult,
) -> None:
    """Score retrieved chunks against a HotpotQA case, accumulate into result."""
    result.count += 1

    # Answer recall: do retrieved chunks contain the answer?
    if _answer_in_chunks(case.answer, chunks, threshold=0.4):
        result.recall_sum += 1.0

    # Answer coverage: word overlap between retrieved text and gold answer
    combined = " ".join(chunks)
    result.answer_coverage_sum += word_overlap(combined, case.answer)

    # MRR: rank of first chunk overlapping with a gold paragraph
    rank = _find_first_relevant_rank(
        chunks, case.supporting_titles, case.context_paragraphs, case.context_titles,
    )
    if rank is not None:
        result.mrr_sum += 1.0 / rank

    # Multi-hop recall: do retrieved chunks cover BOTH gold paragraphs?
    gold_paras_found: set[str] = set()
    for gt in case.supporting_titles:
        for i, t in enumerate(case.context_titles):
            if t == gt:
                gold_text = case.context_paragraphs[i]
                for chunk in chunks:
                    if word_overlap(chunk, gold_text) >= 0.25:
                        gold_paras_found.add(gt)
                        break
                break
    if len(gold_paras_found) == len(case.supporting_titles):
        result.multi_hop_recall_sum += 1.0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_eval(cases: list[HotpotQACase], top_k: int = 10) -> dict[str, StrategyResult]:
    """Run all retrieval strategies on HotpotQA cases."""

    strategies = {
        "flat_l1_entity": StrategyResult("flat_l1_entity"),
        "parent_l0_entity": StrategyResult("parent_l0_entity"),
        "ppr": StrategyResult("ppr"),
        "ppr_l1": StrategyResult("ppr_l1"),
    }

    total = len(cases)
    for ci, case in enumerate(cases):
        t0 = time.time()
        logger.info(
            "[%d/%d] Processing: %s (type=%s, level=%s)",
            ci + 1, total, case.question[:80], case.question_type, case.level,
        )

        # ---- Step 1: Atomize all 10 context paragraphs ----
        paragraphs = case.context_paragraphs
        l0_texts, l0_entities, l0_source_idx = atomize_with_entities(
            paragraphs, ollama_chat,
        )

        if not l0_texts:
            logger.warning("  No knots produced, skipping case")
            continue

        # ---- Step 2: Build L1 groups (entity-based) ----
        l1_groups = build_l1_groups_by_entity(
            l0_texts, l0_entities, l0_source_idx, max_group_size=5,
        )
        l1_texts = [g[0] for g in l1_groups]
        l1_members = [g[1] for g in l1_groups]

        # ---- Step 3: Embed L0 and L1 ----
        all_to_embed = l0_texts + l1_texts
        all_vectors: list[list[float]] = []
        batch_size = 32
        for i in range(0, len(all_to_embed), batch_size):
            batch = all_to_embed[i : i + batch_size]
            vecs = ollama_embed(batch)
            all_vectors.extend(vecs)

        l0_vectors = all_vectors[:len(l0_texts)]
        l1_vectors = all_vectors[len(l0_texts):]

        # ---- Step 4: Build Atom-Entity graph ----
        aeg = build_atom_entity_graph(
            l0_texts, l0_entities, l0_vectors,
            synonym_threshold=0.85,
            embedder_fn=ollama_embed,
        )

        # ---- Step 5: Embed query and extract entities ----
        q_vec = ollama_embed([case.question])[0]
        q_entities = extract_query_entities(case.question, ollama_chat)

        # ---- Step 6: Run all retrieval strategies ----

        # flat_l1_entity: cosine search on L1 vectors
        l1_indices = _find_top_k(q_vec, l1_vectors, top_k)
        flat_l1_chunks = [l1_texts[i] for i in l1_indices]
        _score_chunks(flat_l1_chunks, case, strategies["flat_l1_entity"])

        # parent_l0_entity: L0 cosine -> L1 parent
        l0_indices = _find_top_k(q_vec, l0_vectors, top_k * 2)
        l0_to_l1: dict[int, int] = {}
        for l1_idx, members in enumerate(l1_members):
            for l0_idx in members:
                l0_to_l1[l0_idx] = l1_idx
        seen_l1: set[int] = set()
        parent_chunks: list[str] = []
        for l0_idx in l0_indices:
            l1_idx = l0_to_l1.get(l0_idx)
            if l1_idx is not None and l1_idx not in seen_l1:
                seen_l1.add(l1_idx)
                parent_chunks.append(l1_texts[l1_idx])
                if len(parent_chunks) >= top_k:
                    break
        _score_chunks(parent_chunks, case, strategies["parent_l0_entity"])

        # ppr: PPR on Atom-Entity graph
        ppr_chunks = retrieve_ppr(
            case.question, q_entities, q_vec, aeg,
            damping=0.3, atom_seed_weight=0.1, top_k=top_k * 3,
        )
        # Trim to top_k for fair comparison
        ppr_chunks = ppr_chunks[:top_k]
        _score_chunks(ppr_chunks, case, strategies["ppr"])

        # ppr_l1: PPR atoms -> L1 groups
        ppr_l1_chunks = retrieve_ppr_l1(
            case.question, q_entities, q_vec, aeg,
            l1_members, l1_texts,
            damping=0.3, atom_seed_weight=0.1, top_k=top_k,
        )
        _score_chunks(ppr_l1_chunks, case, strategies["ppr_l1"])

        elapsed = time.time() - t0
        logger.info("  Done in %.1fs (L0=%d, L1=%d, entities=%d, edges=%d)",
                     elapsed, len(l0_texts), len(l1_texts),
                     len(aeg.entity_nodes), aeg.graph.number_of_edges())

    return strategies


def format_results(strategies: dict[str, StrategyResult], n_cases: int) -> str:
    """Format results as a table."""
    lines = [
        f"\n{'='*80}",
        f"HotpotQA Multi-Hop Retrieval Benchmark — {n_cases} cases",
        f"{'='*80}",
        "",
        f"{'Strategy':<20} {'Recall':>8} {'MRR':>8} {'Coverage':>10} {'MultiHop':>10} {'Count':>6}",
        f"{'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*6}",
    ]
    for name in ["flat_l1_entity", "parent_l0_entity", "ppr", "ppr_l1"]:
        s = strategies[name]
        lines.append(
            f"{s.name:<20} {s.recall:>8.3f} {s.mrr:>8.3f} "
            f"{s.answer_coverage:>10.3f} {s.multi_hop_recall:>10.3f} {s.count:>6d}"
        )
    lines.append("")
    lines.append("Recall: fraction of cases where answer words found in retrieved text")
    lines.append("MRR: mean reciprocal rank of first chunk from a gold paragraph")
    lines.append("Coverage: word overlap between retrieved text and gold answer")
    lines.append("MultiHop: fraction of cases where BOTH gold paragraphs covered")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="HotpotQA multi-hop retrieval benchmark")
    parser.add_argument("--cases", type=int, default=50, help="Number of cases (default 50)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k retrieval (default 10)")
    parser.add_argument("--split", default="validation", help="Dataset split")
    args = parser.parse_args()

    logger.info("Loading HotpotQA dataset...")
    cases = load_hotpotqa(split=args.split, max_cases=args.cases)
    logger.info("Loaded %d cases", len(cases))

    # Print distribution
    types = {}
    levels = {}
    for c in cases:
        types[c.question_type] = types.get(c.question_type, 0) + 1
        levels[c.level] = levels.get(c.level, 0) + 1
    logger.info("Types: %s", types)
    logger.info("Levels: %s", levels)

    strategies = run_eval(cases, top_k=args.top_k)
    print(format_results(strategies, len(cases)))


if __name__ == "__main__":
    main()
