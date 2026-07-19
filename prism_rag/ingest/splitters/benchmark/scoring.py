"""Heuristic scoring functions for splitter quality evaluation.

All scorers are pure heuristics — no LLM calls required. The module is
designed for extensibility: future versions can add LLM-based scorers
(GPT-as-judge) as optional alternatives.

When gold-standard reference knots are available (e.g. from the
Propositionizer dataset), the ``gold_alignment`` dimension measures
precision/recall against the gold set using word-level F1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from prism_rag.ingest.splitters.base import Knot, Splitter
from prism_rag.ingest.splitters.benchmark.dataset import BenchmarkCase

# Pronouns that indicate an unresolved reference when they appear at the
# start of a sentence (case-insensitive).
_DANGLING_PRONOUN_RE = re.compile(
    r"^(it|this|that|these|those|they|he|she|its|their)\b",
    re.IGNORECASE,
)

# Simple sentence boundary heuristic for counting sentences in a knot.
_SENTENCE_SPLIT_RE = re.compile(r"[.?!。？！]\s+")


@dataclass
class SplitterScore:
    """Aggregated quality score for a splitter across a benchmark dataset.

    All dimension scores are in [0, 1] where 1 is best.

    Attributes:
        splitter_name: Name of the scored splitter.
        atomicity: Penalizes knots with >1 sentence or >50 words.
        self_containedness: Penalizes knots starting with dangling pronouns.
        faithfulness: Checks knot words are present in source text.
        coverage: Fraction of source text covered by knot output.
        gold_alignment: F1 match against gold-standard propositions (0 if
            no gold available).
        overall: Weighted average of scored dimensions.
    """

    splitter_name: str
    atomicity: float
    self_containedness: float
    faithfulness: float
    coverage: float
    gold_alignment: float = 0.0
    overall: float = 0.0


# Weights when gold is NOT available (must sum to 1.0).
_WEIGHTS_NO_GOLD = {
    "atomicity": 0.25,
    "self_containedness": 0.25,
    "faithfulness": 0.25,
    "coverage": 0.25,
}

# Weights when gold IS available (must sum to 1.0).
_WEIGHTS_WITH_GOLD = {
    "atomicity": 0.20,
    "self_containedness": 0.15,
    "faithfulness": 0.20,
    "coverage": 0.15,
    "gold_alignment": 0.30,
}


def score_atomicity(knots: list[Knot]) -> float:
    """Score atomicity: penalize knots with multiple sentences or >50 words.

    Returns 1.0 when every knot is a single short sentence.
    """
    if not knots:
        return 0.0

    scores: list[float] = []
    for knot in knots:
        text = knot.text.strip()
        if not text:
            scores.append(0.0)
            continue

        # Count sentences: split by sentence-ending punctuation
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        # The last segment may not end with punctuation, so always count it
        if sentences:
            n_sentences = len(sentences)
        else:
            n_sentences = 1

        word_count = len(text.split())

        # Penalty: >1 sentence or >50 words
        sent_penalty = max(0.0, (n_sentences - 1) * 0.3)  # 0.3 per extra sentence
        word_penalty = max(0.0, (word_count - 50) / 100)  # gradual penalty above 50 words

        score = max(0.0, 1.0 - sent_penalty - word_penalty)
        scores.append(score)

    return sum(scores) / len(scores)


def score_self_containedness(knots: list[Knot]) -> float:
    """Score self-containedness: penalize dangling pronouns at sentence start.

    Returns 1.0 when no knot starts with an unresolved pronoun reference.
    """
    if not knots:
        return 0.0

    clean = 0
    for knot in knots:
        text = knot.text.strip()
        if not text:
            continue
        if not _DANGLING_PRONOUN_RE.match(text):
            clean += 1

    total = len([k for k in knots if k.text.strip()])
    return clean / total if total > 0 else 0.0


def score_faithfulness(knots: list[Knot], source_text: str) -> float:
    """Score faithfulness: check that knot words appear in the source.

    Word-level Jaccard-like measure. Returns 1.0 when every word in every
    knot is also present in the source text (no hallucination).
    """
    if not knots:
        return 0.0

    source_words = set(re.findall(r"\w+", source_text.lower()))
    if not source_words:
        return 0.0

    scores: list[float] = []
    for knot in knots:
        knot_words = set(re.findall(r"\w+", knot.text.lower()))
        if not knot_words:
            scores.append(1.0)  # empty knot is vacuously faithful
            continue
        overlap = len(knot_words & source_words)
        scores.append(overlap / len(knot_words))

    return sum(scores) / len(scores)


def score_coverage(knots: list[Knot], source_text: str) -> float:
    """Score coverage: fraction of source text covered by knot output.

    Aggregate knot text length / source text length, capped at 1.0.
    """
    source_len = len(source_text.strip())
    if source_len == 0:
        return 0.0

    knot_len = sum(len(k.text.strip()) for k in knots)
    return min(1.0, knot_len / source_len)


def _word_set(text: str) -> set[str]:
    """Lowercase word set for F1 computation."""
    return set(re.findall(r"\w+", text.lower()))


def _best_f1_match(pred_text: str, gold_texts: list[str]) -> float:
    """Find the best word-level F1 between pred_text and any gold text."""
    pred_words = _word_set(pred_text)
    if not pred_words:
        return 0.0

    best = 0.0
    for gold in gold_texts:
        gold_words = _word_set(gold)
        if not gold_words:
            continue
        overlap = len(pred_words & gold_words)
        if overlap == 0:
            continue
        precision = overlap / len(pred_words)
        recall = overlap / len(gold_words)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def score_gold_alignment(knots: list[Knot], gold_knots: list[Knot]) -> float:
    """Score alignment between predicted knots and gold-standard propositions.

    Uses a greedy best-match strategy:
    - For each predicted knot, find the gold knot with highest word-level F1
    - For each gold knot, find the predicted knot with highest word-level F1
    - Precision = avg of best matches per predicted knot (how accurate are preds?)
    - Recall = avg of best matches per gold knot (how many golds are covered?)
    - Returns F1 of (precision, recall)

    This rewards splitters that produce knots close to the gold decomposition
    without penalizing minor phrasing differences.
    """
    if not knots or not gold_knots:
        return 0.0

    gold_texts = [k.text for k in gold_knots]
    pred_texts = [k.text for k in knots]

    # Precision: for each predicted knot, how well does it match some gold?
    pred_scores = [_best_f1_match(p, gold_texts) for p in pred_texts]
    precision = sum(pred_scores) / len(pred_scores) if pred_scores else 0.0

    # Recall: for each gold knot, how well does some predicted knot match it?
    recall_scores = [_best_f1_match(g, pred_texts) for g in gold_texts]
    recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_splitter(
    splitter: Splitter,
    cases: list[BenchmarkCase],
) -> SplitterScore:
    """Run a splitter on all benchmark cases and return aggregated scores.

    Args:
        splitter: The splitter instance to evaluate.
        cases: Benchmark cases to run.

    Returns:
        A :class:`SplitterScore` with per-dimension and overall scores.
    """
    if not cases:
        return SplitterScore(
            splitter_name=splitter.name,
            atomicity=0.0,
            self_containedness=0.0,
            faithfulness=0.0,
            coverage=0.0,
            gold_alignment=0.0,
            overall=0.0,
        )

    atomicity_scores: list[float] = []
    self_contained_scores: list[float] = []
    faithfulness_scores: list[float] = []
    coverage_scores: list[float] = []
    gold_scores: list[float] = []
    has_gold = False

    for case in cases:
        try:
            knots = splitter.split(case.section_text, doc_context=case.doc_context)
        except Exception:
            knots = []  # graceful degradation on LLM failures
        atomicity_scores.append(score_atomicity(knots))
        self_contained_scores.append(score_self_containedness(knots))
        faithfulness_scores.append(score_faithfulness(knots, case.section_text))
        coverage_scores.append(score_coverage(knots, case.section_text))
        if case.reference_knots:
            has_gold = True
            gold_scores.append(score_gold_alignment(knots, case.reference_knots))

    avg_atom = sum(atomicity_scores) / len(atomicity_scores)
    avg_self = sum(self_contained_scores) / len(self_contained_scores)
    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_cov = sum(coverage_scores) / len(coverage_scores)
    avg_gold = sum(gold_scores) / len(gold_scores) if gold_scores else 0.0

    if has_gold:
        weights = _WEIGHTS_WITH_GOLD
        overall = (
            weights["atomicity"] * avg_atom
            + weights["self_containedness"] * avg_self
            + weights["faithfulness"] * avg_faith
            + weights["coverage"] * avg_cov
            + weights["gold_alignment"] * avg_gold
        )
    else:
        weights = _WEIGHTS_NO_GOLD
        overall = (
            weights["atomicity"] * avg_atom
            + weights["self_containedness"] * avg_self
            + weights["faithfulness"] * avg_faith
            + weights["coverage"] * avg_cov
        )

    return SplitterScore(
        splitter_name=splitter.name,
        atomicity=round(avg_atom, 4),
        self_containedness=round(avg_self, 4),
        faithfulness=round(avg_faith, 4),
        coverage=round(avg_cov, 4),
        gold_alignment=round(avg_gold, 4),
        overall=round(overall, 4),
    )
