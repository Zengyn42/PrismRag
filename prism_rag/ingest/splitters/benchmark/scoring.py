"""Heuristic scoring functions for splitter quality evaluation.

All scorers are pure heuristics — no LLM calls required. The module is
designed for extensibility: future versions can add LLM-based scorers
(GPT-as-judge) as optional alternatives.
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
        overall: Weighted average of the four dimensions.
    """

    splitter_name: str
    atomicity: float
    self_containedness: float
    faithfulness: float
    coverage: float
    overall: float


# Weights for the overall score (must sum to 1.0).
_WEIGHTS = {
    "atomicity": 0.25,
    "self_containedness": 0.25,
    "faithfulness": 0.25,
    "coverage": 0.25,
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
            overall=0.0,
        )

    atomicity_scores: list[float] = []
    self_contained_scores: list[float] = []
    faithfulness_scores: list[float] = []
    coverage_scores: list[float] = []

    for case in cases:
        knots = splitter.split(case.section_text, doc_context=case.doc_context)
        atomicity_scores.append(score_atomicity(knots))
        self_contained_scores.append(score_self_containedness(knots))
        faithfulness_scores.append(score_faithfulness(knots, case.section_text))
        coverage_scores.append(score_coverage(knots, case.section_text))

    avg_atom = sum(atomicity_scores) / len(atomicity_scores)
    avg_self = sum(self_contained_scores) / len(self_contained_scores)
    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_cov = sum(coverage_scores) / len(coverage_scores)

    overall = (
        _WEIGHTS["atomicity"] * avg_atom
        + _WEIGHTS["self_containedness"] * avg_self
        + _WEIGHTS["faithfulness"] * avg_faith
        + _WEIGHTS["coverage"] * avg_cov
    )

    return SplitterScore(
        splitter_name=splitter.name,
        atomicity=round(avg_atom, 4),
        self_containedness=round(avg_self, 4),
        faithfulness=round(avg_faith, 4),
        coverage=round(avg_cov, 4),
        overall=round(overall, 4),
    )
