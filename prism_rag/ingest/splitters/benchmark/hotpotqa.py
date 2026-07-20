"""HotpotQA dataset loader for multi-hop retrieval evaluation.

Loads the HotpotQA distractor setting from HuggingFace where each case
has 10 paragraphs (2 gold + 8 distractors) and a multi-hop question
requiring reasoning across multiple paragraphs.

Usage::

    from prism_rag.ingest.splitters.benchmark.hotpotqa import (
        load_hotpotqa, HotpotQACase,
    )
    cases = load_hotpotqa(split="validation", max_cases=50)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HotpotQACase:
    """A single HotpotQA evaluation case."""

    question: str
    answer: str
    supporting_titles: list[str]  # gold paragraph titles
    supporting_sent_ids: list[int]  # gold sentence indices
    context_titles: list[str]  # all 10 paragraph titles
    context_paragraphs: list[str]  # all 10 paragraphs as full text
    question_type: str  # "comparison" or "bridge"
    level: str  # "easy", "medium", "hard"


def load_hotpotqa(
    split: str = "validation",
    max_cases: int | None = None,
    config: str = "distractor",
) -> list[HotpotQACase]:
    """Load HotpotQA distractor setting from HuggingFace.

    Dataset: hotpotqa/hotpot_qa, config="distractor"
    Each case has 10 paragraphs (2 gold + 8 distractors).
    Paragraphs are joined from sentence lists into full text.

    Args:
        split: Dataset split ("train" or "validation").
        max_cases: Limit number of cases loaded (None = all).
        config: Dataset config name (default "distractor").

    Returns:
        List of HotpotQACase instances.
    """
    from datasets import load_dataset

    logger.info(
        "Loading HotpotQA (config=%s, split=%s, max=%s)...",
        config, split, max_cases,
    )

    ds = load_dataset("hotpotqa/hotpot_qa", config, split=split)

    cases: list[HotpotQACase] = []
    for i, row in enumerate(ds):
        if max_cases is not None and i >= max_cases:
            break

        # Parse context: {"title": [str], "sentences": [list[str]]}
        context = row["context"]
        titles: list[str] = context["title"]
        sentences_lists: list[list[str]] = context["sentences"]

        # Join each paragraph's sentences into one string
        paragraphs: list[str] = []
        for sent_list in sentences_lists:
            paragraphs.append(" ".join(sent_list))

        # Parse supporting facts: {"title": [str], "sent_id": [int]}
        sup_facts = row["supporting_facts"]
        sup_titles = list(set(sup_facts["title"]))
        sup_sent_ids = sup_facts["sent_id"]

        cases.append(HotpotQACase(
            question=row["question"],
            answer=row["answer"],
            supporting_titles=sup_titles,
            supporting_sent_ids=sup_sent_ids,
            context_titles=titles,
            context_paragraphs=paragraphs,
            question_type=row.get("type", "unknown"),
            level=row.get("level", "unknown"),
        ))

    logger.info("Loaded %d HotpotQA cases", len(cases))
    return cases
