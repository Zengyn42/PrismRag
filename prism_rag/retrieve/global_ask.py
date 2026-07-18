"""Phase 4 (v6.0): global_ask — map-reduce question answering over community reports.

Map phase:  For each community report, ask the LLM whether the community is
            relevant to the user's question and extract a per-community answer.
Reduce phase: Combine all per-community answers into a single synthesized answer.

If no pre-generated reports are provided the function generates them on-the-fly
(slower, but works out of the box).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable

from prism_rag.report.community_report import (
    CommunityReport,
    generate_all_community_reports,
)
from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

_OLLAMA_DEFAULT_HOST = "http://localhost:11434"
_OLLAMA_DEFAULT_MODEL = "qwen3.5:9b"

# Truncate individual community summaries in map prompts to avoid token blow-up.
_MAX_SUMMARY_CHARS = 2000

# ── Prompts ──────────────────────────────────────────────────────────────────

_MAP_PROMPT = """\
You are a knowledge analyst. Given a community summary and a user question, \
determine whether this community's knowledge is relevant to the question. \
If relevant, provide a focused answer drawing ONLY from the community summary. \
If not relevant, say "NOT RELEVANT".

## Community: {title} (rating {rating}/10)

{summary}

Key findings:
{key_findings}

## Question

{question}

## Instructions

Answer the question using ONLY information from the community summary above. \
Be concise (1-3 sentences). If the community has no relevant information, \
respond with exactly: NOT RELEVANT
"""

_REDUCE_PROMPT = """\
You are a knowledge synthesis engine. Multiple community-level answers are \
provided below. Synthesize them into a single, coherent, comprehensive answer \
to the user's question.

## Question

{question}

## Community answers

{community_answers}

## Instructions

Produce a clear, well-structured answer that integrates all relevant community \
answers. Cite community titles where appropriate. If no community had relevant \
information, say so honestly. Be thorough but concise.
"""


# ── LLM default ──────────────────────────────────────────────────────────────


def _make_default_llm_fn() -> Callable[[str], str]:
    from prism_rag.report.community_report import _default_ollama_generate

    model = os.environ.get("PRISM_OLLAMA_LLM_MODEL", _OLLAMA_DEFAULT_MODEL)
    host = os.environ.get("PRISM_OLLAMA_HOST", _OLLAMA_DEFAULT_HOST)
    timeout = 300
    return lambda prompt: _default_ollama_generate(
        prompt, model=model, host=host, timeout=timeout
    )


# ── Core ─────────────────────────────────────────────────────────────────────


def global_ask(
    question: str,
    graph: KnowledgeGraph,
    community_reports: list[CommunityReport] | None = None,
    llm_fn: Callable[[str], str] | None = None,
    min_members: int = 3,
) -> str:
    """Answer a question using map-reduce over community reports.

    Args:
        question: The user's question.
        graph: Knowledge graph (used to generate reports on-the-fly if needed).
        community_reports: Pre-generated reports. If None, generates them.
        llm_fn: LLM callable (prompt -> text). Defaults to Ollama.
        min_members: Minimum community size for on-the-fly generation.

    Returns:
        Final synthesized answer string.
    """
    if llm_fn is None:
        llm_fn = _make_default_llm_fn()

    # Generate reports on-the-fly if not provided
    if community_reports is None:
        logger.info("[global_ask] no cached reports, generating on-the-fly")
        community_reports = generate_all_community_reports(
            graph, llm_fn=llm_fn, min_members=min_members
        )

    if not community_reports:
        return "No communities with enough members to answer the question."

    # ── Map phase ────────────────────────────────────────────────────
    community_answers: list[tuple[str, str]] = []  # (title, answer)
    for report in community_reports:
        key_findings_text = "\n".join(f"- {f}" for f in report.key_findings) or "(none)"
        summary_text = report.summary[:_MAX_SUMMARY_CHARS]

        prompt = _MAP_PROMPT.format(
            title=report.title,
            rating=report.rating,
            summary=summary_text,
            key_findings=key_findings_text,
            question=question,
        )
        answer = llm_fn(prompt).strip()
        if answer.upper() != "NOT RELEVANT":
            community_answers.append((report.title, answer))
            logger.debug("[global_ask:map] %s → relevant", report.community_id)
        else:
            logger.debug("[global_ask:map] %s → not relevant", report.community_id)

    if not community_answers:
        return "No community had relevant information for this question."

    # ── Reduce phase ─────────────────────────────────────────────────
    answers_text = "\n\n".join(
        f"### {title}\n{answer}" for title, answer in community_answers
    )
    reduce_prompt = _REDUCE_PROMPT.format(
        question=question,
        community_answers=answers_text,
    )
    final_answer = llm_fn(reduce_prompt).strip()
    return final_answer
