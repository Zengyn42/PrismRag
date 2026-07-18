"""GleaningsSplitter — LLM atomization with GraphRAG-style gleaning rounds.

GraphRAG's key recall trick (graph_extractor.py): after the first extraction
pass, re-prompt with "MANY entities were missed — add them", up to
``max_gleanings`` rounds. Cheap, model-agnostic recall booster.

Adaptation for the stateless ``llm_fn`` backend: instead of continuing one
conversation, each gleaning round re-sends the source text plus the atoms
already extracted, and asks ONLY for missed ones. Rounds stop early when a
pass yields no new atoms (the equivalent of GraphRAG's Y/N loop gate).

Duplicates within/across rounds are dropped by exact-text match here; the
ingest pipeline's 0.90-cosine semantic dedup catches near-duplicates later.
"""

from __future__ import annotations

import logging

from prism_rag.ingest.splitters.base import Knot
from prism_rag.ingest.splitters.llm import LlmSplitter, _extract_json_array

logger = logging.getLogger(__name__)

GLEANING_PROMPT_V1 = """\
You previously atomized a source text into knowledge units (KNOTs), listed
below. MANY atomic knowledge units may have been MISSED — subtle facts,
side remarks, constraints, version numbers, rationales behind decisions.

## Source text

{section_text}

## Already extracted (do NOT repeat these)

{extracted_block}

## Task

Return ONLY the MISSED knowledge units as a JSON array (no markdown fences,
no commentary), same schema as before:
[
  {{"title": "<short label>",
    "body": "<self-contained atomic statement>",
    "ontology_type": "<concept|fact|decision|procedure>",
    "context_note": "<optional, empty string if not needed>"}}
]
Rules unchanged: self-contained, atomic, faithful, worth keeping.
Return [] if nothing was missed.
"""


class GleaningsSplitter(LlmSplitter):
    """LlmSplitter + up to ``max_gleanings`` missed-atom recovery rounds.

    Args:
        max_gleanings: maximum number of follow-up rounds (default 2).
        Remaining args are forwarded to :class:`LlmSplitter`.
    """

    def __init__(self, *args, max_gleanings: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        if max_gleanings < 0:
            raise ValueError("max_gleanings must be >= 0")
        self._max_gleanings = max_gleanings

    @property
    def name(self) -> str:
        return "llm_gleanings"

    def split(
        self,
        section_text: str,
        *,
        doc_context: str | None = None,
    ) -> list[Knot]:
        knots = super().split(section_text, doc_context=doc_context)
        if not knots:
            return knots

        seen_texts = {k.text for k in knots}

        for round_no in range(1, self._max_gleanings + 1):
            extracted_block = "\n".join(
                f"- [{k.ontology_type}] {k.title}: {k.text}" for k in knots
            )
            prompt = GLEANING_PROMPT_V1.format(
                section_text=section_text, extracted_block=extracted_block
            )
            raw = self._llm_fn(prompt)
            try:
                items = _extract_json_array(raw)
            except ValueError:
                logger.warning(
                    "[GleaningsSplitter] round %d output unparseable — stopping",
                    round_no,
                )
                break

            new_knots = [
                k
                for k in self._items_to_knots(items)
                if k.text not in seen_texts
            ]
            if not new_knots:
                logger.info(
                    "[GleaningsSplitter] round %d yielded nothing new — done",
                    round_no,
                )
                break

            for k in new_knots:
                k.method = self.name
                k.metadata["gleaning_round"] = round_no
                seen_texts.add(k.text)
            knots.extend(new_knots)
            logger.info(
                "[GleaningsSplitter] round %d recovered %d missed knot(s)",
                round_no,
                len(new_knots),
            )

        # Base-round knots carry method="llm" from super(); unify.
        for k in knots:
            k.method = self.name
        return knots
