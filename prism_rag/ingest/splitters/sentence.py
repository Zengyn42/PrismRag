"""SentenceSplitter — rule-based sentence boundary splitter.

Splits text at sentence-ending punctuation (. ? ! 。？！) and newlines.
Markdown fenced code blocks (``` ... ```) are preserved intact as single
claims with metadata['is_code'] = True.

This is a deterministic baseline splitter — no LLM calls, no external
dependencies. Useful as a benchmark reference and for offline testing.
"""

from __future__ import annotations

import logging
import re

from prism_rag.ingest.splitters.base import Knot, Splitter

logger = logging.getLogger(__name__)

# Matches a fenced code block: opening ``` (with optional language tag) through closing ```
_CODE_FENCE_RE = re.compile(
    r"```[^\n]*\n.*?```",
    re.DOTALL,
)

# Sentence-ending punctuation (English + Chinese) used as split points.
_SENTENCE_END_RE = re.compile(r"(?<=[.?!。？！])\s*")

# Collapse runs of whitespace to a single space.
_MULTI_WS_RE = re.compile(r"[ \t]+")


def _collapse_whitespace(text: str) -> str:
    """Collapse consecutive internal whitespace (spaces/tabs) to single space."""
    return _MULTI_WS_RE.sub(" ", text).strip()


class SentenceSplitter(Splitter):
    """Rule-based sentence boundary splitter.

    Splitting rules:
      - Sentence-ending punctuation (. ? ! 。？！) acts as a boundary.
      - Newlines act as boundaries.
      - Fenced code blocks (``` ... ```) are kept intact as a single claim
        with ``metadata['is_code'] = True``.
      - Empty / whitespace-only sentences are discarded.
      - Consecutive internal whitespace is collapsed to a single space.
      - Each claim gets ``metadata['index']`` (0-based, consecutive).
    """

    @property
    def name(self) -> str:
        return "sentence"

    def split(
        self,
        section_text: str,
        *,
        doc_context: str | None = None,
    ) -> list[Knot]:
        if not section_text.strip():
            return []

        claims: list[Knot] = []

        # Step 1: Extract code blocks, replace with placeholders, process prose
        # segments between them.
        parts: list[tuple[str, bool]] = []  # (text, is_code)
        last_end = 0
        for m in _CODE_FENCE_RE.finditer(section_text):
            # Prose before this code block
            if m.start() > last_end:
                parts.append((section_text[last_end:m.start()], False))
            parts.append((m.group(0), True))
            last_end = m.end()
        # Trailing prose after last code block (or entire text if no blocks)
        if last_end < len(section_text):
            parts.append((section_text[last_end:], False))

        # Step 2: Process each part
        idx = 0
        for text, is_code in parts:
            if is_code:
                cleaned = text.strip()
                if cleaned:
                    claims.append(Knot(
                        text=cleaned,
                        metadata={"index": idx, "is_code": True},
                    ))
                    idx += 1
            else:
                # Split prose by sentence boundaries and newlines
                sentences = self._split_prose(text)
                for sent in sentences:
                    claims.append(Knot(
                        text=sent,
                        metadata={"index": idx},
                    ))
                    idx += 1

        return claims

    def _split_prose(self, text: str) -> list[str]:
        """Split prose text into sentences, filtering blanks."""
        # First split by newlines
        lines = text.split("\n")
        sentences: list[str] = []
        for line in lines:
            # Split each line by sentence-ending punctuation
            parts = _SENTENCE_END_RE.split(line)
            for part in parts:
                cleaned = _collapse_whitespace(part)
                if cleaned:
                    sentences.append(cleaned)
        return sentences
