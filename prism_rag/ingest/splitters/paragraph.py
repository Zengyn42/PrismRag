"""ParagraphSplitter — splits text at blank-line boundaries.

Each paragraph (separated by one or more blank lines) becomes a single
Knot. Empty paragraphs are skipped. Handles both LF and CRLF.
"""

from __future__ import annotations

import re

from prism_rag.ingest.splitters.base import Knot, Splitter

# One or more blank lines (possibly containing only whitespace).
# Handles both \n and \r\n line endings.
_BLANK_LINE_RE = re.compile(r"(?:\r?\n){2,}|\r?\n(?:[ \t]*\r?\n)+")


class ParagraphSplitter(Splitter):
    """Splits text into paragraphs separated by blank lines.

    Rules:
      - Blank lines (>=1 consecutive empty/whitespace-only lines) act as
        paragraph boundaries.
      - Leading/trailing whitespace is stripped from each paragraph.
      - Empty paragraphs are discarded.
      - Fully empty/whitespace input returns [].
    """

    @property
    def name(self) -> str:
        return "paragraph"

    def split(
        self,
        section_text: str,
        *,
        doc_context: str | None = None,
    ) -> list[Knot]:
        if not section_text.strip():
            return []

        paragraphs = _BLANK_LINE_RE.split(section_text)
        claims: list[Knot] = []
        for para in paragraphs:
            text = para.strip()
            if text:
                claims.append(Knot(text=text))
        return claims
