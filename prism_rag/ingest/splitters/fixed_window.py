"""FixedWindowSplitter — fixed-size sliding window chunker.

Splits text into chunks of at most ``window_size`` characters with
``overlap`` characters shared between adjacent chunks.  Useful as a
simple baseline for embedding-oriented pipelines where uniform chunk
length matters more than linguistic boundaries.
"""

from __future__ import annotations

from prism_rag.ingest.splitters.base import Knot, Splitter


class FixedWindowSplitter(Splitter):
    """Splits text into fixed-size overlapping windows.

    Args:
        window_size: Maximum characters per chunk (default 400).
        overlap: Characters shared between adjacent chunks (default 50).

    Raises:
        ValueError: If ``overlap >= window_size``.
    """

    def __init__(self, window_size: int = 400, overlap: int = 50):
        if overlap >= window_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than window_size ({window_size})"
            )
        self._window_size = window_size
        self._overlap = overlap

    @property
    def name(self) -> str:
        return "fixed_window"

    def split(
        self,
        section_text: str,
        *,
        doc_context: str | None = None,
    ) -> list[Knot]:
        if not section_text.strip():
            return []

        step = self._window_size - self._overlap
        claims: list[Knot] = []
        idx = 0
        pos = 0
        while pos < len(section_text):
            chunk = section_text[pos : pos + self._window_size]
            text = chunk.strip()
            if text:
                claims.append(Knot(
                    text=text,
                    metadata={"window_index": idx},
                ))
                idx += 1
            pos += step

        return claims
