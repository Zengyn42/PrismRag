"""Benchmark case data model."""

from __future__ import annotations

from dataclasses import dataclass, field

from prism_rag.ingest.splitters.base import Knot


@dataclass
class BenchmarkCase:
    """A single benchmark case for evaluating splitter quality.

    Attributes:
        section_text: The input text to split.
        doc_context: Optional document-level context (e.g. title, headings).
        reference_knots: Gold-standard knots that an ideal splitter would
            produce. Used for qualitative comparison; scoring is heuristic
            and does not require exact match against these.
        source: Human-readable label describing where this case came from
            (e.g. "hand-crafted: simple fact paragraph").
    """

    section_text: str
    doc_context: str | None = None
    reference_knots: list[Knot] = field(default_factory=list)
    source: str = ""
