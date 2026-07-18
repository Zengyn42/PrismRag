"""Base splitter protocol and data models for knowledge atomization.

A Splitter converts a block of text (typically one vault-document section)
into a list of Knot objects (KNOT — Knowledge Ontology Token, 知识元).
The Knot is the ONE canonical atomized type: every atomization method
outputs list[Knot], and the atomize pipeline turns Knots into KNOT nodes.

The interface is intentionally minimal so new splitting strategies
(rule-based, LLM-based, Molecular Facts, …) can be benchmarked against
each other using the same harness.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Knot:
    """KNOT — Knowledge Ontology Token (知识元).

    The ONE canonical output type of knowledge atomization. Every
    atomization method (rule-based splitters, LLM propose, Molecular
    Facts, …) MUST output ``list[Knot]`` so methods are interchangeable
    and benchmarkable against each other.

    Attributes:
        text: The atomic knowledge body — a self-contained statement.
        title: Optional short label for the knot (used as KNOT node
            title in the graph; LLM splitters fill this, rule-based
            splitters may leave it empty).
        ontology_type: Ontology category (e.g. "concept", "decision",
            "procedure", "fact"). Defaults to "fact" for rule-based
            splitters; the atomize pipeline may reclassify.
        source_section_id: Reference back to the originating section
            (e.g. atomize_scan section_id).
        context_note: Contextual supplement in the style of Molecular
            Facts decontextualization — extra context that makes the
            knot understandable without the source document.
        method: Name of the splitter/method that produced this knot
            (filled by the pipeline, useful in benchmarks).
        metadata: Arbitrary key-value pairs for method-specific info
            (e.g. confidence score, window index, token count).

    Note: ``knowledge_id`` (KNOW-ID) is deliberately NOT a field here —
    ID assignment/routing is the atomize pipeline's responsibility at
    apply time, never the splitter's.
    """

    text: str
    title: str = ""
    ontology_type: str = "fact"
    source_section_id: str | None = None
    context_note: str | None = None
    method: str = ""
    metadata: dict = field(default_factory=dict)

    def to_claim_dict(self, *, section_id: str | None = None) -> dict:
        """Adapter → the claim dict shape consumed by atomize_propose_impl.

        (keys: section_id, title, body, ontology_type; knowledge_id is
        assigned later by the pipeline's KNOW-ID routing.)
        """
        return {
            "section_id": section_id or self.source_section_id or "",
            "title": self.title,
            "body": self.text,
            "ontology_type": self.ontology_type,
        }


# Backward-compat alias — original name before KNOT unification (2026-07-17).
AtomicClaim = Knot


class Splitter(ABC):
    """Abstract base class for all text splitters.

    Subclasses must implement :meth:`split` and the :attr:`name` property.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this splitter (used in benchmark comparisons)."""
        ...

    @abstractmethod
    def split(
        self,
        section_text: str,
        *,
        doc_context: str | None = None,
    ) -> list[Knot]:
        """Split *section_text* into atomic claims.

        Args:
            section_text: The text to split (typically one document section).
            doc_context: Optional surrounding-document context that may help
                the splitter produce self-contained claims (e.g. document
                title, preceding headings).

        Returns:
            A list of :class:`Knot` instances.  An empty input must
            return an empty list.
        """
        ...


class PassthroughSplitter(Splitter):
    """Minimal reference splitter — returns the entire section as one claim.

    Useful for testing that the Splitter interface is wired up correctly
    and as a no-op baseline in benchmarks.
    """

    @property
    def name(self) -> str:
        return "passthrough"

    def split(
        self,
        section_text: str,
        *,
        doc_context: str | None = None,
    ) -> list[Knot]:
        if not section_text.strip():
            return []
        return [Knot(text=section_text)]
