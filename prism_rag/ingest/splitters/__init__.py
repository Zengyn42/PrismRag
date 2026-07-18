"""Pluggable splitter abstractions for knowledge atomization.

A Splitter takes a section of text and breaks it into atomic claims
suitable for KNOT (Knowledge Ontology Token) creation.

This package defines the protocol; concrete implementations live as
submodules (e.g. passthrough, molecular_facts, …).
"""

from prism_rag.ingest.splitters.base import AtomicClaim, Knot, PassthroughSplitter, Splitter
from prism_rag.ingest.splitters.fixed_window import FixedWindowSplitter
from prism_rag.ingest.splitters.llm import LlmSplitter
from prism_rag.ingest.splitters.paragraph import ParagraphSplitter
from prism_rag.ingest.splitters.registry import (
    SPLITTER_REGISTRY,
    get_splitter,
    list_splitters,
    register_splitter,
)
from prism_rag.ingest.splitters.sentence import SentenceSplitter

__all__ = [
    "AtomicClaim",
    "Knot",
    "FixedWindowSplitter",
    "LlmSplitter",
    "ParagraphSplitter",
    "PassthroughSplitter",
    "SPLITTER_REGISTRY",
    "SentenceSplitter",
    "Splitter",
    "get_splitter",
    "list_splitters",
    "register_splitter",
]
