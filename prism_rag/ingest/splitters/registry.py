"""Splitter registry — name-based lookup and third-party registration.

Built-in splitters (passthrough, sentence) are registered at import time.
Third-party splitters can be added via :func:`register_splitter`.

Usage::

    from prism_rag.ingest.splitters.registry import get_splitter

    s = get_splitter("sentence")
    claims = s.split("Hello world. Goodbye world.")
"""

from __future__ import annotations

from prism_rag.ingest.splitters.base import PassthroughSplitter, Splitter
from prism_rag.ingest.splitters.fixed_window import FixedWindowSplitter
from prism_rag.ingest.splitters.gleanings import GleaningsSplitter
from prism_rag.ingest.splitters.llm import LlmSplitter
from prism_rag.ingest.splitters.paragraph import ParagraphSplitter
from prism_rag.ingest.splitters.sentence import SentenceSplitter

# Global registry: name -> Splitter subclass
SPLITTER_REGISTRY: dict[str, type[Splitter]] = {
    "fixed_window": FixedWindowSplitter,
    "llm": LlmSplitter,
    "llm_gleanings": GleaningsSplitter,
    "paragraph": ParagraphSplitter,
    "passthrough": PassthroughSplitter,
    "sentence": SentenceSplitter,
}


def get_splitter(name: str, **kwargs) -> Splitter:
    """Instantiate a registered splitter by name.

    Args:
        name: Registered splitter name (e.g. ``"passthrough"``, ``"sentence"``).
        **kwargs: Forwarded to the splitter constructor.

    Returns:
        A :class:`Splitter` instance.

    Raises:
        ValueError: If *name* is not in the registry.  The error message
            lists all available splitter names.
    """
    cls = SPLITTER_REGISTRY.get(name)
    if cls is None:
        available = sorted(SPLITTER_REGISTRY)
        raise ValueError(
            f"Unknown splitter {name!r}. "
            f"Available: {', '.join(available)}"
        )
    return cls(**kwargs)


def list_splitters() -> list[str]:
    """Return a sorted list of all registered splitter names."""
    return sorted(SPLITTER_REGISTRY)


def register_splitter(cls: type[Splitter]) -> type[Splitter]:
    """Register a Splitter subclass by its ``name`` property.

    Can be used as a decorator or called directly::

        @register_splitter
        class MySplitter(Splitter):
            @property
            def name(self) -> str:
                return "my_splitter"
            ...

    Args:
        cls: A concrete :class:`Splitter` subclass.

    Returns:
        The class unchanged (so it works as a decorator).

    Raises:
        ValueError: If a splitter with the same name is already registered.
    """
    instance = cls()
    splitter_name = instance.name
    if splitter_name in SPLITTER_REGISTRY:
        raise ValueError(
            f"Splitter {splitter_name!r} is already registered "
            f"(by {SPLITTER_REGISTRY[splitter_name].__name__})"
        )
    SPLITTER_REGISTRY[splitter_name] = cls
    return cls
