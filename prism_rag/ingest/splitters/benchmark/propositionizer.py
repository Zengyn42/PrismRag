"""Load Dense X Propositionizer benchmark dataset from HuggingFace.

Dataset: chentong00/propositionizer-wiki-data (44,857 examples)
Format: {sources: "Title: ... Section: ... Content: ...", targets: "[prop1, prop2, ...]"}

The test split (1,000 examples) is used by default — large enough for
reliable statistics, fast enough for routine benchmarking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from prism_rag.ingest.splitters.base import Knot
from prism_rag.ingest.splitters.benchmark.dataset import BenchmarkCase

logger = logging.getLogger(__name__)

_DATASET_ID = "chentong00/propositionizer-wiki-data"


def _parse_source(raw: str) -> tuple[str, str]:
    """Extract (doc_context, section_text) from propositionizer source format.

    Format: "Title: X. Section: Y. Content: Z"
    Returns (context_string, content_text).
    """
    # Extract title
    title = ""
    section = ""
    content = raw

    if raw.startswith("Title: "):
        # Find "Section: " marker
        sec_idx = raw.find(". Section: ")
        if sec_idx != -1:
            title = raw[len("Title: "):sec_idx]
            remainder = raw[sec_idx + len(". Section: "):]
            # Find "Content: " or ". " after section
            # Section can be empty (just ". Content: ...")
            cont_markers = [". Content: ", "Content: "]
            for marker in cont_markers:
                cont_idx = remainder.find(marker)
                if cont_idx != -1:
                    section = remainder[:cont_idx].strip(". ")
                    content = remainder[cont_idx + len(marker):]
                    break
            else:
                content = remainder
        else:
            content = raw

    ctx_parts = []
    if title:
        ctx_parts.append(f"Title: {title}")
    if section:
        ctx_parts.append(f"Section: {section}")
    doc_context = ". ".join(ctx_parts) if ctx_parts else None

    return doc_context, content


def load_propositionizer_dataset(
    *,
    split: str = "test",
    max_cases: int | None = None,
) -> list[BenchmarkCase]:
    """Load benchmark cases from the Dense X Propositionizer dataset.

    Args:
        split: Dataset split to load ("train", "validation", "test").
            Default "test" (1,000 examples).
        max_cases: Maximum number of cases to load. None = all.

    Returns:
        List of BenchmarkCase with gold-standard propositions as
        reference_knots.

    Raises:
        ImportError: If the ``datasets`` library is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for loading the Propositionizer "
            "dataset. Install with: pip install datasets"
        )

    logger.info(f"[propositionizer] loading split={split} from {_DATASET_ID}")
    ds = load_dataset(_DATASET_ID, split=split)

    cases: list[BenchmarkCase] = []
    for i, row in enumerate(ds):
        if max_cases is not None and i >= max_cases:
            break

        doc_context, section_text = _parse_source(row["sources"])

        # Parse gold propositions
        try:
            props = json.loads(row["targets"])
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"[propositionizer] row {i}: unparseable targets, skipping")
            continue

        if not isinstance(props, list) or not props:
            continue

        reference_knots = [
            Knot(text=str(p), method="gold_propositionizer")
            for p in props
            if isinstance(p, str) and p.strip()
        ]

        if not section_text.strip():
            continue

        cases.append(BenchmarkCase(
            section_text=section_text,
            doc_context=doc_context,
            reference_knots=reference_knots,
            source=f"propositionizer-wiki/{split}:{i}",
        ))

    logger.info(f"[propositionizer] loaded {len(cases)} cases from {split}")
    return cases
