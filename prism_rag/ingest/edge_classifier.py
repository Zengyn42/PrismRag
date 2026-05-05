"""EdgeClassifier — three-tier promotion logic on top of CrossNamespaceProbe.

See spec §三 / §五 for tier definitions and probe interaction.
"""

from __future__ import annotations

from dataclasses import dataclass

from prism_rag.config import ClassifierProfile
from prism_rag.store.cross_namespace_probe import CrossEdgeEntry


def select_tier2_candidates(
    ranked: list[CrossEdgeEntry], margin: float, hard_cap: int,
) -> list[CrossEdgeEntry]:
    """Margin + hard-cap selector. Replaces naive top-K.

    From a confidence-descending list, keep entries whose confidence is
    within `margin` of the top-1, then truncate to `hard_cap`.
    """
    if not ranked:
        return []
    threshold = ranked[0].confidence * (1.0 - margin)
    within = [e for e in ranked if e.confidence >= threshold]
    return within[:hard_cap]
