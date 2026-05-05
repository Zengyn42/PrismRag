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


TIER_1 = 1
TIER_2 = 2
TIER_3 = 3


def classify_one(
    entry: CrossEdgeEntry, *, is_top_1: bool, profile: ClassifierProfile,
) -> int:
    """Decide tier for one probe entry. See spec §三."""
    from prism_rag.store.cross_namespace_probe import MIGRATION_PENDING
    from prism_rag.store.graph import LifecycleClass

    # MIGRATION_PENDING short-circuit: never promote unverified entries
    if entry.last_seen_parsed_at == MIGRATION_PENDING:
        return TIER_3
    # ANCHORED: already promoted, don't re-classify
    if entry.lifecycle_class == LifecycleClass.ANCHORED:
        return TIER_3

    # Tier 1: high confidence + top-1 + stable
    if (
        entry.confidence >= profile.tier1_min_conf
        and is_top_1
        and entry.consecutive_seen >= profile.tier1_min_consecutive
    ):
        return TIER_1
    # Tier 2: above floor + (top-K OR stable)
    if (
        entry.confidence >= profile.tier2_min_conf
        and (is_top_1 or entry.consecutive_seen >= profile.tier2_min_consecutive)
    ):
        return TIER_2
    return TIER_3
