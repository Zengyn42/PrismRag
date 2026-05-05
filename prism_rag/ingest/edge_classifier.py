"""EdgeClassifier — three-tier promotion logic on top of CrossNamespaceProbe.

See spec §三 / §五 for tier definitions and probe interaction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from prism_rag.config import ClassifierProfile, GraphSource
from prism_rag.inbox.store import InboxEntry, InboxStore
from prism_rag.store.cross_namespace_probe import CrossEdgeEntry, CrossNamespaceProbe, MIGRATION_PENDING
from prism_rag.store.federated import FederatedGraph
from prism_rag.store.graph import Edge, LifecycleClass


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


# ── classify_and_route ────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ClassifyReport:
    promoted: int = 0
    queued: int = 0
    rolled_back: int = 0
    discarded: int = 0


def _entry_id(probe_entry: CrossEdgeEntry) -> str:
    sem_src = probe_entry.target_node   # nimbus::doc
    sem_tgt = probe_entry.source_node   # code::file::Symbol
    return (
        f"{sem_src.replace('::', '__').replace('/', '_')}"
        f"__to__"
        f"{sem_tgt.replace('::', '__').replace('/', '_')}"
    )


def classify_and_route(
    fg: FederatedGraph,
    probe: CrossNamespaceProbe,
    inbox: InboxStore,
    nimbus_src: GraphSource,
    profile: ClassifierProfile,
) -> ClassifyReport:
    """Walk all probe entries, classify, and route to Tier 1/2/3."""
    report = ClassifyReport()

    # Group by probe source_node (the code symbol) for top-K determination.
    by_src: dict[str, list[CrossEdgeEntry]] = defaultdict(list)
    for entry in probe._index.values():
        by_src[entry.source_node].append(entry)
    for entries in by_src.values():
        entries.sort(key=lambda e: e.confidence, reverse=True)

    for entries in by_src.values():
        # Tier 2 candidate set per source: margin + cap
        candidates = select_tier2_candidates(
            entries, margin=profile.tier2_margin, hard_cap=profile.tier2_hard_cap
        )
        candidate_ids = {c.edge_id for c in candidates}

        for rank, entry in enumerate(entries):
            is_top_1 = rank == 0
            tier = classify_one(entry, is_top_1=is_top_1, profile=profile)

            if tier == TIER_1:
                _promote_to_tier1(fg, probe, entry, inbox, nimbus_src)
                report.promoted += 1
            elif tier == TIER_2 and entry.edge_id in candidate_ids:
                action = _upsert_inbox(inbox, entry, top_k_rank=rank + 1)
                if action == "added":
                    report.queued += 1
            else:
                action = _maybe_discard(inbox, entry)
                if action == "rolled_back":
                    report.rolled_back += 1
                elif action == "noop":
                    report.discarded += 1
                # "skipped" = StatusTransitionError, don't count

    return report


def _promote_to_tier1(
    fg: FederatedGraph,
    probe: CrossNamespaceProbe,
    entry: CrossEdgeEntry,
    inbox: InboxStore,
    nimbus_src: GraphSource,
) -> None:
    sem_src = entry.target_node     # e.g. "nimbus::doc"
    sem_tgt = entry.source_node     # e.g. "code::a.py::Foo"
    # bare_src: strip the namespace prefix so the edge lives in the nimbus graph
    bare_src = sem_src.split("::", 1)[1] if "::" in sem_src else sem_src
    nimbus = fg.get_graph(nimbus_src.namespace)
    edge = Edge(
        source=bare_src,
        target=sem_tgt,
        relation="mentions_symbol",
        confidence="INFERRED",
        confidence_score=entry.confidence,
        source_pass="conv",
        lifecycle_class=LifecycleClass.ANCHORED,
    )
    nimbus.add_edge(edge)
    # Mark probe entry ANCHORED so future sweeps won't re-classify it.
    probe._index[entry.edge_id].lifecycle_class = LifecycleClass.ANCHORED
    # If inbox has a pending entry for this, mark it auto_promoted.
    inbox_id = _entry_id(entry)
    existing = inbox.get(inbox_id)
    if existing is not None and existing["status"] == "pending":
        try:
            inbox.set_status(
                inbox_id, "auto_promoted",
                decided_by="classifier",
                decision_note=f"Tier 1 (consecutive={entry.consecutive_seen})",
            )
        except Exception:
            pass


def _upsert_inbox(inbox: InboxStore, entry: CrossEdgeEntry, top_k_rank: int) -> str:
    inbox_id = _entry_id(entry)
    existing = inbox.get(inbox_id)
    if existing is not None:
        if existing["status"] == "pending":
            return "updated"
        return "skipped_terminal"
    inbox.append(InboxEntry(
        id=inbox_id,
        source=entry.target_node,
        target=entry.source_node,
        edge_kind="mentions_symbol",
        confidence=entry.confidence,
        confidence_tier=entry.confidence_tier,
        model_id=entry.model_id,
        probe_signals=[{
            "kind": entry.edge_kind,
            "score": entry.confidence,
            "consecutive_seen": entry.consecutive_seen,
            "first_seen_at": entry.first_seen_at,
            "last_seen_at": entry.last_seen_at or entry.first_seen_at,
        }],
        top_k_rank=top_k_rank,
        status="pending",
        created_at=_now_iso(),
    ))
    return "added"


def _maybe_discard(inbox: InboxStore, entry: CrossEdgeEntry) -> str:
    inbox_id = _entry_id(entry)
    existing = inbox.get(inbox_id)
    if existing is not None and existing["status"] == "pending":
        try:
            inbox.set_status(
                inbox_id, "discarded",
                decided_by="classifier",
                decision_note=f"Tier 3 rollback (conf={entry.confidence})",
            )
            return "rolled_back"
        except Exception:
            return "skipped"
    return "noop"
