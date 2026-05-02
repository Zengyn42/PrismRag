"""Cross-namespace edge tracker for FederatedGraph bridge edges.

Every time FederatedGraph.build_bridges() creates a bridge edge between two
different namespaces, it calls CrossNamespaceProbe.on_edge_created(). The
probe accumulates these entries in memory and optionally persists them to a
JSON-lines file for survival across server restarts.

Query API:
  list_cross_edges(min_confidence, allowed_tiers)
  list_new_cross_edges(since)
  list_edges_from_node(node_id)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CrossEdgeEntry:
    edge_id: str           # "{src_ns}::{src_id}→{tgt_ns}::{tgt_id}"
    source_node: str       # "{namespace}::{node_id}"
    target_node: str       # "{namespace}::{node_id}"
    edge_kind: str         # "shared_tag" | "embedding_similar" | ...
    confidence_tier: str   # "EXTRACTED" | "INFERRED" | "AMBIGUOUS"
    confidence: float      # [0.0, 1.0]
    first_seen_at: str     # ISO 8601
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CrossEdgeEntry":
        return cls(**d)


_VALID_TIERS = {"EXTRACTED", "INFERRED", "AMBIGUOUS"}


class CrossNamespaceProbe:
    """Tracks cross-namespace bridge edges created by FederatedGraph.

    Usage::

        probe = CrossNamespaceProbe(log_path=data_dir / "cross_namespace_log.jsonl")
        fg.build_bridges(stores=stores, probe=probe)

        # Query
        edges = probe.list_cross_edges(min_confidence=0.7)
        new   = probe.list_new_cross_edges(since=datetime(2026, 5, 1))
        hits  = probe.list_edges_from_node("code::framework/nodes/llm/claude.py")
    """

    def __init__(self, log_path: Path | None = None) -> None:
        self._entries: list[CrossEdgeEntry] = []
        self._seen_ids: set[str] = set()
        self._log_path = log_path
        if log_path is not None:
            self._load_from_disk()

    # ── Write ─────────────────────────────────────────────────────────────

    def on_edge_created(self, bridge: dict) -> None:
        """Record one bridge dict from FederatedGraph._bridges.

        Bridge dict schema (from build_bridges):
          source_ns, source_id, target_ns, target_id,
          relation, confidence (tier str), weight (float)
        """
        src_ns = bridge.get("source_ns", "")
        src_id = bridge.get("source_id", "")
        tgt_ns = bridge.get("target_ns", "")
        tgt_id = bridge.get("target_id", "")

        if src_ns == tgt_ns:
            return  # within-namespace bridge — not a cross-namespace edge

        edge_id = f"{src_ns}::{src_id}→{tgt_ns}::{tgt_id}"
        if edge_id in self._seen_ids:
            return  # idempotent

        tier = bridge.get("confidence", "INFERRED")
        if tier not in _VALID_TIERS:
            tier = "INFERRED"
        conf = float(bridge.get("weight", 0.7))

        entry = CrossEdgeEntry(
            edge_id=edge_id,
            source_node=f"{src_ns}::{src_id}",
            target_node=f"{tgt_ns}::{tgt_id}",
            edge_kind=bridge.get("relation", "bridge"),
            confidence_tier=tier,
            confidence=conf,
            first_seen_at=datetime.now(timezone.utc).isoformat(),
            evidence=bridge.get("evidence", []),
        )
        self._entries.append(entry)
        self._seen_ids.add(edge_id)

        if self._log_path is not None:
            self._append_to_disk(entry)

    # ── Query ─────────────────────────────────────────────────────────────

    def list_cross_edges(
        self,
        min_confidence: float = 0.0,
        allowed_tiers: list[str] | None = None,
    ) -> list[CrossEdgeEntry]:
        """Return all recorded cross-namespace edges, with optional filters."""
        result = self._entries
        if min_confidence > 0.0:
            result = [e for e in result if e.confidence >= min_confidence]
        if allowed_tiers is not None:
            tiers = set(allowed_tiers)
            result = [e for e in result if e.confidence_tier in tiers]
        return result

    def list_new_cross_edges(self, since: datetime) -> list[CrossEdgeEntry]:
        """Return edges first seen at or after `since`."""
        return [
            e for e in self._entries
            if datetime.fromisoformat(e.first_seen_at) >= since
        ]

    def list_edges_from_node(self, node_id: str) -> list[CrossEdgeEntry]:
        """Return all edges touching `node_id` (qualified or bare ID match)."""
        return [
            e for e in self._entries
            if node_id in e.source_node or node_id in e.target_node
        ]

    @property
    def total(self) -> int:
        return len(self._entries)

    # ── Persistence ────────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        if self._log_path is None or not self._log_path.exists():
            return
        try:
            for line in self._log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = CrossEdgeEntry.from_dict(json.loads(line))
                if entry.edge_id not in self._seen_ids:
                    self._entries.append(entry)
                    self._seen_ids.add(entry.edge_id)
            logger.info(
                f"[cross_ns_probe] loaded {len(self._entries)} entries from {self._log_path}"
            )
        except Exception as exc:
            logger.warning(f"[cross_ns_probe] failed to load log: {exc}")

    def _append_to_disk(self, entry: CrossEdgeEntry) -> None:
        if self._log_path is None:
            return
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(f"[cross_ns_probe] failed to append entry: {exc}")
