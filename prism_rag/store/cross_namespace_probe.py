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

from prism_rag.store.graph import LifecycleClass

MIGRATION_PENDING = "MIGRATION_PENDING"

logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    last_seen_parsed_at: str = ""
    source_file: str = ""
    consecutive_seen: int = 1
    model_id: str = ""
    lifecycle_class: str = LifecycleClass.PROBABILISTIC

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

    def __init__(
        self,
        log_path: Path | None = None,
        model_id: str = "",
    ) -> None:
        self._entries: list[CrossEdgeEntry] = []
        self._seen_ids: set[str] = set()
        self._index: dict[str, CrossEdgeEntry] = {}
        self._model_id = model_id
        self._log_path = log_path
        if log_path is not None:
            self._load_from_disk()

    # ── Write ─────────────────────────────────────────────────────────────

    def record(self, bridge: dict, scan_timestamp: str) -> None:
        """Record one bridge dict observed during a scan.

        Five-branch state machine + ANCHORED short-circuit:
          1. ANCHORED  → only refresh visibility (no parsed-at, no consec change).
          2. New edge  → create with consecutive=1.
          3. MIGRATION_PENDING sentinel → overwrite with real scan_timestamp.
          4. Model changed → reset consecutive=1, update confidence + model_id.
          5. Same model, new scan_timestamp → consecutive++.
          6. Same model, same scan_timestamp → no-op.
        """
        src_ns = bridge.get("source_ns", "")
        src_id = bridge.get("source_id", "")
        tgt_ns = bridge.get("target_ns", "")
        tgt_id = bridge.get("target_id", "")
        if src_ns == tgt_ns:
            return  # within-namespace bridge — not a cross-namespace edge

        edge_id = f"{src_ns}::{src_id}→{tgt_ns}::{tgt_id}"
        existing = self._index.get(edge_id)

        # Branch 1: ANCHORED short-circuit — keep parsed-at/consec frozen.
        if existing is not None and existing.lifecycle_class == LifecycleClass.ANCHORED:
            return

        if existing is None:
            # Branch 2: brand new edge.
            tier = bridge.get("confidence", "INFERRED")
            if tier not in _VALID_TIERS:
                tier = "INFERRED"
            entry = CrossEdgeEntry(
                edge_id=edge_id,
                source_node=f"{src_ns}::{src_id}",
                target_node=f"{tgt_ns}::{tgt_id}",
                edge_kind=bridge.get("relation", "bridge"),
                confidence_tier=tier,
                confidence=float(bridge.get("weight", 0.7)),
                first_seen_at=now_iso(),
                last_seen_parsed_at=scan_timestamp,
                source_file=bridge.get("source_file", ""),
                consecutive_seen=1,
                model_id=self._model_id,
                lifecycle_class=LifecycleClass.PROBABILISTIC,
                evidence=bridge.get("evidence", []),
            )
            self._index[edge_id] = entry
            # Mirror into legacy list so existing query APIs keep working.
            self._entries.append(entry)
            self._seen_ids.add(edge_id)
        elif existing.last_seen_parsed_at == MIGRATION_PENDING:
            # Branch 3: first real confirmation after migration sentinel.
            existing.last_seen_parsed_at = scan_timestamp
            existing.confidence = float(bridge.get("weight", 0.7))
            existing.model_id = self._model_id
            # consecutive_seen stays at 1 (first real confirmation)
        elif existing.model_id != self._model_id:
            # Branch 4: embedding model changed — reset streak.
            existing.consecutive_seen = 1
            existing.last_seen_parsed_at = scan_timestamp
            existing.model_id = self._model_id
            existing.confidence = float(bridge.get("weight", 0.7))
        elif existing.last_seen_parsed_at != scan_timestamp:
            # Branch 5: same model, new scan — bump streak.
            existing.consecutive_seen += 1
            existing.last_seen_parsed_at = scan_timestamp
        else:
            # Branch 6: same model, same scan — idempotent no-op.
            return

        if self._log_path is not None:
            self._rewrite_to_disk()

    # Backward-compat alias for existing callers (FederatedGraph.build_bridges).
    def on_edge_created(self, bridge: dict) -> None:
        """Legacy entrypoint — defaults scan_timestamp to now() for callers
        that haven't been updated to the new record() signature yet.
        """
        self.record(bridge, scan_timestamp=now_iso())

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
                    self._index[entry.edge_id] = entry
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

    def _rewrite_to_disk(self) -> None:
        """Rewrite the entire log atomically.

        Used by record() because the new state-machine semantics mutate
        existing entries (consecutive_seen, last_seen_parsed_at, …) and
        cannot be expressed as append-only.
        """
        if self._log_path is None:
            return
        try:
            from prism_rag.utils.io import atomic_write
            lines = "\n".join(
                json.dumps(e.to_dict(), ensure_ascii=False)
                for e in self._index.values()
            )
            content = (lines + "\n") if lines else ""
            atomic_write(self._log_path, content)
        except Exception as exc:
            logger.warning(f"[cross_ns_probe] failed to rewrite log: {exc}")
