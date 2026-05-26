"""InboxStore — JSONL-backed pending-edge queue.

Schema per entry: see spec section 4 (PrismRag v5.2). Append-only for new
entries; status updates rewrite the whole file via atomic_write.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prism_rag.utils.io import atomic_write


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class InboxEntry:
    id: str
    source: str                      # semantic direction: nimbus::doc
    target: str                      # semantic direction: code::file::Symbol
    edge_kind: str                   # always "mentions_symbol" in v5.2
    confidence: float
    confidence_tier: str
    model_id: str
    probe_signals: list[dict[str, Any]]
    top_k_rank: int
    status: str                      # pending | approved | rejected | auto_promoted | discarded
    created_at: str
    decided_at: str | None = None
    decided_by: str | None = None
    decision_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TERMINAL_STATUSES = frozenset({"approved", "rejected", "auto_promoted", "discarded"})


class StatusTransitionError(ValueError):
    pass


class InboxStore:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._entries: list[dict[str, Any]] = []
        self._index: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            self._entries.append(d)
            self._index[d["id"]] = d

    def get(self, edge_id: str) -> dict[str, Any] | None:
        return self._index.get(edge_id)

    def append(self, entry: InboxEntry) -> None:
        d = entry.to_dict()
        if d["id"] in self._index:
            raise ValueError(f"duplicate id: {d['id']}")
        self._entries.append(d)
        self._index[d["id"]] = d

    def update_pending(self, edge_id: str, new_data: dict[str, Any]) -> None:
        existing = self._index.get(edge_id)
        if existing is None:
            raise KeyError(edge_id)
        if existing["status"] != "pending":
            raise StatusTransitionError(
                f"cannot update non-pending entry {edge_id} (status={existing['status']})"
            )
        for k in ("confidence", "probe_signals", "top_k_rank", "model_id"):
            if k in new_data:
                existing[k] = new_data[k]

    def set_status(
        self, edge_id: str, new_status: str, *,
        decided_by: str, decision_note: str = "",
    ) -> None:
        e = self._index.get(edge_id)
        if e is None:
            raise KeyError(edge_id)
        if e["status"] in _TERMINAL_STATUSES:
            raise StatusTransitionError(
                f"cannot transition {edge_id} from {e['status']} to {new_status}"
            )
        e["status"] = new_status
        e["decided_at"] = now_iso()
        e["decided_by"] = decided_by
        e["decision_note"] = decision_note

    def list_pending(self, top_n: int = 10, sort_by: str = "confidence") -> list[dict[str, Any]]:
        pending = [e for e in self._entries if e["status"] == "pending"]
        if sort_by == "confidence":
            pending.sort(key=lambda e: e["confidence"], reverse=True)
        elif sort_by == "created_at":
            pending.sort(key=lambda e: e["created_at"], reverse=True)
        elif sort_by == "consecutive_seen":
            pending.sort(
                key=lambda e: max((s.get("consecutive_seen", 0) for s in e["probe_signals"]), default=0),
                reverse=True,
            )
        return pending[:top_n]

    def list_all(self, status: str | None = None, top_n: int = 50) -> list[dict[str, Any]]:
        rows = self._entries if status is None else [e for e in self._entries if e["status"] == status]
        return rows[:top_n]

    def save_atomic(self) -> None:
        content = "\n".join(json.dumps(e, ensure_ascii=False) for e in self._entries)
        atomic_write(self._path, content + ("\n" if content else ""))
