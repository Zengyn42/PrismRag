"""Deduplication decision log for atomize_propose / atomize_apply.

Decisions are appended as JSONL to dedup_log.jsonl (same dir as inbox.jsonl).
Each line is a DedupSnapshot serialised to JSON.

Usage:
    from prism_rag.ingest.dedup_log import DedupSnapshot, write_snapshot, list_snapshots

    snap = DedupSnapshot(
        decision_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        action='reuse',
        claim_title='Fresh-per-call design decision',
        reused_id='KNOW-000043',
        source_doc='设计细节/PrismRag v5.3.md',
        similarity_score=0.94,
        pre_state={'mentions_edges': []},
    )
    write_snapshot(log_path, snap)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DedupSnapshot:
    """Immutable record of one deduplication decision."""

    # Unique ID for this decision (UUID4).
    decision_id: str
    # ISO-8601 UTC timestamp.
    timestamp: str
    # 'reuse' — an existing KNOW node was reused instead of creating a new one.
    # 'create' — no similar node found; new KNOW node was created normally.
    action: str  # Literal['reuse', 'create']
    # Human-readable title of the claim that triggered this decision.
    claim_title: str
    # Source document that was being atomized (vault-relative path).
    source_doc: str
    # Cosine similarity score that triggered the decision.
    similarity_score: float
    # Graph edges/nodes state before the decision (for rollback).
    pre_state: dict[str, Any] = field(default_factory=dict)
    # KNOW-ID of the existing node that was reused (None for action='create').
    reused_id: str | None = None
    # Optional rollback status set by rollback_snapshot().
    rollback_status: str | None = None


def write_snapshot(log_path: Path, snapshot: DedupSnapshot) -> None:
    """Append a DedupSnapshot to the JSONL log at log_path (created if absent)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(snapshot), ensure_ascii=False)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    logger.debug(f"[dedup_log] wrote snapshot {snapshot.decision_id} action={snapshot.action}")


def list_snapshots(log_path: Path) -> list[DedupSnapshot]:
    """Return all snapshots from the JSONL log, oldest first.

    Missing or empty log returns an empty list (no error).
    Malformed lines are skipped with a warning.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    snapshots: list[DedupSnapshot] = []
    for lineno, raw in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            d = json.loads(raw)
            snapshots.append(DedupSnapshot(**d))
        except Exception as exc:
            logger.warning(f"[dedup_log] skipping malformed line {lineno} in {log_path}: {exc}")
    return snapshots


def rollback_snapshot(
    decision_id: str,
    log_path: Path,
    vault_root: Path,
    graph_path: Path,
) -> str:
    """Undo the graph effects of a 'reuse' decision.

    For action='reuse', this removes:
    - The MENTIONS edge from source_doc → reused_id
    - Any CONTEXT_REF nodes created during that apply run

    Saves the updated graph and marks the snapshot as rolled_back in the log.

    Args:
        decision_id: UUID of the DedupSnapshot to roll back.
        log_path: Path to dedup_log.jsonl.
        vault_root: Vault root (for resolving source_doc paths).
        graph_path: Path to graph.json to modify.

    Returns:
        Human-readable status string.
    """
    snapshots = list_snapshots(log_path)
    target: DedupSnapshot | None = None
    for snap in snapshots:
        if snap.decision_id == decision_id:
            target = snap
            break

    if target is None:
        return f"rollback_snapshot: decision_id {decision_id!r} not found in {log_path}"

    if target.action != "reuse":
        return (
            f"rollback_snapshot: action={target.action!r} — only 'reuse' decisions "
            "have graph effects; nothing to roll back."
        )

    if target.rollback_status == "rolled_back":
        return f"rollback_snapshot: {decision_id!r} already rolled back, skipping."

    if not graph_path.exists():
        return f"rollback_snapshot: graph not found at {graph_path}, snapshot NOT rolled back."

    # Load graph and remove the MENTIONS edge + any CONTEXT_REF nodes.
    try:
        from prism_rag.store.graph import KnowledgeGraph

        graph = KnowledgeGraph.load(graph_path)
        removed_edges = 0
        removed_nodes = 0

        source_doc = target.source_doc
        reused_id = target.reused_id

        # Remove MENTIONS edge (source_doc → reused_id with source_pass='dedup').
        if (
            reused_id
            and graph.g.has_node(source_doc)
            and graph.g.has_node(reused_id)
            and graph.g.has_edge(source_doc, reused_id)
        ):
            edge_data = graph.g.edges[source_doc, reused_id]
            if edge_data.get("source_pass") == "dedup":
                graph.g.remove_edge(source_doc, reused_id)
                removed_edges += 1
                logger.info(
                    f"[dedup_log] rollback: removed MENTIONS edge {source_doc} → {reused_id}"
                )

        # Remove CONTEXT_REF nodes linked to this source_doc + reused_id.
        ctx_nodes_to_remove = [
            nid
            for nid, data in list(graph.g.nodes(data=True))
            if (
                data.get("kind") == "context_ref"
                and data.get("source_file") == source_doc
                and (reused_id is None or nid.startswith(f"CONTEXT_REF_{reused_id}_"))
            )
        ]
        for nid in ctx_nodes_to_remove:
            graph.g.remove_node(nid)
            removed_nodes += 1
            logger.info(f"[dedup_log] rollback: removed CONTEXT_REF node {nid}")

        graph.save(graph_path)
    except Exception as exc:
        logger.error(f"[dedup_log] rollback failed for {decision_id}: {exc}")
        return f"rollback_snapshot: ERROR — {exc}"

    # Mark snapshot as rolled_back in the log.
    _mark_rolled_back(log_path, decision_id)

    return (
        f"rollback_snapshot: OK — removed {removed_edges} edge(s) and "
        f"{removed_nodes} CONTEXT_REF node(s) for decision {decision_id!r}."
    )


def _mark_rolled_back(log_path: Path, decision_id: str) -> None:
    """Rewrite the log with the matched snapshot's rollback_status set to 'rolled_back'."""
    snapshots = list_snapshots(log_path)
    lines: list[str] = []
    for snap in snapshots:
        if snap.decision_id == decision_id:
            snap.rollback_status = "rolled_back"
        lines.append(json.dumps(asdict(snap), ensure_ascii=False))

    # Atomic rewrite
    import os
    import tempfile

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=log_path.parent, prefix=".tmp-dedup-", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n" if lines else "")
        os.replace(tmp, log_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
