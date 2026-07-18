"""Drift-detection utilities for KNOT node lifecycle management.

Provides reusable functions for flagging knowledge nodes as "suspected"
when their source documentation references stale code symbols.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def flag_knots_suspected(
    graph: "KnowledgeGraph",
    doc_source_file: str,
    reason: str = "",
) -> list[str]:
    """Set status="suspected" on all knowledge nodes linked to *doc_source_file*.

    A knowledge node is considered linked if:
      - its ``source_file`` attribute matches *doc_source_file*, OR
      - its ``frontmatter.atomized_from`` matches *doc_source_file*.

    Already-suspected nodes are left unchanged (idempotent).

    Args:
        graph: The in-memory KnowledgeGraph to mutate.
        doc_source_file: Source file path (as stored in graph node attrs).
        reason: Human-readable reason for the flag (stored in node metadata).

    Returns:
        List of node IDs that were newly flagged as "suspected".
    """
    flagged: list[str] = []

    for node_id, data in graph.g.nodes(data=True):
        if data.get("kind") != "knowledge":
            continue

        # Match by source_file or frontmatter.atomized_from
        node_source = data.get("source_file", "")
        fm = data.get("frontmatter") or {}
        atomized_from = fm.get("atomized_from", "")

        if node_source != doc_source_file and atomized_from != doc_source_file:
            continue

        current_status = data.get("status", "confirmed")
        if current_status == "suspected":
            # Already flagged — idempotent, skip
            continue

        data["status"] = "suspected"
        if reason:
            meta = data.get("metadata") or {}
            meta["drift_reason"] = reason
            data["metadata"] = meta

        flagged.append(node_id)
        logger.info(
            "[drift] flagged %s as suspected (source=%s, reason=%s)",
            node_id,
            doc_source_file,
            reason or "unspecified",
        )

    return flagged
