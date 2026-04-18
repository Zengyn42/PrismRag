"""Obsidian Vault MCP — 结构化审计日志

All write operations append a JSONL line to data/audit.jsonl.
Audit write is best-effort — a failed audit does not abort the main op.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("obsidian_mcp.audit")


def _audit_path() -> Path:
    """Return the path to the audit JSONL file.

    Uses PRISM_DATA_DIR if set, else defaults to ./data/audit.jsonl.
    Tests can monkeypatch this function to redirect writes.
    """
    from prism_rag.config import PrismRagSettings
    settings = PrismRagSettings()
    return settings.data_dir / "audit.jsonl"


def log_operation(
    tool: str,
    target: str,
    action: str,
    status: str,
    cas_before: str = "",
    cas_after: str = "",
    **extra: Any,
) -> None:
    """Record an operation to the audit log (JSONL + stdlib logger)."""
    entry: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "target": target,
        "action": action,
        "status": status,
    }
    if cas_before:
        entry["cas_before"] = cas_before
    if cas_after:
        entry["cas_after"] = cas_after
    if extra:
        entry.update(extra)

    try:
        path = _audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[audit] failed to write audit log: {e}")

    logger.info(json.dumps(entry, ensure_ascii=False))
