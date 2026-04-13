"""
Obsidian Vault MCP — 结构化审计日志

所有写操作记录到结构化 JSON 日志。
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("obsidian_mcp.audit")


def log_operation(
    tool: str,
    target: str,
    action: str,
    status: str,
    cas_before: str = "",
    cas_after: str = "",
    **extra: Any,
) -> None:
    """记录一次操作到审计日志。"""
    entry = {
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

    logger.info(json.dumps(entry, ensure_ascii=False))
