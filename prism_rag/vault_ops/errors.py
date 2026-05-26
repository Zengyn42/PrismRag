"""
Obsidian Vault MCP — Error codes and unified response construction

VaultErrorCode enum + unified success/error response format.
All tool return values are constructed via ok() / fail() to ensure consistent format.
"""

from enum import Enum
from typing import Any


class VaultErrorCode(str, Enum):
    CONFLICT = "conflict"
    NOT_FOUND = "not_found"
    ALREADY_EXISTS = "already_exists"
    PERMISSION_DENIED = "permission_denied"
    VALIDATION_ERROR = "validation_error"
    FRONTMATTER_PARSE_ERROR = "frontmatter_parse_error"
    PATH_TRAVERSAL = "path_traversal"
    INDEX_FAILED = "index_failed"
    INTERNAL_ERROR = "internal_error"


def ok(data: dict[str, Any] | None = None, **metadata: Any) -> dict:
    """Construct a success response."""
    resp: dict[str, Any] = {"status": "success"}
    if data is not None:
        resp["data"] = data
    if metadata:
        resp["metadata"] = metadata
    return resp


def fail(
    code: VaultErrorCode,
    message: str,
    **metadata: Any,
) -> dict:
    """Construct an error response."""
    resp: dict[str, Any] = {
        "status": "error",
        "error_code": code.value,
        "message": message,
    }
    if metadata:
        resp["metadata"] = metadata
    return resp
