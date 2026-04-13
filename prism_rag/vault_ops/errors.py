"""
Obsidian Vault MCP — 错误码与统一响应构造

VaultErrorCode 枚举 + 统一的成功/错误响应格式。
所有 tool 返回值都经过 ok() / fail() 构造，确保格式一致。
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
    """构造成功响应。"""
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
    """构造错误响应。"""
    resp: dict[str, Any] = {
        "status": "error",
        "error_code": code.value,
        "message": message,
    }
    if metadata:
        resp["metadata"] = metadata
    return resp
