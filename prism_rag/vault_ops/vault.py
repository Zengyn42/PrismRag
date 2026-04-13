"""
Obsidian Vault MCP — Vault 路径管理与安全护栏

三层防御纵深：
  L1: 路径沙箱 — realpath 检查是否在 VAULT_BASE_DIR 内
  L2: 敏感目录黑名单 — .obsidian/, .git/, .trash/, node_modules/
  L3: 删除保护 — 移至 .trash/ 而非真删（由 tool 层实现）

所有文件路径操作必须经过 resolve_path() 验证。
"""

import os
from pathlib import Path

from prism_rag.vault_ops.errors import VaultErrorCode, fail

# 敏感目录黑名单（禁止写入/删除）
_BLOCKED_DIRS = frozenset({
    ".obsidian",
    ".git",
    ".trash",
    "node_modules",
    ".DS_Store",
})

# 允许的文件后缀
_ALLOWED_EXTENSIONS = frozenset({
    ".md", ".markdown", ".txt", ".canvas", ".base",
})


class Vault:
    """Vault 实例 — 绑定一个 base_dir，提供路径解析和安全校验。"""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).resolve()
        if not self.base_dir.is_dir():
            raise ValueError(f"Vault 目录不存在: {self.base_dir}")

    def resolve_path(self, relative_path: str) -> Path | dict:
        """
        将 Vault 相对路径解析为绝对路径。
        成功返回 Path，失败返回 fail() 错误响应。

        安全校验：
          1. 路径解析后必须在 base_dir 内（防穿越）
          2. 不得位于黑名单目录下
        """
        # 清理输入
        cleaned = relative_path.strip().lstrip("/").lstrip("\\")
        if not cleaned:
            return fail(VaultErrorCode.VALIDATION_ERROR, "路径不能为空")

        # 解析绝对路径
        abs_path = (self.base_dir / cleaned).resolve()

        # L1: 路径沙箱
        try:
            abs_path.relative_to(self.base_dir)
        except ValueError:
            return fail(
                VaultErrorCode.PATH_TRAVERSAL,
                f"路径逃逸: {relative_path} 解析到 Vault 外部",
            )

        # L2: 敏感目录黑名单
        rel_parts = abs_path.relative_to(self.base_dir).parts
        for part in rel_parts:
            if part in _BLOCKED_DIRS:
                return fail(
                    VaultErrorCode.PERMISSION_DENIED,
                    f"禁止访问敏感目录: {part}/",
                )

        return abs_path

    def resolve_dir(self, relative_path: str = "") -> Path | dict:
        """解析目录路径。空字符串 = vault 根目录。"""
        if not relative_path or relative_path == ".":
            return self.base_dir

        abs_path = (self.base_dir / relative_path.strip().lstrip("/")).resolve()

        try:
            abs_path.relative_to(self.base_dir)
        except ValueError:
            return fail(
                VaultErrorCode.PATH_TRAVERSAL,
                f"目录路径逃逸: {relative_path}",
            )

        # 黑名单检查
        rel_parts = abs_path.relative_to(self.base_dir).parts
        for part in rel_parts:
            if part in _BLOCKED_DIRS:
                return fail(
                    VaultErrorCode.PERMISSION_DENIED,
                    f"禁止访问敏感目录: {part}/",
                )

        return abs_path

    def relative_path(self, abs_path: Path) -> str:
        """绝对路径 → Vault 内相对路径字符串。"""
        return str(abs_path.relative_to(self.base_dir))

    def is_note(self, path: Path) -> bool:
        """是否为笔记文件（按后缀判断）。"""
        return path.suffix.lower() in _ALLOWED_EXTENSIONS
