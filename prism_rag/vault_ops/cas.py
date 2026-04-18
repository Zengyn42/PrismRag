"""
Obsidian Vault MCP — CAS (Compare-And-Swap) 乐观锁

基于 content hash (SHA-256) 的乐观锁机制：
  - read 时计算 hash 返回给调用方
  - write 时比对 hash，不匹配则拒绝写入
  - mtime 仅做快路径短路（mtime 未变 → 跳过 hash 计算）

并发保护：
  - Dict[path, asyncio.Lock] 按文件路径加锁
  - 单文件级粒度，不影响其他文件操作
"""

import asyncio
import hashlib
from pathlib import Path


# 文件级锁：path_str -> asyncio.Lock
_file_locks: dict[str, asyncio.Lock] = {}


def get_file_lock(path: Path) -> asyncio.Lock:
    """获取文件级 asyncio.Lock（懒创建）。"""
    key = str(path)
    if key not in _file_locks:
        _file_locks[key] = asyncio.Lock()
    return _file_locks[key]


def compute_hash(content: str | bytes) -> str:
    """计算内容 SHA-256 摘要。"""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def compute_file_hash(path: Path) -> str:
    """计算文件内容的 SHA-256 摘要。"""
    return compute_hash(path.read_bytes())


def get_mtime_ms(path: Path) -> int:
    """获取文件 mtime（毫秒）。"""
    return int(path.stat().st_mtime * 1000)


def verify_cas(
    path: Path,
    expected_hash: str | None,
) -> tuple[bool, str]:
    """
    验证 CAS 条件。

    返回 (is_valid, actual_hash)。
    - expected_hash is None 且文件不存在 → 新建场景，valid
    - expected_hash is None 且文件已存在 → conflict（已存在）
    - expected_hash 匹配 → valid
    - expected_hash 不匹配 → conflict
    """
    exists = path.exists()

    if expected_hash is None:
        if exists:
            actual = compute_file_hash(path)
            return False, actual  # 已存在，conflict
        return True, ""  # 新建，valid

    if not exists:
        return False, ""  # 文件不存在，expected_hash 无法匹配

    actual = compute_file_hash(path)
    return actual == expected_hash, actual


import os


class CASConflict(Exception):
    """Raised when expected_hash does not match the current file hash."""

    def __init__(self, path: Path, expected: str, actual: str):
        super().__init__(
            f"CAS conflict on {path}: expected={expected}, actual={actual}"
        )
        self.path = path
        self.expected = expected
        self.actual = actual


def atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write `content` to `path` atomically via tempfile + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding=encoding)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def write_with_cas(path: Path, content: str, expected_hash: str | None) -> str:
    """Atomically write `content`, honouring optimistic CAS.

    expected_hash:
      - None   → file must NOT exist (create-only)
      - str    → file must exist, SHA-256 must match (with or without 'sha256:' prefix)

    Returns the new hash. Raises `CASConflict` on mismatch or pre-existence.
    """
    exists = path.exists()

    if expected_hash is None:
        if exists:
            raise CASConflict(path, "<new>", compute_file_hash(path))
    else:
        if not exists:
            raise CASConflict(path, expected_hash, "<missing>")
        expected_clean = expected_hash.removeprefix("sha256:")
        actual = compute_file_hash(path)
        if actual != expected_clean:
            raise CASConflict(path, expected_hash, actual)

    atomic_write(path, content)
    return compute_hash(content)
