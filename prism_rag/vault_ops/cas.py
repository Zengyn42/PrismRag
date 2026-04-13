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
