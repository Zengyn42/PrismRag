"""
Obsidian Vault MCP — CAS (Compare-And-Swap) optimistic locking

Content hash (SHA-256) based optimistic locking mechanism:
  - On read: compute hash and return it to the caller
  - On write: compare hash; reject the write if it does not match
  - mtime is used only as a fast-path short-circuit (mtime unchanged → skip hash computation)

Concurrency protection:
  - Dict[path, asyncio.Lock] keyed by file path
  - Single-file granularity; does not affect operations on other files
"""

import asyncio
import hashlib
from pathlib import Path


# File-level locks: path_str -> asyncio.Lock
_file_locks: dict[str, asyncio.Lock] = {}


def get_file_lock(path: Path) -> asyncio.Lock:
    """Return a file-level asyncio.Lock for the given path (lazily created)."""
    key = str(path)
    if key not in _file_locks:
        _file_locks[key] = asyncio.Lock()
    return _file_locks[key]


def compute_hash(content: str | bytes) -> str:
    """Compute the SHA-256 digest of the given content."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def compute_file_hash(path: Path) -> str:
    """Compute the SHA-256 digest of the file at the given path."""
    return compute_hash(path.read_bytes())


def get_mtime_ms(path: Path) -> int:
    """Return the file modification time in milliseconds."""
    return int(path.stat().st_mtime * 1000)


def verify_cas(
    path: Path,
    expected_hash: str | None,
) -> tuple[bool, str]:
    """
    Verify CAS preconditions.

    Returns (is_valid, actual_hash).
    - expected_hash is None and file does not exist → create scenario, valid
    - expected_hash is None and file already exists → conflict (already exists)
    - expected_hash matches → valid
    - expected_hash does not match → conflict
    """
    exists = path.exists()

    if expected_hash is None:
        if exists:
            actual = compute_file_hash(path)
            return False, actual  # already exists, conflict
        return True, ""  # create scenario, valid

    if not exists:
        return False, ""  # file does not exist, expected_hash cannot match

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
