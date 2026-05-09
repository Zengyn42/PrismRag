"""Registry — file-locked KNOW-ID allocator.

Maintains a monotone counter in a JSON file. Uses a per-path threading.Lock
for same-process thread safety combined with fcntl.LOCK_EX for cross-process
safety. Format: KNOW-{n:06d} with 6-digit zero-padding (expandable to 7+ as
counter grows).

Note on locking strategy
------------------------
fcntl.flock is advisory and per open-file-description. On Linux, threads
within the same process that independently open the same path each get a
distinct file description, so flock alone does NOT prevent intra-process
races. We therefore maintain a class-level dict of per-path threading.Lock
objects so that all Registry instances sharing the same path serialise through
the same lock before acquiring the file lock.
"""
from __future__ import annotations

import fcntl
import json
import threading
from pathlib import Path


_ID_PREFIX = "KNOW"

# Class-level lock registry: absolute path -> threading.Lock
_path_locks: dict[str, threading.Lock] = {}
_path_locks_mutex = threading.Lock()


def _get_thread_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _path_locks_mutex:
        if key not in _path_locks:
            _path_locks[key] = threading.Lock()
        return _path_locks[key]


def _format_id(n: int) -> str:
    return f"{_ID_PREFIX}-{n:06d}"


class Registry:
    """Allocates monotonically increasing KNOW-IDs from a JSON counter file."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def alloc_id(self) -> str:
        """Allocate one KNOW-ID. Thread- and process-safe."""
        return self.batch_alloc(1)[0]

    def batch_alloc(self, count: int) -> list[str]:
        """Allocate `count` KNOW-IDs atomically. Returns list in order."""
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")

        self._path.parent.mkdir(parents=True, exist_ok=True)

        thread_lock = _get_thread_lock(self._path)
        with thread_lock:
            with self._path.open("a+", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    raw = f.read().strip()
                    state = json.loads(raw) if raw else {}
                    next_n = state.get("next", 1)
                    ids = [_format_id(next_n + i) for i in range(count)]
                    state["next"] = next_n + count
                    f.seek(0)
                    f.truncate()
                    f.write(json.dumps(state, ensure_ascii=False) + "\n")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

        return ids
