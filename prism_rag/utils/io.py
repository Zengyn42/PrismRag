"""Atomic file write utility — Sprint 1 protocol contract.

`atomic_write(path, content)` guarantees readers never see partial content,
even under concurrent reads. Implementation uses a sibling tmp file and
POSIX `os.replace` (atomic rename within the same filesystem).
"""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write `content` to `path` atomically.

    Steps:
      1. Write to `<path>.tmp` (sibling, same filesystem)
      2. `os.replace(tmp, path)` — atomic on POSIX
      3. On any exception, remove tmp file and re-raise

    Concurrent readers either see the old file content (still complete)
    or the new file content (also complete). They never see a partial
    write.
    """
    path = Path(path)
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
