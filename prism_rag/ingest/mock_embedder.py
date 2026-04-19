"""Mock embedding backend for tests.

Derives a deterministic 768-dim pseudo-vector from the SHA-256 of content.
Same content → same vector → reproducible similarity edges.
"""
from __future__ import annotations

import hashlib
import struct


def mock_embed_text(text: str) -> list[float]:
    """Return a 768-dim pseudo-embedding derived from SHA-256 of `text`.

    The vector is L2-normalized so cosine-similarity semantics hold.
    """
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    floats: list[float] = []
    i = 0
    while len(floats) < 768:
        h = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        for j in range(0, 32, 4):
            val = struct.unpack("<I", h[j:j+4])[0]
            floats.append((val / 2**32) * 2 - 1)
        i += 1
    floats = floats[:768]
    norm = sum(f * f for f in floats) ** 0.5
    if norm > 0:
        floats = [f / norm for f in floats]
    return floats
