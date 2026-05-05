"""Tests for atomic write utility (Sprint 1 protocol)."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from prism_rag.utils.io import atomic_write


def test_atomic_write_creates_file(tmp_path):
    target = tmp_path / "out.json"
    atomic_write(target, '{"hello": "world"}')
    assert target.read_text() == '{"hello": "world"}'


def test_atomic_write_overwrites_existing(tmp_path):
    target = tmp_path / "out.json"
    target.write_text("OLD")
    atomic_write(target, "NEW")
    assert target.read_text() == "NEW"


def test_atomic_write_cleans_tmp_on_success(tmp_path):
    target = tmp_path / "out.json"
    atomic_write(target, "x")
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_atomic_write_no_torn_reads(tmp_path):
    """Concurrent readers never see partial content."""
    target = tmp_path / "big.json"
    big = "A" * 100_000
    target.write_text(big)
    stop = threading.Event()
    seen_torn = []

    def reader():
        while not stop.is_set():
            try:
                content = target.read_text()
            except FileNotFoundError:
                continue
            if content and content[0] != content[-1]:
                seen_torn.append(content[:50])

    rt = threading.Thread(target=reader)
    rt.start()
    try:
        for i in range(50):
            new = ("B" if i % 2 == 0 else "C") * 100_000
            atomic_write(target, new)
            time.sleep(0.001)
    finally:
        stop.set()
        rt.join()
    assert seen_torn == []
