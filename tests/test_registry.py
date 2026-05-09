"""Tests for Registry — file-locked KNOW-ID allocator."""
from __future__ import annotations

import re
import threading
from pathlib import Path

import pytest

from prism_rag.store.registry import Registry


KNOW_ID_PATTERN = re.compile(r"^KNOW-\d{6,}$")


def test_registry_alloc_id_format(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    kid = reg.alloc_id()
    assert KNOW_ID_PATTERN.match(kid), f"Bad format: {kid!r}"


def test_registry_alloc_id_is_sequential(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    ids = [reg.alloc_id() for _ in range(5)]
    # Extract numeric parts and verify monotone increasing
    nums = [int(k.split("-")[1]) for k in ids]
    assert nums == sorted(nums)
    assert len(set(ids)) == 5  # all unique


def test_registry_persists_across_instances(tmp_path):
    path = tmp_path / "registry.json"
    reg1 = Registry(path)
    first = reg1.alloc_id()
    num1 = int(first.split("-")[1])

    reg2 = Registry(path)
    second = reg2.alloc_id()
    num2 = int(second.split("-")[1])

    assert num2 > num1  # registry state persisted


def test_registry_no_id_reuse(tmp_path):
    path = tmp_path / "registry.json"
    ids = set()
    for _ in range(10):
        reg = Registry(path)  # Fresh instance each time
        ids.add(reg.alloc_id())
    assert len(ids) == 10


def test_registry_batch_alloc(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    ids = reg.batch_alloc(5)
    assert len(ids) == 5
    assert len(set(ids)) == 5
    nums = [int(k.split("-")[1]) for k in ids]
    assert nums == sorted(nums)


def test_registry_concurrent_alloc_no_duplicates(tmp_path):
    """Concurrent allocations from different Registry instances must not produce duplicates."""
    path = tmp_path / "registry.json"
    results: list[str] = []
    lock = threading.Lock()
    errors: list[Exception] = []

    def alloc_one():
        try:
            reg = Registry(path)
            kid = reg.alloc_id()
            with lock:
                results.append(kid)
        except Exception as e:
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=alloc_one) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors: {errors}"
    assert len(results) == 20
    assert len(set(results)) == 20, "Duplicate IDs allocated!"
