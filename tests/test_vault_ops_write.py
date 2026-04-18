"""Tests for vault_ops write path (Section 6)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_rag.vault_ops.cas import (
    CASConflict,
    atomic_write,
    compute_hash,
    write_with_cas,
)


def test_atomic_write_creates_file(tmp_path):
    target = tmp_path / "sub" / "note.md"
    target.parent.mkdir()
    atomic_write(target, "hello")
    assert target.read_text() == "hello"


def test_atomic_write_leaves_no_tmp(tmp_path):
    target = tmp_path / "note.md"
    atomic_write(target, "hello")
    tmps = list(tmp_path.glob("*.tmp"))
    assert tmps == []


def test_write_with_cas_fresh_file(tmp_path):
    target = tmp_path / "new.md"
    new_hash = write_with_cas(target, "hello", expected_hash=None)
    assert target.read_text() == "hello"
    assert new_hash == compute_hash("hello")


def test_write_with_cas_matching_hash(tmp_path):
    target = tmp_path / "existing.md"
    target.write_text("v1")
    v1_hash = compute_hash("v1")
    new_hash = write_with_cas(target, "v2", expected_hash=v1_hash)
    assert target.read_text() == "v2"
    assert new_hash == compute_hash("v2")


def test_write_with_cas_conflict_raises(tmp_path):
    target = tmp_path / "existing.md"
    target.write_text("v1")
    with pytest.raises(CASConflict) as exc_info:
        write_with_cas(target, "v2", expected_hash="sha256:deadbeef")
    # The exception message or .expected should contain our wrong hash
    assert "deadbeef" in str(exc_info.value) or exc_info.value.expected == "sha256:deadbeef"
