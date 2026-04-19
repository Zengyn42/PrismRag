"""Tests for ported vault write tools (Step 2 Task 2).

Calls the private _*_impl functions directly so no FastMCP instance is
required.  The ``single_vault`` fixture monkey-patches PrismRagSettings
inside vault_tools so the tools see a controlled single-namespace vault.

The ``ingest_file`` call inside each write is also patched to a lightweight
stub so tests don't need a real graph or API key — graph-sync behaviour is
verified via the response ``graph_update`` field.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from prism_rag.config import GraphSource, PrismRagSettings


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def single_vault(tmp_path, monkeypatch):
    """Single-namespace vault with sample notes."""
    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()

    # Root note with frontmatter and sections
    (vault / "note.md").write_text(
        "---\ntitle: Hello\ntags: [x, y]\n---\n\n# Intro\n\nIntro content.\n\n## Decision\n\nOriginal decision text.\n\n## Summary\n\nSummary text.",
        encoding="utf-8",
    )

    # Nested note (no frontmatter) for path tests
    sub = vault / "sub"
    sub.mkdir()
    (sub / "nested.md").write_text("# Nested\n\nNested content.", encoding="utf-8")

    # Seed an empty graph so graph-loading code doesn't choke
    from prism_rag.store.graph import KnowledgeGraph
    KnowledgeGraph().save(data / "graph.json")

    settings = PrismRagSettings(
        graphs=[
            GraphSource(
                namespace="default",
                vault_path=vault,
                data_dir=data,
                writable=True,
            )
        ],
    )

    # Patch PrismRagSettings inside vault_tools so _resolve_vault uses our fixture
    monkeypatch.setattr(
        "prism_rag.mcp_server.vault_tools.PrismRagSettings",
        lambda: settings,
    )

    return vault, data, settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json(raw: str) -> dict:
    return json.loads(raw)


def _stub_ingest(resolved_path, settings=None, skip_embed=True, skip_persist=False):
    """Lightweight stand-in for ingest_file; avoids real graph I/O."""
    return {
        "node_id": "stub",
        "action": "added",
        "ast_edges": 0,
        "similarity_edges": 0,
        "total_nodes": 1,
        "total_edges": 0,
        "communities": 0,
    }


# ---------------------------------------------------------------------------
# Tests: write_note
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_note_creates_new_file(single_vault):
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _write_note_impl

    target = "new_doc.md"
    content = "# New\n\nHello world."

    with patch("prism_rag.mcp_server.vault_tools._sync_graph", side_effect=lambda *a, **kw: _stub_ingest(a[0])):
        raw = await _write_note_impl(target, content, cas_hash="", namespace="default")

    result = _json(raw)
    assert result["status"] == "ok", result
    data_block = result["data"]
    assert len(data_block["cas_hash"]) == 64  # SHA-256 hex
    assert data_block["path"] == target
    assert (vault / target).read_text(encoding="utf-8") == content


@pytest.mark.asyncio
async def test_write_note_cas_conflict_on_stale_hash(single_vault, monkeypatch):
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _write_note_impl

    # Redirect audit_log._audit_path to our temp data dir so the JSONL is
    # visible to the test (audit_log reads PrismRagSettings directly, not the
    # monkeypatched vault_tools version).
    audit_path = data / "audit.jsonl"
    monkeypatch.setattr(
        "prism_rag.vault_ops.audit_log._audit_path",
        lambda: audit_path,
    )

    # note.md already exists; stale hash triggers CONFLICT
    raw = await _write_note_impl("note.md", "new content", cas_hash="aabbcc" * 10 + "aabb", namespace="default")
    result = _json(raw)

    assert result["status"] == "error"
    assert result["error_code"] in ("conflict", "already_exists")

    # Verify audit was written
    assert audit_path.exists(), "Audit log should have been written"
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert any("conflict" in line for line in lines)


@pytest.mark.asyncio
async def test_write_note_already_exists_without_hash(single_vault):
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _write_note_impl

    # note.md exists; empty cas_hash => ALREADY_EXISTS
    raw = await _write_note_impl("note.md", "oops", cas_hash="", namespace="default")
    result = _json(raw)

    assert result["status"] == "error"
    assert result["error_code"] == "already_exists"


@pytest.mark.asyncio
async def test_write_note_triggers_graph_update(single_vault):
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _write_note_impl

    captured = {}

    def fake_sync(path, settings, tool_name):
        captured["path"] = str(path)
        captured["tool"] = tool_name
        return {"node_id": "new/graph_test.md", "action": "added"}

    with patch("prism_rag.mcp_server.vault_tools._sync_graph", side_effect=fake_sync):
        raw = await _write_note_impl("graph_test.md", "# Graph Test", cas_hash="", namespace="default")

    result = _json(raw)
    assert result["status"] == "ok"
    assert "graph_update" in result
    assert result["graph_update"]["node_id"] == "new/graph_test.md"
    assert captured["tool"] == "write_note"


# ---------------------------------------------------------------------------
# Tests: patch_note
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_patch_note_replaces_section(single_vault):
    vault, data, settings = single_vault
    from prism_rag.vault_ops.cas import compute_file_hash
    from prism_rag.mcp_server.vault_tools import _patch_note_impl

    note_path = vault / "note.md"
    original_hash = compute_file_hash(note_path)

    with patch("prism_rag.mcp_server.vault_tools._sync_graph", side_effect=lambda *a, **kw: _stub_ingest(a[0])):
        raw = await _patch_note_impl(
            "note.md",
            section_heading="## Decision",
            new_content="\nPatched decision content.\n",
            cas_hash=original_hash,
            namespace="default",
        )

    result = _json(raw)
    assert result["status"] == "ok", result
    assert result["data"]["sections_affected"] == ["## Decision"]

    # Verify only the targeted section changed
    new_text = note_path.read_text(encoding="utf-8")
    assert "Patched decision content." in new_text
    assert "Original decision text." not in new_text
    # Other sections untouched
    assert "Intro content." in new_text
    assert "Summary text." in new_text
    # Frontmatter preserved
    assert "title: Hello" in new_text


@pytest.mark.asyncio
async def test_patch_note_heading_not_found(single_vault):
    vault, data, settings = single_vault
    from prism_rag.vault_ops.cas import compute_file_hash
    from prism_rag.mcp_server.vault_tools import _patch_note_impl

    note_path = vault / "note.md"
    current_hash = compute_file_hash(note_path)

    raw = await _patch_note_impl(
        "note.md",
        section_heading="## NonExistentSection",
        new_content="irrelevant",
        cas_hash=current_hash,
        namespace="default",
    )

    result = _json(raw)
    assert result["status"] == "error"
    assert result["error_code"] == "validation_error"
    assert "NonExistentSection" in result["message"]


@pytest.mark.asyncio
async def test_patch_note_file_not_found(single_vault):
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _patch_note_impl

    raw = await _patch_note_impl(
        "does_not_exist.md",
        section_heading="## Anything",
        new_content="x",
        namespace="default",
    )

    result = _json(raw)
    assert result["status"] == "error"
    assert result["error_code"] == "not_found"


# ---------------------------------------------------------------------------
# Tests: update_frontmatter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_frontmatter_merges_fields(single_vault):
    vault, data, settings = single_vault
    from prism_rag.vault_ops.markdown_ops import parse_frontmatter
    from prism_rag.mcp_server.vault_tools import _update_frontmatter_impl

    with patch("prism_rag.mcp_server.vault_tools._sync_graph", side_effect=lambda *a, **kw: _stub_ingest(a[0])):
        raw = await _update_frontmatter_impl(
            "note.md",
            updates={"status": "reviewed", "priority": 1},
            namespace="default",
        )

    result = _json(raw)
    assert result["status"] == "ok", result

    # Check merged frontmatter in file
    new_text = (vault / "note.md").read_text(encoding="utf-8")
    fm, body = parse_frontmatter(new_text)
    assert fm["title"] == "Hello"       # original field preserved
    assert fm["tags"] == ["x", "y"]     # original field preserved
    assert fm["status"] == "reviewed"   # new field added
    assert fm["priority"] == 1          # new field added


@pytest.mark.asyncio
async def test_update_frontmatter_updates_graph_hash(single_vault):
    vault, data, settings = single_vault
    from prism_rag.vault_ops.cas import compute_file_hash
    from prism_rag.mcp_server.vault_tools import _update_frontmatter_impl

    note_path = vault / "note.md"
    hash_before = compute_file_hash(note_path)

    graph_calls = []

    def fake_sync(path, settings, tool_name):
        graph_calls.append({"path": str(path), "tool": tool_name})
        return {"node_id": "note", "action": "updated"}

    with patch("prism_rag.mcp_server.vault_tools._sync_graph", side_effect=fake_sync):
        raw = await _update_frontmatter_impl(
            "note.md",
            updates={"new_key": "new_value"},
            namespace="default",
        )

    result = _json(raw)
    assert result["status"] == "ok"

    hash_after = result["data"]["cas_hash"]
    assert hash_after != hash_before, "Hash should change after frontmatter update"

    # Graph sync was called exactly once
    assert len(graph_calls) == 1
    assert graph_calls[0]["tool"] == "update_frontmatter"

    # Graph update reflected in response
    assert result["graph_update"]["action"] == "updated"


@pytest.mark.asyncio
async def test_update_frontmatter_cas_conflict(single_vault):
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _update_frontmatter_impl

    # Supply a wrong hash
    raw = await _update_frontmatter_impl(
        "note.md",
        updates={"key": "val"},
        cas_hash="0" * 64,
        namespace="default",
    )

    result = _json(raw)
    assert result["status"] == "error"
    assert result["error_code"] == "conflict"
