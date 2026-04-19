"""Tests for management vault tools: move_note, delete_note, manage_tags.

Calls the private _*_impl functions directly so no FastMCP instance is
required.  The ``single_vault`` fixture monkey-patches PrismRagSettings
inside vault_tools so the tools see a controlled single-namespace vault.

Graph-sync calls to _sync_graph are patched to a lightweight stub for
most tests; tests that verify graph state interact with the real
KnowledgeGraph via the graph.json seeded in the fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from prism_rag.config import GraphSource, PrismRagSettings
from prism_rag.store.graph import KnowledgeGraph, Node


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

    # Plain note without knowledge_id (id will be path-based)
    (vault / "a.md").write_text(
        "---\ntitle: Note A\ntags: [alpha, beta]\n---\n\n# A\n\nContent A.",
        encoding="utf-8",
    )

    # Note with explicit knowledge_id (id will be knowledge_id-based)
    (vault / "know.md").write_text(
        "---\ntitle: KnowNote\nknowledge_id: KNOW-X\ntags: [x]\n---\n\n# Know\n\nKnowledge content.",
        encoding="utf-8",
    )

    # A note we'll use for tag tests
    (vault / "tags_note.md").write_text(
        "---\ntitle: Tags Note\ntags: [a, b, c]\n---\n\n# Tags\n\nContent.",
        encoding="utf-8",
    )

    # Seed an empty graph
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
# Tests: move_note
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_move_note_renames_file_on_disk(single_vault):
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _move_note_impl

    assert (vault / "a.md").exists()
    assert not (vault / "renamed.md").exists()

    with patch("prism_rag.mcp_server.vault_tools._sync_graph", side_effect=lambda *a, **kw: _stub_ingest(a[0])):
        raw = await _move_note_impl("a.md", "renamed.md", namespace="default")

    result = _json(raw)
    assert result["status"] == "ok", result
    assert not (vault / "a.md").exists(), "Source should be gone after move"
    assert (vault / "renamed.md").exists(), "Destination should exist after move"
    assert result["data"]["source"] == "a.md"
    assert result["data"]["dest"] == "renamed.md"


@pytest.mark.asyncio
async def test_move_note_dest_exists_fails(single_vault):
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _move_note_impl

    # Both a.md and know.md exist
    raw = await _move_note_impl("a.md", "know.md", namespace="default")
    result = _json(raw)
    assert result["status"] == "error"
    assert result["error_code"] == "already_exists"
    # Source still exists
    assert (vault / "a.md").exists()


@pytest.mark.asyncio
async def test_move_note_graph_sync_no_knowledge_id(single_vault):
    """Moving a path-based node should remove old node and add new one."""
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _move_note_impl

    # Pre-seed the graph with the old node id ("a")
    graph_path = data / "graph.json"
    kg = KnowledgeGraph()
    kg.add_node(Node(id="a", label="a", kind="note", source_file="a.md"))
    kg.save(graph_path)

    with patch("prism_rag.mcp_server.vault_tools._sync_graph", side_effect=lambda *a, **kw: _stub_ingest(a[0])):
        raw = await _move_note_impl("a.md", "b.md", namespace="default")

    result = _json(raw)
    assert result["status"] == "ok", result
    assert result["data"]["id_changed"] is True

    # Old node should be gone; stub ingest adds "stub" not "b", but the
    # removal of the stale node is what we test here.
    kg2 = KnowledgeGraph.load(graph_path)
    assert "a" not in kg2.g, "Stale path-based node should have been removed"


@pytest.mark.asyncio
async def test_move_note_graph_sync_preserves_knowledge_id(single_vault):
    """Moving a knowledge_id-based node should NOT change node ID."""
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _move_note_impl

    # Pre-seed with the KNOW-X node
    graph_path = data / "graph.json"
    kg = KnowledgeGraph()
    kg.add_node(Node(id="KNOW-X", label="KnowNote", kind="knowledge", source_file="know.md"))
    kg.save(graph_path)

    with patch("prism_rag.mcp_server.vault_tools._sync_graph", side_effect=lambda *a, **kw: _stub_ingest(a[0])):
        raw = await _move_note_impl("know.md", "moved_know.md", namespace="default")

    result = _json(raw)
    assert result["status"] == "ok", result
    assert result["data"]["id_changed"] is False, "knowledge_id nodes should keep same id"

    # KNOW-X node should still be in graph (not removed)
    kg2 = KnowledgeGraph.load(graph_path)
    assert "KNOW-X" in kg2.g, "KNOW-X node should still exist (id unchanged)"


# ---------------------------------------------------------------------------
# Tests: delete_note
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_note_soft_deletes_to_trash(single_vault):
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _delete_note_impl

    assert (vault / "a.md").exists()

    raw = await _delete_note_impl("a.md", namespace="default")
    result = _json(raw)

    assert result["status"] == "ok", result
    assert not (vault / "a.md").exists(), "Original file should be gone"

    trash_rel = result["data"]["trash_path"]
    trash_abs = vault / trash_rel
    assert trash_abs.exists(), f"Trashed file should exist at {trash_abs}"
    assert ".trash" in trash_rel, "Trash path should be inside .trash/"


@pytest.mark.asyncio
async def test_delete_note_removes_from_graph(single_vault):
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _delete_note_impl

    # Pre-seed graph with the node
    graph_path = data / "graph.json"
    kg = KnowledgeGraph()
    kg.add_node(Node(id="a", label="a", kind="note", source_file="a.md"))
    kg.save(graph_path)

    raw = await _delete_note_impl("a.md", namespace="default")
    result = _json(raw)

    assert result["status"] == "ok", result

    # Node should be removed from graph
    kg2 = KnowledgeGraph.load(graph_path)
    assert "a" not in kg2.g, "Deleted node should be removed from graph"


@pytest.mark.asyncio
async def test_delete_note_trash_collision_adds_timestamp(single_vault):
    """Deleting same file twice should produce two files in .trash/ (no overwrite)."""
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _delete_note_impl

    # Create .trash/a.md so first collision can be triggered
    trash_dir = vault / ".trash"
    trash_dir.mkdir(exist_ok=True)
    (trash_dir / "a.md").write_text("pre-existing trash", encoding="utf-8")

    raw = await _delete_note_impl("a.md", namespace="default")
    result = _json(raw)

    assert result["status"] == "ok", result
    trash_rel = result["data"]["trash_path"]
    # The new trashed file should have a timestamp suffix, not just "a.md"
    assert trash_rel != ".trash/a.md", "Should have timestamp suffix to avoid collision"
    assert (vault / trash_rel).exists()


# ---------------------------------------------------------------------------
# Tests: manage_tags
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manage_tags_add_new_tags(single_vault):
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _manage_tags_impl
    from prism_rag.vault_ops.markdown_ops import parse_frontmatter

    with patch("prism_rag.mcp_server.vault_tools._sync_graph", side_effect=lambda *a, **kw: _stub_ingest(a[0])):
        raw = await _manage_tags_impl(
            "tags_note.md",
            add=["d", "e"],
            namespace="default",
        )

    result = _json(raw)
    assert result["status"] == "ok", result

    tags_after = result["data"]["tags_after"]
    assert "a" in tags_after
    assert "b" in tags_after
    assert "c" in tags_after
    assert "d" in tags_after
    assert "e" in tags_after

    # Verify on disk
    text = (vault / "tags_note.md").read_text(encoding="utf-8")
    fm, _ = parse_frontmatter(text)
    assert set(fm["tags"]) == {"a", "b", "c", "d", "e"}


@pytest.mark.asyncio
async def test_manage_tags_remove_existing_tags(single_vault):
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _manage_tags_impl
    from prism_rag.vault_ops.markdown_ops import parse_frontmatter

    with patch("prism_rag.mcp_server.vault_tools._sync_graph", side_effect=lambda *a, **kw: _stub_ingest(a[0])):
        raw = await _manage_tags_impl(
            "tags_note.md",
            remove=["b"],
            namespace="default",
        )

    result = _json(raw)
    assert result["status"] == "ok", result

    tags_after = result["data"]["tags_after"]
    assert "b" not in tags_after
    assert "a" in tags_after
    assert "c" in tags_after

    # Verify on disk
    text = (vault / "tags_note.md").read_text(encoding="utf-8")
    fm, _ = parse_frontmatter(text)
    assert set(fm["tags"]) == {"a", "c"}


@pytest.mark.asyncio
async def test_manage_tags_cas_conflict(single_vault):
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _manage_tags_impl

    # Wrong CAS hash → CONFLICT, file unchanged
    raw = await _manage_tags_impl(
        "tags_note.md",
        add=["z"],
        cas_hash="0" * 64,
        namespace="default",
    )

    result = _json(raw)
    assert result["status"] == "error"
    assert result["error_code"] == "conflict"

    # File unchanged
    text = (vault / "tags_note.md").read_text(encoding="utf-8")
    assert "z" not in text


@pytest.mark.asyncio
async def test_manage_tags_deduplication(single_vault):
    """Adding a tag that already exists should not duplicate it."""
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _manage_tags_impl

    with patch("prism_rag.mcp_server.vault_tools._sync_graph", side_effect=lambda *a, **kw: _stub_ingest(a[0])):
        raw = await _manage_tags_impl(
            "tags_note.md",
            add=["a"],  # 'a' already exists
            namespace="default",
        )

    result = _json(raw)
    assert result["status"] == "ok", result
    tags_after = result["data"]["tags_after"]
    assert tags_after.count("a") == 1, "Duplicate tags should not be added"


@pytest.mark.asyncio
async def test_manage_tags_strips_hash_prefix(single_vault):
    """Tags supplied with leading '#' should be normalized."""
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _manage_tags_impl

    with patch("prism_rag.mcp_server.vault_tools._sync_graph", side_effect=lambda *a, **kw: _stub_ingest(a[0])):
        raw = await _manage_tags_impl(
            "tags_note.md",
            add=["#newtag"],
            remove=["#b"],
            namespace="default",
        )

    result = _json(raw)
    assert result["status"] == "ok", result
    tags_after = result["data"]["tags_after"]
    assert "newtag" in tags_after
    assert "b" not in tags_after
    assert "#newtag" not in tags_after, "Leading '#' should be stripped"


@pytest.mark.asyncio
async def test_manage_tags_returns_tags_before(single_vault):
    """Response should include the original tag list before modification."""
    vault, data, settings = single_vault
    from prism_rag.mcp_server.vault_tools import _manage_tags_impl

    with patch("prism_rag.mcp_server.vault_tools._sync_graph", side_effect=lambda *a, **kw: _stub_ingest(a[0])):
        raw = await _manage_tags_impl(
            "tags_note.md",
            add=["new"],
            namespace="default",
        )

    result = _json(raw)
    assert result["status"] == "ok", result
    assert set(result["data"]["tags_before"]) == {"a", "b", "c"}
    assert "new" in result["data"]["tags_after"]
    assert "new" not in result["data"]["tags_before"]
