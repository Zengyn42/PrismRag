"""Tests for ported vault read tools (Step 2 Task 1).

Calls the private _*_impl functions directly so no FastMCP instance is
required.  The ``single_vault`` fixture monkey-patches PrismRagSettings
inside vault_tools so the tools see a controlled single-namespace vault.
"""

from __future__ import annotations

import json

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

    # Root note with frontmatter
    (vault / "note.md").write_text(
        "---\ntitle: Hello\ntags: [x, y]\n---\n\n# Body\n\nSome content here.",
        encoding="utf-8",
    )

    # Nested note (no frontmatter)
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

    return vault, data


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _json(raw: str) -> dict:
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Tests: read_note
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_note_returns_content_and_frontmatter(single_vault):
    from prism_rag.mcp_server.vault_tools import _read_note_impl

    raw = await _read_note_impl("note.md", namespace="default")
    result = _json(raw)

    assert result["status"] == "success"
    data = result["data"]
    assert "# Body" in data["content"]
    assert data["frontmatter"]["title"] == "Hello"
    assert data["frontmatter"]["tags"] == ["x", "y"]
    assert len(data["cas_hash"]) == 64  # SHA-256 hex
    assert data["mtime_ms"] > 0
    assert data["path"] == "note.md"
    assert data["namespace"] == "default"


@pytest.mark.asyncio
async def test_read_note_not_found_returns_error(single_vault):
    from prism_rag.mcp_server.vault_tools import _read_note_impl

    raw = await _read_note_impl("does_not_exist.md")
    result = _json(raw)

    assert result["status"] == "error"
    assert result["error_code"] == "not_found"


@pytest.mark.asyncio
async def test_read_note_path_traversal_returns_error(single_vault):
    from prism_rag.mcp_server.vault_tools import _read_note_impl

    raw = await _read_note_impl("../../etc/passwd")
    result = _json(raw)

    assert result["status"] == "error"
    assert result["error_code"] == "path_traversal"


# ---------------------------------------------------------------------------
# Tests: list_files
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_files_non_recursive(single_vault):
    vault, _ = single_vault
    from prism_rag.mcp_server.vault_tools import _list_files_impl

    raw = await _list_files_impl(directory="", pattern="*.md", recursive=False)
    result = _json(raw)

    assert result["status"] == "success"
    paths = [f["path"] for f in result["data"]["files"]]
    assert "note.md" in paths
    # Recursive is False — nested.md must NOT appear
    assert not any("nested" in p for p in paths)
    assert result["data"]["count"] == len(paths)


@pytest.mark.asyncio
async def test_list_files_recursive(single_vault):
    from prism_rag.mcp_server.vault_tools import _list_files_impl

    raw = await _list_files_impl(directory="", pattern="*.md", recursive=True)
    result = _json(raw)

    assert result["status"] == "success"
    paths = [f["path"] for f in result["data"]["files"]]
    assert "note.md" in paths
    assert any("nested.md" in p for p in paths)
    assert result["data"]["count"] == 2


@pytest.mark.asyncio
async def test_list_files_missing_directory_returns_error(single_vault):
    from prism_rag.mcp_server.vault_tools import _list_files_impl

    raw = await _list_files_impl(directory="nonexistent_dir")
    result = _json(raw)

    assert result["status"] == "error"
    assert result["error_code"] == "not_found"


# ---------------------------------------------------------------------------
# Tests: get_frontmatter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_frontmatter_returns_only_frontmatter(single_vault):
    from prism_rag.mcp_server.vault_tools import _get_frontmatter_impl

    raw = await _get_frontmatter_impl("note.md")
    result = _json(raw)

    assert result["status"] == "success"
    data = result["data"]
    assert "content" not in data  # only frontmatter, not full content
    assert data["frontmatter"]["title"] == "Hello"
    assert data["path"] == "note.md"
    assert data["namespace"] == "default"


@pytest.mark.asyncio
async def test_get_frontmatter_no_frontmatter_returns_empty_dict(single_vault):
    from prism_rag.mcp_server.vault_tools import _get_frontmatter_impl

    raw = await _get_frontmatter_impl("sub/nested.md")
    result = _json(raw)

    assert result["status"] == "success"
    assert result["data"]["frontmatter"] == {}


# ---------------------------------------------------------------------------
# Tests: _resolve_vault helper
# ---------------------------------------------------------------------------


def test_resolve_vault_single_namespace_no_arg(single_vault):
    from prism_rag.mcp_server.vault_tools import _resolve_vault

    result = _resolve_vault("")
    assert not isinstance(result, dict), f"Expected (vault, src) tuple, got error: {result}"
    vault, src = result
    assert src.namespace == "default"


def test_resolve_vault_explicit_namespace(single_vault):
    from prism_rag.mcp_server.vault_tools import _resolve_vault

    result = _resolve_vault("default")
    assert not isinstance(result, dict)
    _, src = result
    assert src.namespace == "default"


def test_resolve_vault_unknown_namespace_returns_error(single_vault):
    from prism_rag.mcp_server.vault_tools import _resolve_vault

    result = _resolve_vault("does_not_exist")
    assert isinstance(result, dict)
    assert result["status"] == "error"
    assert result["error_code"] == "not_found"
