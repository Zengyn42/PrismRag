"""Tests for search_files and get_links vault tools (Step 2 Task 4).

Calls the private _search_files_impl and _get_links_impl functions directly so
no FastMCP instance is required.  The ``single_vault`` fixture monkey-patches
PrismRagSettings inside vault_tools so the tools see a controlled
single-namespace vault.
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
    """Single-namespace vault populated with notes for search testing."""
    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()

    # Note whose filename matches "session"
    (vault / "session_mode.md").write_text(
        "# Session Mode\n\nThis note is about activation functions.",
        encoding="utf-8",
    )

    # Note whose *content* (not filename) contains "discussion"
    (vault / "planning.md").write_text(
        "# Planning\n\nA long discussion about strategy.",
        encoding="utf-8",
    )

    # Note with wikilinks — outgoing
    (vault / "linker.md").write_text(
        "# Linker\n\nSee [[other]] and [[third]] for details.\n\nAlso embeds ![[asset]].",
        encoding="utf-8",
    )

    # Notes that will be linked-to (for incoming-link tests)
    (vault / "target.md").write_text("# Target\n\nThis is the target.", encoding="utf-8")
    (vault / "a.md").write_text("# A\n\nReferences [[target]] in body.", encoding="utf-8")
    (vault / "b.md").write_text("# B\n\nAlso links to [[target]].", encoding="utf-8")

    # A note that references itself (to test self-exclusion in incoming)
    (vault / "self_ref.md").write_text(
        "# Self Ref\n\nThis note refers to [[self_ref]] itself.", encoding="utf-8"
    )

    # Seed an empty graph
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

    monkeypatch.setattr(
        "prism_rag.mcp_server.vault_tools.PrismRagSettings",
        lambda: settings,
    )

    return vault


# ---------------------------------------------------------------------------
# search_files tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_files_matches_filename(single_vault):
    from prism_rag.mcp_server.vault_tools import _search_files_impl

    raw = await _search_files_impl(query="session")
    result = json.loads(raw)

    assert result["status"] == "success"
    paths = [m["path"] for m in result["data"]["matches"]]
    assert any("session_mode.md" in p for p in paths)


@pytest.mark.asyncio
async def test_search_files_matches_content(single_vault):
    from prism_rag.mcp_server.vault_tools import _search_files_impl

    # "discussion" does not appear in any filename, only in planning.md content
    raw = await _search_files_impl(query="discussion")
    result = json.loads(raw)

    assert result["status"] == "success"
    paths = [m["path"] for m in result["data"]["matches"]]
    assert any("planning.md" in p for p in paths)
    # Must have content hits, not filename hit
    planning_match = next(m for m in result["data"]["matches"] if "planning.md" in m["path"])
    assert planning_match["filename_match"] is False
    assert len(planning_match["line_hits"]) > 0


@pytest.mark.asyncio
async def test_search_files_filename_only_skips_content(single_vault):
    from prism_rag.mcp_server.vault_tools import _search_files_impl

    # "discussion" lives only in content — filename_only=True should yield 0 matches
    raw = await _search_files_impl(query="discussion", filename_only=True)
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["data"]["count"] == 0


@pytest.mark.asyncio
async def test_search_files_case_insensitive_default(single_vault):
    from prism_rag.mcp_server.vault_tools import _search_files_impl

    # Upper-case query should still find session_mode.md by default
    raw = await _search_files_impl(query="SESSION")
    result = json.loads(raw)

    assert result["status"] == "success"
    paths = [m["path"] for m in result["data"]["matches"]]
    assert any("session_mode.md" in p for p in paths)


@pytest.mark.asyncio
async def test_search_files_case_sensitive_filter(single_vault):
    from prism_rag.mcp_server.vault_tools import _search_files_impl

    # "SESSION" (all caps) should NOT match "session_mode.md" when case_sensitive=True
    raw = await _search_files_impl(query="SESSION", case_sensitive=True)
    result = json.loads(raw)

    assert result["status"] == "success"
    paths = [m["path"] for m in result["data"]["matches"]]
    assert not any("session_mode.md" in p for p in paths)


@pytest.mark.asyncio
async def test_search_files_truncates_at_max_results(tmp_path, monkeypatch):
    from prism_rag.mcp_server.vault_tools import _search_files_impl
    from prism_rag.store.graph import KnowledgeGraph

    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()

    # Create 10 files all containing "alpha" in their filename
    for i in range(10):
        (vault / f"alpha_{i}.md").write_text(f"# Note {i}", encoding="utf-8")

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

    raw = await _search_files_impl(query="alpha", max_results=3)
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["data"]["count"] == 3
    assert result["data"]["truncated"] is True


# ---------------------------------------------------------------------------
# get_links tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_links_extracts_outgoing_wikilinks(single_vault):
    from prism_rag.mcp_server.vault_tools import _get_links_impl

    raw = await _get_links_impl(path="linker.md")
    result = json.loads(raw)

    assert result["status"] == "success"
    outgoing = result["data"]["outgoing"]
    assert "other" in outgoing
    assert "third" in outgoing


@pytest.mark.asyncio
async def test_get_links_finds_incoming_from_other_files(single_vault):
    from prism_rag.mcp_server.vault_tools import _get_links_impl

    raw = await _get_links_impl(path="target.md")
    result = json.loads(raw)

    assert result["status"] == "success"
    incoming = result["data"]["incoming"]
    assert any("a.md" in p for p in incoming)
    assert any("b.md" in p for p in incoming)


@pytest.mark.asyncio
async def test_get_links_self_file_not_in_incoming(single_vault):
    from prism_rag.mcp_server.vault_tools import _get_links_impl

    # self_ref.md references [[self_ref]] — it must NOT appear in its own incoming
    raw = await _get_links_impl(path="self_ref.md")
    result = json.loads(raw)

    assert result["status"] == "success"
    incoming = result["data"]["incoming"]
    assert not any("self_ref.md" in p for p in incoming)
