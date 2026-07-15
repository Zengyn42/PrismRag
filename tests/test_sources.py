"""Tests for the prism_rag.sources pluggable extractor layer.

Covers:
  - Three source kinds independently toggleable
  - DocsSourceExtractor excludes knowledge/ files
  - MemorySourceExtractor raises clear error when atomize unavailable
  - KnotLoader loads only knowledge files
  - Old config (no sources field) preserves backward compat
  - GraphSource.sources validation
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prism_rag.config import GraphSource
from prism_rag.sources.base import VALID_SOURCES, SourceKind
from prism_rag.sources.code_source import CodeSourceExtractor
from prism_rag.sources.docs_source import DocsSourceExtractor, _is_knot_file
from prism_rag.sources.knot_loader import KnotLoader
from prism_rag.sources.memory_source import AtomizeUnavailableError, MemorySourceExtractor


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def vault_with_knowledge(tmp_path: Path) -> Path:
    """Create a minimal vault with both regular docs and knowledge/ files."""
    # Regular doc
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "daily.md").write_text(
        "---\ntitle: Daily Note\n---\n\nSome daily notes content."
    )
    (tmp_path / "readme.md").write_text(
        "---\ntitle: README\n---\n\n# Project README"
    )

    # Knowledge files (KNOT)
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "KNOW-000001.md").write_text(
        "---\nknowledge_id: KNOW-000001\ntitle: Atomic Concept\n---\n\nThis is a KNOT node."
    )
    (tmp_path / "knowledge" / "KNOW-000002.md").write_text(
        "---\nknowledge_id: KNOW-000002\ntitle: Another Concept\n---\n\nSecond KNOT."
    )

    # A file outside knowledge/ but with knowledge_id (edge case)
    (tmp_path / "notes" / "inline_knowledge.md").write_text(
        "---\nknowledge_id: KNOW-000003\ntitle: Inline Knowledge\n---\n\nKnowledge outside knowledge/ dir."
    )

    return tmp_path


@pytest.fixture
def code_repo(tmp_path: Path) -> Path:
    """Create a minimal Python repository."""
    (tmp_path / "main.py").write_text(
        "def hello():\n    return 'world'\n"
    )
    (tmp_path / "utils.py").write_text(
        "def add(a, b):\n    return a + b\n"
    )
    return tmp_path


# ── SourceKind & VALID_SOURCES ────────────────────────────────────────────────


def test_valid_sources_contains_all_kinds():
    assert "code" in VALID_SOURCES
    assert "docs" in VALID_SOURCES
    assert "memory" in VALID_SOURCES
    assert "knot" in VALID_SOURCES


# ── DocsSourceExtractor ──────────────────────────────────────────────────────


def test_docs_discover_excludes_knowledge_dir(vault_with_knowledge: Path):
    ext = DocsSourceExtractor()
    paths = ext.discover(vault_with_knowledge)

    # Should find regular docs but NOT knowledge/ files
    names = [p.name for p in paths]
    assert "daily.md" in names
    assert "readme.md" in names
    assert "KNOW-000001.md" not in names
    assert "KNOW-000002.md" not in names


def test_docs_parse_excludes_knowledge_files(vault_with_knowledge: Path):
    ext = DocsSourceExtractor()
    result = ext.parse(vault_with_knowledge)

    node_ids = {n.id for n in result.nodes}
    node_kinds = {n.kind for n in result.nodes}

    # Should not contain any knowledge nodes
    assert "knowledge" not in node_kinds or all(
        n.kind != "knowledge" for n in result.nodes
        if not n.id.startswith("nimbus:vault:")  # exclude root node
    )
    # Should not contain KNOW- IDs
    assert not any(nid.startswith("KNOW-") for nid in node_ids)
    # Inline knowledge file should also be excluded (has knowledge_id frontmatter)
    assert "KNOW-000003" not in node_ids


def test_docs_parse_includes_regular_notes(vault_with_knowledge: Path):
    ext = DocsSourceExtractor()
    result = ext.parse(vault_with_knowledge)

    # Should contain regular note nodes
    node_labels = {n.label for n in result.nodes}
    assert "daily" in node_labels or "Daily Note" in node_labels
    assert "readme" in node_labels or "README" in node_labels


# ── KnotLoader ────────────────────────────────────────────────────────────────


def test_knot_loader_discovers_knowledge_files(vault_with_knowledge: Path):
    loader = KnotLoader()
    paths = loader.discover(vault_with_knowledge)

    names = [p.name for p in paths]
    assert "KNOW-000001.md" in names
    assert "KNOW-000002.md" in names
    # Regular docs should NOT be in knot discovery
    assert "daily.md" not in names
    assert "readme.md" not in names


def test_knot_loader_parse_returns_knowledge_nodes(vault_with_knowledge: Path):
    loader = KnotLoader()
    result = loader.parse(vault_with_knowledge)

    # Should find knowledge nodes
    knowledge_nodes = [n for n in result.nodes if n.kind == "knowledge"]
    assert len(knowledge_nodes) >= 2

    # IDs should be KNOW- prefixed
    kid_ids = {n.id for n in knowledge_nodes}
    assert "KNOW-000001" in kid_ids
    assert "KNOW-000002" in kid_ids


def test_knot_loader_empty_vault(tmp_path: Path):
    """KnotLoader on a vault with no knowledge files returns empty result."""
    (tmp_path / "notes.md").write_text("---\ntitle: Regular\n---\nContent.")
    loader = KnotLoader()
    result = loader.parse(tmp_path)
    assert len(result.nodes) == 0


# ── CodeSourceExtractor ──────────────────────────────────────────────────────


def test_code_source_discover(code_repo: Path):
    ext = CodeSourceExtractor()
    paths = ext.discover(code_repo)
    names = {p.name for p in paths}
    assert "main.py" in names
    assert "utils.py" in names


def test_code_source_parse(code_repo: Path):
    ext = CodeSourceExtractor()
    result = ext.parse(code_repo)

    # Should produce nodes for modules and functions
    assert len(result.nodes) > 0
    node_kinds = {n.kind for n in result.nodes}
    assert "module" in node_kinds
    assert "function" in node_kinds


def test_code_source_kind():
    ext = CodeSourceExtractor()
    assert ext.kind == "code"


# ── MemorySourceExtractor ────────────────────────────────────────────────────


def test_memory_source_raises_atomize_error(tmp_path: Path):
    """Memory source must raise clear error when atomize is unavailable."""
    (tmp_path / "MEMORY.md").write_text("# Memory\nSome agent memory.")

    ext = MemorySourceExtractor(memory_paths=[tmp_path])
    # discover should work
    paths = ext.discover(tmp_path)
    assert any(p.name == "MEMORY.md" for p in paths)

    # parse must raise
    with pytest.raises(AtomizeUnavailableError, match="memory source requires atomize"):
        ext.parse(tmp_path)


def test_memory_source_discover_files_and_dirs(tmp_path: Path):
    """Memory source can discover from both file paths and directory paths."""
    (tmp_path / "session.md").write_text("Session log.")
    subdir = tmp_path / "logs"
    subdir.mkdir()
    (subdir / "log1.md").write_text("Log 1.")
    (subdir / "log2.md").write_text("Log 2.")
    (subdir / "data.txt").write_text("Not markdown.")

    ext = MemorySourceExtractor(memory_paths=[tmp_path / "session.md", subdir])
    paths = ext.discover(tmp_path)
    names = {p.name for p in paths}
    assert "session.md" in names
    assert "log1.md" in names
    assert "log2.md" in names
    assert "data.txt" not in names


def test_memory_source_kind():
    ext = MemorySourceExtractor()
    assert ext.kind == "memory"


# ── GraphSource config ───────────────────────────────────────────────────────


def test_graph_source_default_sources():
    """GraphSource defaults to ['docs'] for backward compat."""
    gs = GraphSource(namespace="test", vault_path=Path("/tmp"), data_dir=Path("/tmp"))
    assert gs.sources == ["docs"]


def test_graph_source_custom_sources():
    gs = GraphSource(
        namespace="test",
        vault_path=Path("/tmp"),
        data_dir=Path("/tmp"),
        sources=["docs", "code", "knot"],
    )
    assert gs.sources == ["docs", "code", "knot"]


def test_graph_source_sources_from_string():
    """Sources can be parsed from comma-separated string (env var style)."""
    gs = GraphSource(
        namespace="test",
        vault_path=Path("/tmp"),
        data_dir=Path("/tmp"),
        sources="docs,code",  # type: ignore[arg-type]
    )
    assert gs.sources == ["docs", "code"]


def test_graph_source_invalid_source_raises():
    with pytest.raises(ValueError, match="Invalid source"):
        GraphSource(
            namespace="test",
            vault_path=Path("/tmp"),
            data_dir=Path("/tmp"),
            sources=["docs", "invalid_source"],
        )


def test_graph_source_atomize_docs_default():
    gs = GraphSource(namespace="test", vault_path=Path("/tmp"), data_dir=Path("/tmp"))
    assert gs.atomize_docs is False


def test_graph_source_memory_paths_default():
    gs = GraphSource(namespace="test", vault_path=Path("/tmp"), data_dir=Path("/tmp"))
    assert gs.memory_paths == []


# ── Backward compatibility: old config without sources field ─────────────────


def test_old_config_without_sources_field():
    """GraphSource created without sources field behaves like old config."""
    gs = GraphSource(
        namespace="nimbus",
        vault_path=Path("/home/test/vault"),
        data_dir=Path("/home/test/data"),
    )
    # Default is docs-only, preserving backward compat
    assert gs.sources == ["docs"]
    assert gs.atomize_docs is False
    assert gs.memory_paths == []
    assert gs.graph_path == Path("/home/test/data/graph.json")


def test_old_config_json_roundtrip():
    """GraphSource from JSON without sources field works (PRISM_GRAPHS env var)."""
    import json
    old_style_json = json.dumps({
        "namespace": "nimbus",
        "vault_path": "/tmp/vault",
        "data_dir": "/tmp/data",
    })
    gs = GraphSource.model_validate_json(old_style_json)
    assert gs.sources == ["docs"]
    assert gs.namespace == "nimbus"


# ── _is_knot_file helper ────────────────────────────────────────────────────


def test_is_knot_file_by_frontmatter(tmp_path: Path):
    from prism_rag.ingest.vault_loader import VaultDocument
    doc = VaultDocument(
        path=tmp_path / "notes" / "test.md",
        vault_root=tmp_path,
        content="Test",
        frontmatter={"knowledge_id": "KNOW-000001"},
    )
    assert _is_knot_file(doc) is True


def test_is_knot_file_by_directory(tmp_path: Path):
    from prism_rag.ingest.vault_loader import VaultDocument
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True)
    doc = VaultDocument(
        path=knowledge_dir / "test.md",
        vault_root=tmp_path,
        content="Test",
        frontmatter={},
    )
    assert _is_knot_file(doc) is True


def test_is_not_knot_file(tmp_path: Path):
    from prism_rag.ingest.vault_loader import VaultDocument
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir(parents=True)
    doc = VaultDocument(
        path=notes_dir / "regular.md",
        vault_root=tmp_path,
        content="Regular note",
        frontmatter={"title": "Regular"},
    )
    assert _is_knot_file(doc) is False


# ── Integration: docs + knot = full vault coverage ──────────────────────────


def test_docs_plus_knot_covers_full_vault(vault_with_knowledge: Path):
    """docs + knot extractors together should cover all nodes that the old
    ObsidianParser would have produced."""
    from prism_rag.ingest.obsidian_parser import ObsidianParser

    # Old behavior: ObsidianParser on entire vault
    old_parser = ObsidianParser()
    old_result = old_parser.parse(vault_with_knowledge)
    old_node_ids = {n.id for n in old_result.nodes if n.kind in ("note", "knowledge")}

    # New behavior: docs + knot
    docs_ext = DocsSourceExtractor()
    docs_result = docs_ext.parse(vault_with_knowledge)
    docs_ids = {n.id for n in docs_result.nodes if n.kind in ("note", "knowledge")}

    knot_loader = KnotLoader()
    knot_result = knot_loader.parse(vault_with_knowledge)
    knot_ids = {n.id for n in knot_result.nodes if n.kind in ("note", "knowledge")}

    combined = docs_ids | knot_ids
    # Combined should cover all old nodes
    assert old_node_ids == combined, (
        f"Missing: {old_node_ids - combined}, Extra: {combined - old_node_ids}"
    )
