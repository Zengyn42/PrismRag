"""Tests for name collision resolution (Section 4)."""
from __future__ import annotations

import logging

from prism_rag.ingest.ast_extractor import extract_ast, _build_doc_index
from prism_rag.ingest.vault_loader import load_vault
from prism_rag.store.graph import KnowledgeGraph


def test_collision_same_alias_logs_warning(tmp_path, caplog):
    """Two files sharing an alias 'foo', same priority tier → log warning."""
    a = tmp_path / "a.md"
    a.write_text("---\naliases: [foo]\n---\nA")
    b = tmp_path / "b.md"
    b.write_text("---\naliases: [foo]\n---\nB")

    docs = load_vault(tmp_path)
    caplog.set_level(logging.WARNING, logger="prism_rag.ingest.ast_extractor")
    _build_doc_index(docs)

    assert any("collision" in rec.message.lower() or "ambiguous" in rec.message.lower()
               for rec in caplog.records)


def test_wikilink_to_ambiguous_is_dropped(tmp_path, caplog):
    """[[foo]] wikilink pointing to an ambiguous alias should NOT produce an edge."""
    a = tmp_path / "a.md"
    a.write_text("---\naliases: [foo]\n---\nA")
    b = tmp_path / "b.md"
    b.write_text("---\naliases: [foo]\n---\nB")
    c = tmp_path / "c.md"
    c.write_text("See [[foo]]")

    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    caplog.set_level(logging.WARNING, logger="prism_rag.ingest.ast_extractor")
    extract_ast(graph, docs)

    # No edge from c to either a or b via the 'foo' alias
    assert not graph.g.has_edge("c", "a")
    assert not graph.g.has_edge("c", "b")


def test_knowledge_id_beats_filename(tmp_path):
    """Collision between knowledge_id=foo and a plain filename foo.md → knowledge_id wins."""
    a = tmp_path / "设计" / "foo.md"
    a.parent.mkdir()
    a.write_text("plain file called foo")
    b = tmp_path / "knowledge" / "KNOW-200-foo.md"
    b.parent.mkdir()
    b.write_text("---\nknowledge_id: foo\n---\nk-node aliased as foo")

    docs = load_vault(tmp_path)
    idx = _build_doc_index(docs)
    # "foo" key must resolve to the knowledge-id owner
    assert idx["foo"] == "foo"  # doc.id === knowledge_id === "foo"


def test_canonical_flag_beats_filename(tmp_path):
    """canonical: true frontmatter beats a colliding plain filename."""
    a = tmp_path / "notes" / "bar.md"
    a.parent.mkdir()
    a.write_text("plain bar")
    b = tmp_path / "docs" / "canonical-bar.md"
    b.parent.mkdir()
    b.write_text("---\naliases: [bar]\ncanonical: true\n---\nCanonical definition")

    docs = load_vault(tmp_path)
    idx = _build_doc_index(docs)
    assert idx["bar"] == "docs/canonical-bar"


def test_knowledge_dir_beats_plain_file(tmp_path):
    """File under knowledge/ wins over a colliding plain filename."""
    a = tmp_path / "random" / "baz.md"
    a.parent.mkdir()
    a.write_text("random baz")
    b = tmp_path / "knowledge" / "baz.md"
    b.parent.mkdir()
    b.write_text("knowledge baz")  # no knowledge_id, no canonical — just the directory signal

    docs = load_vault(tmp_path)
    idx = _build_doc_index(docs)
    assert idx["baz"] == "knowledge/baz"
