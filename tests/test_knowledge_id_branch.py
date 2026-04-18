"""Tests for the knowledge_id branch (Section 3)."""
from __future__ import annotations

from pathlib import Path

from prism_rag.ingest.ast_extractor import extract_ast
from prism_rag.ingest.vault_loader import VaultDocument, load_vault
from prism_rag.store.graph import KnowledgeGraph, Node


def test_nodekind_accepts_knowledge():
    """NodeKind literal must include 'knowledge' for Phase 2 atomic nodes."""
    node = Node(id="KNOW-001", label="K1", kind="knowledge")
    assert node.kind == "knowledge"


def test_vault_document_id_uses_knowledge_id(tmp_path):
    """When frontmatter has knowledge_id, VaultDocument.id returns it."""
    p = tmp_path / "sub" / "my-note.md"
    p.parent.mkdir()
    p.write_text("---\nknowledge_id: KNOW-042\n---\n\nbody")
    doc = VaultDocument.from_path(p, tmp_path)
    assert doc.id == "KNOW-042"


def test_vault_document_id_falls_back_to_path(tmp_path):
    """When no knowledge_id, VaultDocument.id is the relative path stem."""
    p = tmp_path / "sub" / "my-note.md"
    p.parent.mkdir()
    p.write_text("no frontmatter")
    doc = VaultDocument.from_path(p, tmp_path)
    assert doc.id == "sub/my-note"


def test_knowledge_id_wikilink_resolves(tmp_path):
    """[[KNOW-042]] in another file must resolve to the node with knowledge_id=KNOW-042."""
    # File with knowledge_id
    a = tmp_path / "knowledge" / "KNOW-042-session.md"
    a.parent.mkdir()
    a.write_text("---\nknowledge_id: KNOW-042\n---\n\nBody of K42")

    # File linking to KNOW-042 by its id
    b = tmp_path / "设计细节" / "some-doc.md"
    b.parent.mkdir()
    b.write_text("See [[KNOW-042]] for details.")

    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)

    # KNOW-042 exists as a node
    assert "KNOW-042" in graph.g.nodes
    assert graph.g.nodes["KNOW-042"]["kind"] == "knowledge"

    # Edge from some-doc to KNOW-042
    assert graph.g.has_edge("设计细节/some-doc", "KNOW-042")


def test_regular_note_still_kind_note(tmp_path):
    """Files without knowledge_id get kind='note'."""
    p = tmp_path / "regular.md"
    p.write_text("no frontmatter, just body")
    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)
    assert graph.g.nodes["regular"]["kind"] == "note"
