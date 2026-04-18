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
