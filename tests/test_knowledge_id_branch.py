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
