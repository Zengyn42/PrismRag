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


def test_relations_frontmatter_produces_edges(tmp_path):
    """frontmatter.relations.{depends_on,supersedes,...} emit typed EXTRACTED edges."""
    a = tmp_path / "knowledge" / "KNOW-001-base.md"
    a.parent.mkdir()
    a.write_text("---\nknowledge_id: KNOW-001\n---\n\nBase concept")

    b = tmp_path / "knowledge" / "KNOW-042-dep.md"
    b.write_text(
        "---\n"
        "knowledge_id: KNOW-042\n"
        "relations:\n"
        "  depends_on: [KNOW-001]\n"
        "  supersedes: []\n"
        "---\n\nDepends on KNOW-001"
    )

    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)

    assert graph.g.has_edge("KNOW-042", "KNOW-001")
    edge = graph.g.edges["KNOW-042", "KNOW-001"]
    assert edge["relation"] == "depends_on"
    assert edge["confidence"] == "EXTRACTED"
    assert edge["source_pass"] == "ast"


def test_relations_supersedes_edge_type(tmp_path):
    a = tmp_path / "knowledge" / "KNOW-040.md"
    a.parent.mkdir()
    a.write_text("---\nknowledge_id: KNOW-040\n---\n\nOld")
    b = tmp_path / "knowledge" / "KNOW-100.md"
    b.write_text(
        "---\n"
        "knowledge_id: KNOW-100\n"
        "relations:\n"
        "  supersedes: [KNOW-040]\n"
        "---\n\nNew"
    )

    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)

    assert graph.g.edges["KNOW-100", "KNOW-040"]["relation"] == "supersedes"


def test_embed_false_skips_node(tmp_path):
    """Node with frontmatter embed: false must NOT appear in embeddable set."""
    from prism_rag.ingest.embedder import _get_embeddable_nodes

    a = tmp_path / "knowledge" / "KNOW-REL.md"
    a.parent.mkdir()
    a.write_text(
        "---\n"
        "knowledge_id: KNOW-REL\n"
        "type: relation\n"
        "embed: false\n"
        "---\n\nRelation-only node"
    )
    b = tmp_path / "note.md"
    b.write_text("regular content")

    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)

    embeddable_ids = {node_id for node_id, _ in _get_embeddable_nodes(graph)}

    # KNOW-REL has embed: false → excluded
    assert "KNOW-REL" not in embeddable_ids
    # note.md has no embed directive → included (kind="note" + has content)
    assert "note" in embeddable_ids


def test_embed_default_true_knowledge_node(tmp_path):
    """Knowledge node WITHOUT embed directive is still embeddable."""
    from prism_rag.ingest.embedder import _get_embeddable_nodes

    p = tmp_path / "KNOW-X.md"
    p.write_text("---\nknowledge_id: KNOW-X\n---\n\ncontent here")
    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)

    embeddable_ids = {node_id for node_id, _ in _get_embeddable_nodes(graph)}
    # KNOW-X is kind=knowledge, no embed: false → should be included
    assert "KNOW-X" in embeddable_ids
