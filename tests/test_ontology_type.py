"""Tests for ontology_type field (Section 5)."""
from __future__ import annotations

import typing

from prism_rag.ingest.ast_extractor import extract_ast
from prism_rag.ingest.vault_loader import load_vault
from prism_rag.store.graph import KnowledgeGraph, Node, OntologyType


def test_ontology_type_literal_values():
    values = set(typing.get_args(OntologyType))
    assert values == {
        "concept", "entity", "process", "tool", "project",
        "fact", "decision", "rule", "procedure", "relation",
        "unclassified",
    }


def test_node_ontology_type_default_none():
    n = Node(id="x", label="X", kind="note")
    assert n.ontology_type is None


def test_node_ontology_type_set():
    n = Node(id="x", label="X", kind="knowledge", ontology_type="decision")
    assert n.ontology_type == "decision"


def test_ontology_type_from_frontmatter(tmp_path):
    """frontmatter type: decision → Node.ontology_type=decision."""
    p = tmp_path / "decision.md"
    p.write_text(
        "---\n"
        "knowledge_id: KNOW-D1\n"
        "type: decision\n"
        "---\n\n"
        "A decision about X"
    )
    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)
    assert graph.g.nodes["KNOW-D1"]["ontology_type"] == "decision"


def test_invalid_type_becomes_unclassified(tmp_path):
    """Unknown type: value → ontology_type=unclassified."""
    p = tmp_path / "note.md"
    p.write_text("---\ntype: nonsense\n---\nbody")
    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)
    assert graph.g.nodes["note"]["ontology_type"] == "unclassified"


def test_no_type_leaves_ontology_type_none(tmp_path):
    """No type: frontmatter → ontology_type=None."""
    p = tmp_path / "note.md"
    p.write_text("no frontmatter body")
    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)
    ont = graph.g.nodes["note"].get("ontology_type")
    assert ont is None
