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
