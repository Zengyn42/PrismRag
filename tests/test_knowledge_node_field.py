"""Tests for Node.knowledge_id field and obsidian_parser setting it."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from prism_rag.store.graph import Node, KnowledgeGraph


def test_node_knowledge_id_defaults_none():
    node = Node(id="test::foo", label="foo")
    assert node.knowledge_id is None


def test_node_knowledge_id_serializes():
    node = Node(id="KNOW-000042", label="test", knowledge_id="KNOW-000042")
    d = node.to_dict()
    assert d["knowledge_id"] == "KNOW-000042"


def test_node_knowledge_id_in_json_roundtrip(tmp_path):
    g = KnowledgeGraph()
    g.add_node(Node(id="KNOW-000042", label="test", knowledge_id="KNOW-000042", kind="knowledge"))
    path = tmp_path / "graph.json"
    g.save(path)
    g2 = KnowledgeGraph.load(path)
    data = g2.g.nodes["KNOW-000042"]
    assert data.get("knowledge_id") == "KNOW-000042"


def test_node_knowledge_id_none_preserved_in_roundtrip(tmp_path):
    g = KnowledgeGraph()
    g.add_node(Node(id="regular-note", label="note", kind="note"))
    path = tmp_path / "graph.json"
    g.save(path)
    g2 = KnowledgeGraph.load(path)
    data = g2.g.nodes["regular-note"]
    assert data.get("knowledge_id") is None


def test_obsidian_parser_sets_knowledge_id(tmp_path):
    """obsidian_parser should set node.knowledge_id from frontmatter."""
    from prism_rag.ingest.obsidian_parser import ObsidianParser
    from prism_rag.ingest.vault_loader import load_vault

    vault = tmp_path / "vault"
    vault.mkdir()
    doc = vault / "test.md"
    doc.write_text(textwrap.dedent("""\
        ---
        knowledge_id: KNOW-000001
        title: Test Knowledge
        ---
        Some content here.
    """))

    docs, _ = load_vault(vault)
    parser = ObsidianParser()
    result = parser.parse(docs, vault_root=vault)

    # Find the knowledge node
    know_nodes = [n for n in result.nodes if n.kind == "knowledge"]
    assert len(know_nodes) == 1
    assert know_nodes[0].knowledge_id == "KNOW-000001"
