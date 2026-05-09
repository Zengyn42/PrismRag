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
    docs, _ = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)
    assert graph.g.nodes["KNOW-D1"]["ontology_type"] == "decision"


def test_invalid_type_becomes_unclassified(tmp_path):
    """Unknown type: value → ontology_type=unclassified."""
    p = tmp_path / "note.md"
    p.write_text("---\ntype: nonsense\n---\nbody")
    docs, _ = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)
    assert graph.g.nodes["note"]["ontology_type"] == "unclassified"


def test_no_type_leaves_ontology_type_none(tmp_path):
    """No type: frontmatter → ontology_type=None."""
    p = tmp_path / "note.md"
    p.write_text("no frontmatter body")
    docs, _ = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)
    ont = graph.g.nodes["note"].get("ontology_type")
    assert ont is None


def test_search_knowledge_filters_by_ontology_type(tmp_path, monkeypatch):
    """search_knowledge with ontology_type='decision' excludes non-decisions."""
    import json
    from prism_rag.config import PrismRagSettings, GraphSource
    from prism_rag.mcp_server import server as mcp_server

    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()

    # Link them so BFS from the decision node can reach the fact node
    (vault / "dec.md").write_text(
        "---\nknowledge_id: KNOW-D\ntype: decision\n---\nA decision about X. [[fact]]"
    )
    (vault / "fact.md").write_text(
        "---\nknowledge_id: KNOW-F\ntype: fact\n---\nA fact about X"
    )

    # Build and persist graph
    from prism_rag.ingest.vault_loader import load_vault as lv
    from prism_rag.ingest.ast_extractor import extract_ast
    from prism_rag.store.graph import KnowledgeGraph as KG
    g = KG()
    _lv_docs, _ = lv(vault)
    extract_ast(g, _lv_docs)
    g.save(data / "graph.json")

    # Reset server state
    mcp_server._federated = None

    settings = PrismRagSettings(
        graphs=[GraphSource(namespace="default", vault_path=vault, data_dir=data)],
    )
    monkeypatch.setattr(
        "prism_rag.mcp_server.server.PrismRagSettings",
        lambda: settings,
    )

    # Query "dec" matches the decision node; BFS will also reach the linked fact node.
    # With ontology_type="decision" filter, KNOW-F (fact) must be excluded.
    result = mcp_server.search_knowledge(query="dec", ontology_type="decision")
    parsed = json.loads(result)
    # Be lenient on response shape — just verify KNOW-F is NOT in results and KNOW-D IS.
    response_text = result
    assert "KNOW-F" not in response_text
    assert "KNOW-D" in response_text
