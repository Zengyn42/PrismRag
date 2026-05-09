"""End-to-end smoke tests for the Phase 1 MVP pipeline.

These tests exercise:
1. Basic graph construction and JSON roundtrip (unit)
2. AST extraction on synthetic markdown (unit)
3. Full pipeline on the real NimbusVault (integration, skipped if unavailable)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from prism_rag.cluster.leiden import run_leiden
from prism_rag.ingest.ast_extractor import (
    _extract_inline_tags,
    _extract_wikilinks,
    extract_ast,
)
from prism_rag.ingest.vault_loader import VaultDocument, discover_markdown_files, load_vault
from prism_rag.report.graph_report import generate_report
from prism_rag.store.graph import Edge, KnowledgeGraph, Node

VAULT_PATH = Path(os.environ.get("PRISM_VAULT_PATH", Path.home() / "Foundation" / "NimbusVault"))


# ═══════════════════════════════════════════════════════════════════
# Unit tests — graph store
# ═══════════════════════════════════════════════════════════════════


def test_graph_add_node_and_edge():
    g = KnowledgeGraph()
    g.add_node(Node(id="a", label="A", kind="note", tokens=100))
    g.add_node(Node(id="b", label="B", kind="note", tokens=200))
    g.add_edge(Edge(source="a", target="b", relation="links_to", confidence="EXTRACTED"))
    assert g.node_count == 2
    assert g.edge_count == 1


def test_graph_add_edge_creates_stub_for_missing_endpoints():
    g = KnowledgeGraph()
    g.add_edge(Edge(source="new1", target="new2", relation="links_to", confidence="EXTRACTED"))
    assert g.node_count == 2
    assert g.edge_count == 1
    assert g.g.nodes["new1"].get("label") == "new1"


def test_graph_json_roundtrip(tmp_path):
    g = KnowledgeGraph()
    g.add_node(Node(id="a", label="A", kind="note", tokens=100, content="hello"))
    g.add_node(Node(id="b", label="B", kind="tag"))
    g.add_edge(
        Edge(
            source="a",
            target="b",
            relation="tagged_as",
            confidence="EXTRACTED",
            confidence_score=1.0,
            weight=1.0,
            source_pass="ast",
        )
    )
    path = tmp_path / "graph.json"
    g.save(path)
    assert path.exists()

    loaded = KnowledgeGraph.load(path)
    assert loaded.node_count == 2
    assert loaded.edge_count == 1
    assert loaded.g.nodes["a"]["content"] == "hello"
    assert loaded.g.nodes["b"]["kind"] == "tag"
    # Edge data preserved
    edge_data = loaded.g.edges["a", "b"]
    assert edge_data["relation"] == "tagged_as"
    assert edge_data["source_pass"] == "ast"


# ═══════════════════════════════════════════════════════════════════
# Unit tests — AST regex extractors
# ═══════════════════════════════════════════════════════════════════


def test_extract_wikilinks_basic():
    text = "See [[Note A]] and [[Note B|display]]."
    result = _extract_wikilinks(text)
    assert ("Note A", "links_to") in result
    assert ("Note B", "links_to") in result


def test_extract_wikilinks_section_and_block():
    text = "See [[Note#Heading]] and [[Note^abc123]]."
    result = _extract_wikilinks(text)
    relations = [r for _, r in result]
    assert "links_to_section" in relations
    assert "links_to_block" in relations


def test_extract_wikilinks_embed():
    text = "Image: ![[diagram.png]]"
    result = _extract_wikilinks(text)
    assert result == [("diagram.png", "embeds")]


def test_extract_inline_tags():
    text = "This is #tagged and #another/nested but not `#in-code`."
    tags = _extract_inline_tags(text)
    assert "tagged" in tags
    assert "another/nested" in tags
    assert "in-code" not in tags


def test_extract_inline_tags_ignores_code_blocks():
    text = """
normal #good-tag here
```python
# This is a comment, not a #bad-tag
```
more #another-good
"""
    tags = _extract_inline_tags(text)
    assert "good-tag" in tags
    assert "another-good" in tags
    assert "bad-tag" not in tags


# ═══════════════════════════════════════════════════════════════════
# Unit tests — AST extraction on synthetic vault
# ═══════════════════════════════════════════════════════════════════


def _make_doc(path: Path, vault_root: Path, content: str) -> VaultDocument:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return VaultDocument.from_path(path, vault_root)


def test_extract_ast_builds_wikilink_edges(tmp_path):
    vault = tmp_path / "vault"
    doc_a = _make_doc(vault / "A.md", vault, "# A\nSee [[B]] for more.")
    doc_b = _make_doc(vault / "B.md", vault, "# B\nContent")
    docs = [doc_a, doc_b]

    g = KnowledgeGraph()
    extract_ast(g, docs)

    assert g.node_count >= 2  # A, B (+ maybe tags)
    # There should be an edge from A → B with relation links_to
    assert g.g.has_edge("A", "B")
    edge = g.g.edges["A", "B"]
    assert edge["relation"] == "links_to"
    assert edge["confidence"] == "EXTRACTED"


def test_extract_ast_creates_tag_nodes(tmp_path):
    vault = tmp_path / "vault"
    doc = _make_doc(vault / "Note.md", vault, "Content #architecture #design")
    g = KnowledgeGraph()
    extract_ast(g, [doc])

    assert "tag:architecture" in g.g.nodes
    assert "tag:design" in g.g.nodes
    assert g.g.has_edge("Note", "tag:architecture")
    assert g.g.has_edge("Note", "tag:design")


def test_extract_ast_handles_frontmatter(tmp_path):
    vault = tmp_path / "vault"
    content = """---
tags: [planning, roadmap]
category: Meta
aliases: [PlanDoc]
---

# Main

Some [[Other]] content.
"""
    doc = _make_doc(vault / "Plan.md", vault, content)
    g = KnowledgeGraph()
    extract_ast(g, [doc])

    # Frontmatter tags
    assert "tag:planning" in g.g.nodes
    assert "tag:roadmap" in g.g.nodes
    # Category
    assert "category:Meta" in g.g.nodes
    assert g.g.has_edge("Plan", "category:Meta")


def test_extract_ast_dangling_wikilink_skipped(tmp_path):
    vault = tmp_path / "vault"
    doc = _make_doc(vault / "Lonely.md", vault, "See [[Nonexistent]].")
    g = KnowledgeGraph()
    extract_ast(g, [doc])
    # Dangling link should not create an edge; graph only has the one note
    assert "Nonexistent" not in g.g.nodes
    note_nodes = [n for n, d in g.g.nodes(data=True) if d.get("kind") == "note"]
    assert note_nodes == ["Lonely"]


# ═══════════════════════════════════════════════════════════════════
# Unit tests — Leiden clustering
# ═══════════════════════════════════════════════════════════════════


def test_leiden_on_small_graph():
    g = KnowledgeGraph()
    # Two densely-connected components: {a,b,c} and {d,e,f}
    for nid in ("a", "b", "c", "d", "e", "f"):
        g.add_node(Node(id=nid, label=nid.upper(), kind="note"))
    for s, t in [("a", "b"), ("b", "c"), ("a", "c"), ("d", "e"), ("e", "f"), ("d", "f")]:
        g.add_edge(Edge(source=s, target=t, relation="links_to", confidence="EXTRACTED"))

    run_leiden(g, seed=42)
    assert len(g.communities) >= 1  # at least one community
    # Every node should have a community_id
    for nid in g.g.nodes():
        assert g.g.nodes[nid].get("community_id") is not None


def test_leiden_empty_graph_is_noop():
    g = KnowledgeGraph()
    n = run_leiden(g)
    assert n == 0
    assert len(g.communities) == 0


# ═══════════════════════════════════════════════════════════════════
# Unit tests — Report generation
# ═══════════════════════════════════════════════════════════════════


def test_generate_report_writes_file(tmp_path):
    g = KnowledgeGraph()
    g.add_node(Node(id="a", label="A", kind="note"))
    g.add_node(Node(id="b", label="B", kind="note"))
    g.add_edge(Edge(source="a", target="b", relation="links_to", confidence="EXTRACTED"))
    run_leiden(g, seed=42)

    out = tmp_path / "GRAPH_REPORT.md"
    content = generate_report(g, out)
    assert out.exists()
    assert "NimbusVault 知识图报告" in content
    assert "God Nodes" in content


# ═══════════════════════════════════════════════════════════════════
# Integration test — real NimbusVault
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not VAULT_PATH.exists(),
    reason=f"NimbusVault not found at {VAULT_PATH}",
)
def test_end_to_end_on_real_nimbus_vault(tmp_path):
    """Run Pass 1 + Pass 4 + Pass 5 on the real NimbusVault, verify non-empty output."""
    paths = discover_markdown_files(VAULT_PATH)
    assert len(paths) > 5, f"Expected >5 md files in NimbusVault, got {len(paths)}"

    docs, _ = load_vault(VAULT_PATH)
    assert len(docs) == len(paths)

    g = KnowledgeGraph()
    extract_ast(g, docs)
    assert g.node_count > 0
    assert g.edge_count > 0, "Expected at least one EXTRACTED edge (wikilinks or tags)"

    run_leiden(g, seed=42)
    assert len(g.communities) > 0

    # Every note should have a community_id
    note_nodes = [n for n, d in g.g.nodes(data=True) if d.get("kind") == "note"]
    assert len(note_nodes) > 0
    with_comm = [n for n in note_nodes if g.g.nodes[n].get("community_id")]
    assert len(with_comm) == len(note_nodes), "Every note should be assigned to a community"

    # Persistence + report
    graph_out = tmp_path / "graph.json"
    report_out = tmp_path / "GRAPH_REPORT.md"
    g.save(graph_out)
    generate_report(g, report_out, vault_root=VAULT_PATH)

    assert graph_out.exists()
    assert report_out.exists()
    assert graph_out.stat().st_size > 100
    assert report_out.stat().st_size > 100

    # Load back and verify roundtrip
    loaded = KnowledgeGraph.load(graph_out)
    assert loaded.node_count == g.node_count
    assert loaded.edge_count == g.edge_count
