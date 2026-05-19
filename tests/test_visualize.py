"""Tests for PrismRag v5.6 — Graph Visualization.

Covers:
  - Obsidian URI correct generation and URL encoding
  - OBSIDIAN_JS injection into the generated HTML
  - Portal nodes (context_ref) render as hexagon with amber color
  - Federation map HTML contains all namespace nodes
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Skip entire module if pyvis is not installed
pyvis = pytest.importorskip("pyvis", reason="pyvis not installed; skipping visualization tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph():
    """Build a minimal KnowledgeGraph with note, knowledge, and context_ref nodes."""
    from prism_rag.store.graph import KnowledgeGraph, Node

    kg = KnowledgeGraph()
    kg.add_node(Node(
        id="设计细节/PrismRag-v5.md",
        label="PrismRag v5 设计",
        kind="note",
        source_file="设计细节/PrismRag-v5.md",
        content="Some content",
    ))
    kg.add_node(Node(
        id="KNOW-000001",
        label="Fresh Per Call Decision",
        kind="knowledge",
        knowledge_id="KNOW-000001",
        source_file="knowledge/KNOW-000001.md",
        content="Knowledge body text",
    ))
    kg.add_node(Node(
        id="KNOW-000002",
        label="Atom with no source_file",
        kind="knowledge",
        knowledge_id="KNOW-000002",
        source_file="",   # missing source_file → synthesize path
        content="",
    ))
    kg.add_node(Node(
        id="ctx_ref_001",
        label="Context: Prior Decision",
        kind="context_ref",
        source_file="设计细节/PrismRag-v5.md",
        content="Context note text",
    ))
    return kg


def _make_multi_graph(tmp_path: Path):
    """Build a FederatedGraph with 'nimbus' and 'code' namespaces."""
    from prism_rag.store.graph import KnowledgeGraph, Node
    from prism_rag.store.federated import FederatedGraph
    from prism_rag.config import PrismRagSettings, GraphSource

    # nimbus graph
    nimbus_dir = tmp_path / "nimbus"
    nimbus_dir.mkdir()
    kg_nimbus = KnowledgeGraph()
    kg_nimbus.add_node(Node(id="note-1", label="Nimbus Note", kind="note"))
    kg_nimbus.save(nimbus_dir / "graph.json")

    # code graph
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    kg_code = KnowledgeGraph()
    kg_code.add_node(Node(id="fn-1", label="some_function", kind="function"))
    kg_code.save(code_dir / "graph.json")

    settings = PrismRagSettings(graphs=[
        GraphSource(namespace="nimbus", vault_path=nimbus_dir, data_dir=nimbus_dir),
        GraphSource(namespace="code", vault_path=code_dir, data_dir=code_dir),
    ])
    return FederatedGraph.load(settings.resolved_graphs)


# ---------------------------------------------------------------------------
# P1 — Obsidian URI generation
# ---------------------------------------------------------------------------

def test_obsidian_uri_note_node(tmp_path):
    """Note nodes get obsidian_uri pointing to vault file when vault_name is set."""
    from prism_rag.report.visualize import generate_html

    kg = _make_graph()
    out = tmp_path / "graph.html"
    generate_html(kg, out, vault_name="NimbusVault")

    html = out.read_text(encoding="utf-8")
    assert "obsidian://open" in html, "Obsidian URI scheme not found in HTML"
    assert "NimbusVault" in html, "Vault name not in obsidian URI"
    assert "PrismRag-v5.md" in html, "File path not in obsidian URI"


def test_obsidian_uri_url_encoding(tmp_path):
    """Spaces and special chars in file paths must be percent-encoded in the URI."""
    from prism_rag.report.visualize import generate_html
    from prism_rag.store.graph import KnowledgeGraph, Node

    kg = KnowledgeGraph()
    kg.add_node(Node(
        id="spaces/My Note File.md",
        label="My Note",
        kind="note",
        source_file="spaces/My Note File.md",
    ))
    out = tmp_path / "graph.html"
    generate_html(kg, out, vault_name="My Vault")

    html = out.read_text(encoding="utf-8")
    # Spaces must be percent-encoded (%20)
    assert "%20" in html, "Spaces in file path not URL-encoded"
    # Vault name with space must also be encoded
    assert "My%20Vault" in html, "Vault name spaces not URL-encoded"


def test_obsidian_uri_knowledge_node_synthesizes_path(tmp_path):
    """Knowledge nodes with empty source_file synthesize path from knowledge_id."""
    from prism_rag.report.visualize import generate_html

    kg = _make_graph()
    out = tmp_path / "graph.html"
    generate_html(kg, out, vault_name="NimbusVault")

    html = out.read_text(encoding="utf-8")
    # KNOW-000002 has empty source_file → should synthesize "knowledge/KNOW-000002.md"
    assert "KNOW-000002" in html
    assert "knowledge%2FKNOW-000002.md" in html or "knowledge/KNOW-000002" in html, (
        "Synthesized path for knowledge node with empty source_file not found"
    )


def test_no_obsidian_uri_when_vault_name_absent(tmp_path):
    """Without vault_name, no obsidian:// URIs should appear in the HTML."""
    from prism_rag.report.visualize import generate_html

    kg = _make_graph()
    out = tmp_path / "graph.html"
    generate_html(kg, out, vault_name=None)

    html = out.read_text(encoding="utf-8")
    assert "obsidian://" not in html, "Obsidian URI found despite vault_name=None"


# ---------------------------------------------------------------------------
# P2 — OBSIDIAN_JS injection
# ---------------------------------------------------------------------------

def test_obsidian_js_injected(tmp_path):
    """OBSIDIAN_JS block must be present in the generated HTML before </body>."""
    from prism_rag.report.visualize import generate_html

    kg = _make_graph()
    out = tmp_path / "graph.html"
    generate_html(kg, out, vault_name="NimbusVault")

    html = out.read_text(encoding="utf-8")
    assert "_prismNodeData" in html, "JS node data map not injected"
    # The JS must appear before </body>
    js_pos = html.find("_prismNodeData")
    body_close_pos = html.rfind("</body>")
    assert js_pos < body_close_pos, "_prismNodeData JS must appear before </body>"


def test_obsidian_js_contains_hash_focus(tmp_path):
    """Hash-based focus logic (stabilizationIterationsDone) must be present."""
    from prism_rag.report.visualize import generate_html

    kg = _make_graph()
    out = tmp_path / "graph.html"
    generate_html(kg, out, vault_name="NimbusVault")

    html = out.read_text(encoding="utf-8")
    assert "stabilizationIterationsDone" in html, "Hash focus handler not injected"
    assert "window.location.hash" in html, "Hash detection not injected"


def test_obsidian_js_node_data_valid_json(tmp_path):
    """_prismNodeData value must be valid JSON parseable data."""
    from prism_rag.report.visualize import generate_html
    import re

    kg = _make_graph()
    out = tmp_path / "graph.html"
    generate_html(kg, out, vault_name="NimbusVault")

    html = out.read_text(encoding="utf-8")
    # Extract the JSON from _prismNodeData = {...};
    m = re.search(r'var _prismNodeData = ({.*?});', html, re.DOTALL)
    assert m is not None, "_prismNodeData assignment not found in HTML"
    data = json.loads(m.group(1))
    assert isinstance(data, dict)
    # Note node should have obsidian_uri key
    note_id = "设计细节/PrismRag-v5.md"
    assert note_id in data
    assert "obsidian_uri" in data[note_id]
    assert data[note_id]["obsidian_uri"].startswith("obsidian://open")


# ---------------------------------------------------------------------------
# P3 — Portal nodes (context_ref)
# ---------------------------------------------------------------------------

def test_portal_node_hexagon_shape(tmp_path):
    """context_ref nodes must use hexagon shape (checked via pyvis internal state)."""
    from prism_rag.report.visualize import generate_html
    from pyvis.network import Network

    kg = _make_graph()
    # We check pyvis internals before HTML generation by calling generate_html
    # and then inspecting the HTML output for vis.js shape attribute.
    out = tmp_path / "graph.html"
    generate_html(kg, out, vault_name="NimbusVault")

    html = out.read_text(encoding="utf-8")
    # vis.js serialises node shape as "shape":"hexagon" in the nodes DataSet
    assert '"hexagon"' in html, "Hexagon shape not found for portal node"


def test_portal_node_amber_color(tmp_path):
    """context_ref nodes must use the portal amber color #F5A623."""
    from prism_rag.report.visualize import generate_html

    kg = _make_graph()
    out = tmp_path / "graph.html"
    generate_html(kg, out, vault_name="NimbusVault")

    html = out.read_text(encoding="utf-8")
    assert "#F5A623" in html.upper() or "f5a623" in html.lower(), (
        "Portal amber color #F5A623 not found in HTML"
    )


def test_portal_node_label_prefix(tmp_path):
    """context_ref node labels must be prefixed with ⬡ (U+2B21).

    pyvis serialises node labels via json.dumps with ensure_ascii=True by default,
    so ⬡ appears as the JSON escape \\u2b21 in the HTML source.  Both the literal
    character and its JSON escape form are accepted.
    """
    from prism_rag.report.visualize import generate_html

    kg = _make_graph()
    out = tmp_path / "graph.html"
    generate_html(kg, out, vault_name="NimbusVault")

    html = out.read_text(encoding="utf-8")
    # Accept both the literal glyph and the JSON/JS Unicode escape sequence.
    has_literal = "⬡" in html
    has_escape = "\\u2b21" in html.lower() or "u2b21" in html.lower()
    assert has_literal or has_escape, (
        "Portal ⬡ prefix (U+2B21) not found in HTML — "
        "expected either literal '⬡' or JSON escape '\\u2b21'"
    )


def test_portal_node_has_portal_href_in_js(tmp_path):
    """context_ref nodes must appear in _prismNodeData with portal_href."""
    from prism_rag.report.visualize import generate_html
    import re

    kg = _make_graph()
    out = tmp_path / "graph.html"
    generate_html(kg, out, vault_name="NimbusVault")

    html = out.read_text(encoding="utf-8")
    m = re.search(r'var _prismNodeData = ({.*?});', html, re.DOTALL)
    assert m is not None
    data = json.loads(m.group(1))
    assert "ctx_ref_001" in data, "context_ref node not in _prismNodeData"
    assert "portal_href" in data["ctx_ref_001"], "portal_href missing from context_ref JS data"


