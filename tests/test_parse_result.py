"""T3.9 — tests for ParseResult Pydantic validation and iron law (D6)."""

import pytest
from pydantic import ValidationError

from prism_rag.ingest.base_tree import ParseTree, TreeNode
from prism_rag.ingest.parse_result import (
    EdgeRecord,
    NodeRecord,
    ParseResult,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _node(id="nimbus::a", namespace="nimbus", kind="note", confidence_tier="EXTRACTED", confidence=1.0):
    return NodeRecord(
        id=id,
        namespace=namespace,
        kind=kind,
        label="A",
        content="content",
        source_file="a.md",
        confidence_tier=confidence_tier,
        confidence=confidence,
    )


def _edge(src="nimbus::a", tgt="nimbus::b", kind="contains", confidence_tier="EXTRACTED", confidence=1.0, evidence=None):
    return EdgeRecord(
        source_id=src,
        target_id=tgt,
        kind=kind,
        confidence_tier=confidence_tier,
        confidence=confidence,
        evidence=evidence or [],
    )


def _simple_tree(namespace="nimbus"):
    root = TreeNode(
        id=f"{namespace}::root",
        namespace=namespace,
        kind="note" if namespace != "conv" else "session",
        label="Root",
        content="root content",
        source_file="root.md",
    )
    return ParseTree(root=root, namespace=namespace, source_file="root.md")


# ── NodeRecord validation ─────────────────────────────────────────────────────


def test_node_extracted_valid():
    n = _node(confidence_tier="EXTRACTED", confidence=1.0)
    assert n.confidence_tier == "EXTRACTED"


def test_node_extracted_boundary():
    n = _node(confidence_tier="EXTRACTED", confidence=0.95)
    assert n.confidence == 0.95


def test_node_extracted_below_range_raises():
    with pytest.raises(ValidationError, match="confidence_tier=EXTRACTED"):
        _node(confidence_tier="EXTRACTED", confidence=0.94)


def test_node_inferred_valid():
    n = _node(confidence_tier="INFERRED", confidence=0.80)
    assert n.confidence_tier == "INFERRED"


def test_node_inferred_range_low():
    n = _node(confidence_tier="INFERRED", confidence=0.30)
    assert n.confidence == 0.30


def test_node_inferred_above_range_raises():
    with pytest.raises(ValidationError, match="confidence_tier=INFERRED"):
        _node(confidence_tier="INFERRED", confidence=0.95)


def test_node_ambiguous_valid():
    n = _node(confidence_tier="AMBIGUOUS", confidence=0.10)
    assert n.confidence_tier == "AMBIGUOUS"


def test_node_ambiguous_above_range_raises():
    with pytest.raises(ValidationError, match="confidence_tier=AMBIGUOUS"):
        _node(confidence_tier="AMBIGUOUS", confidence=0.30)


# ── EdgeRecord validation ─────────────────────────────────────────────────────


def test_edge_extracted_valid():
    e = _edge(confidence_tier="EXTRACTED", confidence=1.0)
    assert e.confidence_tier == "EXTRACTED"


def test_edge_inferred_valid():
    e = _edge(confidence_tier="INFERRED", confidence=0.70)
    assert e.confidence == 0.70


def test_edge_tier_mismatch_raises():
    with pytest.raises(ValidationError, match="confidence_tier=INFERRED"):
        _edge(confidence_tier="INFERRED", confidence=0.10)


# ── Iron law (D6): ConvParser edges must be INFERRED ─────────────────────────


def test_iron_law_conv_evidence_rejects_extracted():
    with pytest.raises(ValidationError, match="ConvParser"):
        _edge(
            confidence_tier="EXTRACTED",
            confidence=1.0,
            evidence=["conv::session/1", "ConvParser"],
        )


def test_iron_law_conv_evidence_allows_inferred():
    e = _edge(
        confidence_tier="INFERRED",
        confidence=0.80,
        evidence=["conv::session/1"],
    )
    assert e.confidence_tier == "INFERRED"


def test_iron_law_non_conv_extracted_ok():
    e = _edge(
        confidence_tier="EXTRACTED",
        confidence=1.0,
        evidence=["ast-pass"],
    )
    assert e.confidence_tier == "EXTRACTED"


# ── ParseResult.from_tree ─────────────────────────────────────────────────────


def test_from_tree_nimbus_extracted():
    tree = _simple_tree(namespace="nimbus")
    result = ParseResult.from_tree(tree, parser_id="TestParser")
    assert result.parser_id == "TestParser"
    assert result.namespace == "nimbus"
    assert len(result.nodes) == 1
    assert result.nodes[0].confidence_tier == "EXTRACTED"
    assert result.nodes[0].confidence == 1.0


def test_from_tree_conv_inferred():
    tree = _simple_tree(namespace="conv")
    result = ParseResult.from_tree(tree, parser_id="ConvParser")
    assert result.nodes[0].confidence_tier == "INFERRED"
    assert result.nodes[0].confidence == 0.80


def test_from_tree_parent_child_edge():
    root = TreeNode(
        id="nimbus::parent",
        namespace="nimbus",
        kind="note",
        label="Parent",
        content="parent",
        source_file="p.md",
    )
    child = TreeNode(
        id="nimbus::child",
        namespace="nimbus",
        kind="section",
        label="Child",
        content="child",
        source_file="p.md",
    )
    root.add_child(child)
    tree = ParseTree(root=root, namespace="nimbus", source_file="p.md")
    result = ParseResult.from_tree(tree, parser_id="TestParser")

    assert len(result.nodes) == 2
    assert len(result.edges) == 1
    e = result.edges[0]
    assert e.kind == "contains"
    assert e.source_id == "nimbus::parent"
    assert e.target_id == "nimbus::child"
    assert e.confidence_tier == "EXTRACTED"


def test_from_tree_extra_edges():
    tree = _simple_tree()
    extra = [_edge(src="nimbus::root", tgt="nimbus::x", kind="wikilink")]
    result = ParseResult.from_tree(tree, parser_id="TestParser", extra_edges=extra)
    kinds = {e.kind for e in result.edges}
    assert "wikilink" in kinds


def test_parse_result_timestamp_set():
    tree = _simple_tree()
    result = ParseResult.from_tree(tree, parser_id="TestParser")
    assert result.timestamp is not None
