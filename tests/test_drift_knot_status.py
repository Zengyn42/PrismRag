"""Tests for drift-detection KNOT status flagging.

Validates that flag_knots_suspected correctly marks knowledge nodes
as "suspected" and leaves unrelated nodes untouched.
"""
from __future__ import annotations

import pytest

from prism_rag.store.graph import KnowledgeGraph, Node
from prism_rag.ingest.drift import flag_knots_suspected


def _make_graph_with_knots() -> KnowledgeGraph:
    """Build a small graph with two docs and several knowledge nodes."""
    g = KnowledgeGraph()

    # Doc nodes
    g.add_node(Node(id="doc-a", label="Doc A", kind="note", source_file="notes/doc-a.md"))
    g.add_node(Node(id="doc-b", label="Doc B", kind="note", source_file="notes/doc-b.md"))

    # Knowledge nodes linked to doc-a (via source_file)
    g.add_node(Node(
        id="KNOW-000001",
        label="Knot 1",
        kind="knowledge",
        source_file="notes/doc-a.md",
        content="Some fact from doc A",
    ))
    g.add_node(Node(
        id="KNOW-000002",
        label="Knot 2",
        kind="knowledge",
        source_file="notes/doc-a.md",
        content="Another fact from doc A",
    ))

    # Knowledge node linked to doc-b (via source_file)
    g.add_node(Node(
        id="KNOW-000003",
        label="Knot 3",
        kind="knowledge",
        source_file="notes/doc-b.md",
        content="Fact from doc B",
    ))

    # Knowledge node linked via frontmatter.atomized_from
    g.add_node(Node(
        id="KNOW-000004",
        label="Knot 4",
        kind="knowledge",
        source_file="knowledge/KNOW-000004.md",
        content="Fact atomized from doc A",
        frontmatter={"atomized_from": "notes/doc-a.md"},
    ))

    return g


class TestFlagKnotsSuspected:
    """Core unit tests for flag_knots_suspected."""

    def test_flags_matching_source_file(self):
        """Knowledge nodes whose source_file matches get flagged."""
        graph = _make_graph_with_knots()

        flagged = flag_knots_suspected(graph, "notes/doc-a.md", reason="stale ref")

        assert "KNOW-000001" in flagged
        assert "KNOW-000002" in flagged
        assert graph.g.nodes["KNOW-000001"]["status"] == "suspected"
        assert graph.g.nodes["KNOW-000002"]["status"] == "suspected"

    def test_flags_matching_atomized_from(self):
        """Knowledge nodes whose frontmatter.atomized_from matches also get flagged."""
        graph = _make_graph_with_knots()

        flagged = flag_knots_suspected(graph, "notes/doc-a.md", reason="stale ref")

        assert "KNOW-000004" in flagged
        assert graph.g.nodes["KNOW-000004"]["status"] == "suspected"

    def test_does_not_flag_unrelated_nodes(self):
        """Nodes linked to a different doc are not affected."""
        graph = _make_graph_with_knots()

        flag_knots_suspected(graph, "notes/doc-a.md", reason="stale ref")

        # doc-b's knot should remain confirmed (default)
        assert graph.g.nodes["KNOW-000003"].get("status", "confirmed") != "suspected"

    def test_does_not_flag_non_knowledge_nodes(self):
        """Doc/note nodes are never flagged even if source_file matches."""
        graph = _make_graph_with_knots()

        flagged = flag_knots_suspected(graph, "notes/doc-a.md", reason="test")

        assert "doc-a" not in flagged

    def test_idempotent_double_flag(self):
        """Flagging already-suspected nodes does not crash or re-flag."""
        graph = _make_graph_with_knots()

        first = flag_knots_suspected(graph, "notes/doc-a.md", reason="first pass")
        second = flag_knots_suspected(graph, "notes/doc-a.md", reason="second pass")

        # Second call should return empty (nothing newly flagged)
        assert len(second) == 0
        # Nodes should still be suspected
        assert graph.g.nodes["KNOW-000001"]["status"] == "suspected"
        assert graph.g.nodes["KNOW-000002"]["status"] == "suspected"
        # First call should have flagged them
        assert len(first) > 0

    def test_reason_stored_in_metadata(self):
        """The reason string is stored in node metadata."""
        graph = _make_graph_with_knots()

        flag_knots_suspected(graph, "notes/doc-a.md", reason="symbol X deleted")

        meta = graph.g.nodes["KNOW-000001"].get("metadata", {})
        assert meta.get("drift_reason") == "symbol X deleted"

    def test_no_reason_omits_metadata_key(self):
        """When reason is empty, drift_reason is not added to metadata."""
        graph = _make_graph_with_knots()

        flag_knots_suspected(graph, "notes/doc-a.md")

        meta = graph.g.nodes["KNOW-000001"].get("metadata", {})
        assert "drift_reason" not in meta

    def test_no_matching_nodes_returns_empty(self):
        """Flagging a doc with no knowledge nodes returns empty list."""
        graph = _make_graph_with_knots()

        flagged = flag_knots_suspected(graph, "notes/nonexistent.md", reason="test")

        assert flagged == []

    def test_empty_graph(self):
        """Works on an empty graph without errors."""
        graph = KnowledgeGraph()

        flagged = flag_knots_suspected(graph, "any-file.md", reason="test")

        assert flagged == []
