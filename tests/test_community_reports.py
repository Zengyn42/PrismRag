"""Tests for community reports and global_ask (Phase 4, v6.0).

All tests use an injected mock llm_fn — no Ollama required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from prism_rag.report.community_report import (
    CommunityReport,
    generate_all_community_reports,
    generate_community_report,
    load_community_reports,
    save_community_reports,
    _extract_json_object,
)
from prism_rag.retrieve.global_ask import global_ask
from prism_rag.store.graph import Community, Edge, KnowledgeGraph, Node


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_graph(n_communities: int = 2, members_per: int = 4) -> KnowledgeGraph:
    """Build a small test graph with communities pre-assigned."""
    kg = KnowledgeGraph()
    for c in range(n_communities):
        comm_id = f"community_{c:03d}"
        god_nodes: list[str] = []
        for m in range(members_per):
            nid = f"node_{c}_{m}"
            kg.add_node(Node(
                id=nid,
                label=f"Node {c}-{m}",
                kind="knowledge",
                content=f"Content of node {c}-{m}: some knowledge about topic {c}.",
                community_id=comm_id,
            ))
            if m < 2:
                god_nodes.append(nid)
        # Internal edges
        for m in range(members_per - 1):
            kg.add_edge(Edge(
                source=f"node_{c}_{m}",
                target=f"node_{c}_{m + 1}",
                relation="references",
            ))
        kg.communities[comm_id] = Community(
            id=comm_id,
            label=f"Topic {c}",
            god_nodes=god_nodes,
            member_count=members_per,
            internal_density=0.5,
        )
    return kg


def _mock_llm_report(prompt: str) -> str:
    """Mock LLM that returns a valid community report JSON."""
    return json.dumps({
        "title": "Mock Community Title",
        "summary": "This community covers mock topics for testing purposes.",
        "rating": 7,
        "key_findings": ["Finding A", "Finding B", "Finding C"],
        "cited_nodes": ["node_0_0", "node_0_1"],
    })


def _mock_llm_map_reduce(prompt: str) -> str:
    """Mock LLM that handles both map and reduce prompts."""
    if "NOT RELEVANT" in prompt:
        # Map prompt — return a relevant answer
        if "Topic 0" in prompt or "Mock Community" in prompt:
            return "This community has relevant info about the mock topic."
        return "NOT RELEVANT"
    # Reduce prompt
    if "Community answers" in prompt or "community answers" in prompt:
        return "Synthesized answer: the mock topic is well covered."
    # Fallback for report generation
    return _mock_llm_report(prompt)


# ── Tests: _extract_json_object ──────────────────────────────────────────────


class TestExtractJsonObject:
    def test_plain_object(self):
        text = '{"title": "hello", "rating": 5}'
        result = _extract_json_object(text)
        assert result["title"] == "hello"
        assert result["rating"] == 5

    def test_fenced_object(self):
        text = 'Some prose\n```json\n{"title": "fenced"}\n```\nmore prose'
        result = _extract_json_object(text)
        assert result["title"] == "fenced"

    def test_object_with_leading_text(self):
        text = 'Here is the result: {"title": "embedded", "rating": 3}'
        result = _extract_json_object(text)
        assert result["title"] == "embedded"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No parseable JSON object"):
            _extract_json_object("just plain text with no json at all")


# ── Tests: generate_community_report ─────────────────────────────────────────


class TestGenerateCommunityReport:
    def test_basic_report(self):
        graph = _build_graph(n_communities=1, members_per=4)
        report = generate_community_report(
            graph, "community_000", llm_fn=_mock_llm_report
        )
        assert isinstance(report, CommunityReport)
        assert report.community_id == "community_000"
        assert report.title == "Mock Community Title"
        assert report.rating == 7
        assert len(report.key_findings) == 3
        assert "node_0_0" in report.cited_nodes

    def test_missing_community_raises(self):
        graph = _build_graph(n_communities=1)
        with pytest.raises(KeyError, match="community_999"):
            generate_community_report(graph, "community_999", llm_fn=_mock_llm_report)

    def test_fallback_on_bad_llm_output(self):
        """When LLM returns garbage, we get a fallback report instead of crashing."""
        graph = _build_graph(n_communities=1, members_per=4)
        report = generate_community_report(
            graph, "community_000", llm_fn=lambda p: "not json at all"
        )
        assert isinstance(report, CommunityReport)
        assert report.community_id == "community_000"
        # Fallback title is the community label
        assert report.title == "Topic 0"
        assert report.rating == 5  # default fallback


# ── Tests: generate_all_community_reports ────────────────────────────────────


class TestGenerateAllCommunityReports:
    def test_filters_small_communities(self):
        graph = _build_graph(n_communities=2, members_per=4)
        # Add a tiny community
        kg_tiny_id = "community_tiny"
        kg_tiny_node = Node(id="tiny_0", label="Tiny", kind="knowledge", community_id=kg_tiny_id)
        graph.add_node(kg_tiny_node)
        graph.communities[kg_tiny_id] = Community(
            id=kg_tiny_id, label="Tiny", god_nodes=["tiny_0"],
            member_count=1, internal_density=0.0,
        )

        reports = generate_all_community_reports(
            graph, llm_fn=_mock_llm_report, min_members=3
        )
        # Should only get 2 reports (the tiny community is skipped)
        assert len(reports) == 2
        report_ids = {r.community_id for r in reports}
        assert kg_tiny_id not in report_ids

    def test_empty_graph_returns_empty(self):
        graph = KnowledgeGraph()
        reports = generate_all_community_reports(graph, llm_fn=_mock_llm_report)
        assert reports == []

    def test_no_communities_returns_empty(self):
        graph = KnowledgeGraph()
        graph.add_node(Node(id="n1", label="Alone", kind="knowledge"))
        reports = generate_all_community_reports(graph, llm_fn=_mock_llm_report)
        assert reports == []


# ── Tests: save / load ──────────────────────────────────────────────────────


class TestSaveLoadReports:
    def test_round_trip(self, tmp_path: Path):
        reports = [
            CommunityReport(
                community_id="c_000",
                title="Test Title",
                summary="Test summary.",
                rating=8,
                key_findings=["A", "B"],
                cited_nodes=["n1"],
            ),
        ]
        json_path = tmp_path / "community_reports.json"
        save_community_reports(reports, json_path)

        loaded = load_community_reports(json_path)
        assert len(loaded) == 1
        assert loaded[0].title == "Test Title"
        assert loaded[0].rating == 8
        assert loaded[0].key_findings == ["A", "B"]

        # Markdown file should also exist
        md_path = json_path.with_suffix(".md")
        assert md_path.exists()
        md_content = md_path.read_text()
        assert "Test Title" in md_content

    def test_load_nonexistent_returns_empty(self, tmp_path: Path):
        loaded = load_community_reports(tmp_path / "missing.json")
        assert loaded == []


# ── Tests: global_ask ────────────────────────────────────────────────────────


class TestGlobalAsk:
    def test_map_reduce_with_cached_reports(self):
        graph = _build_graph(n_communities=2, members_per=4)
        reports = [
            CommunityReport(
                community_id="community_000",
                title="Topic 0",
                summary="This community covers topic 0.",
                rating=8,
                key_findings=["Topic 0 is important"],
                cited_nodes=["node_0_0"],
            ),
            CommunityReport(
                community_id="community_001",
                title="Topic 1",
                summary="This community covers topic 1.",
                rating=5,
                key_findings=["Topic 1 is minor"],
                cited_nodes=["node_1_0"],
            ),
        ]
        answer = global_ask(
            "What is topic 0?",
            graph,
            community_reports=reports,
            llm_fn=_mock_llm_map_reduce,
        )
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_on_the_fly_report_generation(self):
        """When no cached reports provided, generates them automatically."""
        graph = _build_graph(n_communities=1, members_per=4)
        answer = global_ask(
            "What is this about?",
            graph,
            community_reports=None,
            llm_fn=_mock_llm_map_reduce,
        )
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_empty_graph_returns_message(self):
        graph = KnowledgeGraph()
        answer = global_ask(
            "anything",
            graph,
            community_reports=None,
            llm_fn=_mock_llm_map_reduce,
        )
        assert "No communities" in answer

    def test_no_relevant_communities(self):
        """When all communities return NOT RELEVANT, we get an appropriate message."""
        graph = _build_graph(n_communities=1, members_per=4)
        reports = [
            CommunityReport(
                community_id="community_000",
                title="Irrelevant Topic",
                summary="Totally unrelated stuff.",
                rating=2,
                key_findings=[],
                cited_nodes=[],
            ),
        ]

        def _always_not_relevant(prompt: str) -> str:
            return "NOT RELEVANT"

        answer = global_ask(
            "some question",
            graph,
            community_reports=reports,
            llm_fn=_always_not_relevant,
        )
        assert "No community" in answer

    def test_single_member_community_skipped_on_fly(self):
        """Communities with < min_members are skipped during on-the-fly generation."""
        graph = KnowledgeGraph()
        graph.add_node(Node(id="alone", label="Alone", kind="knowledge", community_id="c_solo"))
        graph.communities["c_solo"] = Community(
            id="c_solo", label="Solo", god_nodes=["alone"],
            member_count=1, internal_density=0.0,
        )
        answer = global_ask(
            "anything",
            graph,
            community_reports=None,
            llm_fn=_mock_llm_map_reduce,
            min_members=3,
        )
        assert "No communities" in answer


# ── Tests: CommunityReport dataclass ─────────────────────────────────────────


class TestCommunityReportDataclass:
    def test_to_dict(self):
        r = CommunityReport(
            community_id="c_000",
            title="T",
            summary="S",
            rating=5,
            key_findings=["a"],
            cited_nodes=["n1"],
        )
        d = r.to_dict()
        assert d["community_id"] == "c_000"
        assert d["rating"] == 5
        assert isinstance(d["key_findings"], list)
