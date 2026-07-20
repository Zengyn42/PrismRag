"""Tests for multi-granularity knot retrieval.

All LLM and embedding calls are mocked — no Ollama required.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from prism_rag.ingest.splitters.base import Knot, Splitter
from prism_rag.ingest.splitters.benchmark.multi_granularity import (
    MultiGranularityIndex,
    build_l1_groups,
    build_multi_granularity_index,
    retrieve_collapsed,
    retrieve_flat_l0,
    retrieve_flat_l1,
    retrieve_multi_layer,
    retrieve_parent_l0,
    _extract_keywords,
    _cosine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int, idx: int) -> list[float]:
    """Create a unit vector with 1.0 at position idx, 0.0 elsewhere."""
    v = [0.0] * dim
    v[idx % dim] = 1.0
    return v


def _make_vec(*values: float) -> list[float]:
    """Create a simple vector from values."""
    return list(values)


class MockSplitter(Splitter):
    """Returns one Knot per sentence (split on period)."""

    @property
    def name(self) -> str:
        return "mock"

    def split(self, section_text: str, *, doc_context: str | None = None) -> list[Knot]:
        sentences = [s.strip() for s in section_text.split(".") if s.strip()]
        return [Knot(text=s, method="mock") for s in sentences]


# ---------------------------------------------------------------------------
# Tests: L1 Grouping
# ---------------------------------------------------------------------------

class TestL1Grouping:

    def test_basic_grouping_two_sources_window3(self):
        """6 knots from 2 sources, window=3 -> 3 groups (2 from src0, 1 from src1)."""
        l0_texts = ["A", "B", "C", "D", "E", "F"]
        l0_source_idx = [0, 0, 0, 0, 1, 1]

        groups = build_l1_groups(l0_texts, l0_source_idx, window=3)

        assert len(groups) == 3

        # Group 0: src 0, members [0,1,2]
        text0, members0, src0 = groups[0]
        assert src0 == 0
        assert members0 == [0, 1, 2]
        assert text0 == "A B C"

        # Group 1: src 0, members [3]
        text1, members1, src1 = groups[1]
        assert src1 == 0
        assert members1 == [3]
        assert text1 == "D"

        # Group 2: src 1, members [4,5]
        text2, members2, src2 = groups[2]
        assert src2 == 1
        assert members2 == [4, 5]
        assert text2 == "E F"

    def test_source_boundary_breaks_group(self):
        """Window=5 but source changes after 2 items."""
        l0_texts = ["A", "B", "C", "D"]
        l0_source_idx = [0, 0, 1, 1]

        groups = build_l1_groups(l0_texts, l0_source_idx, window=5)

        assert len(groups) == 2
        assert groups[0][1] == [0, 1]
        assert groups[1][1] == [2, 3]

    def test_empty_input(self):
        groups = build_l1_groups([], [], window=3)
        assert groups == []

    def test_single_knot(self):
        groups = build_l1_groups(["X"], [0], window=3)
        assert len(groups) == 1
        assert groups[0] == ("X", [0], 0)


# ---------------------------------------------------------------------------
# Tests: L2 Tag Generation
# ---------------------------------------------------------------------------

class TestL2Tags:

    def test_mock_llm_returns_tag(self):
        from prism_rag.ingest.splitters.benchmark.multi_granularity import _generate_l2_tags

        mock_llm = MagicMock(return_value="Database Persistence")
        l1_texts = ["Redis RDB snapshots", "Redis AOF logs", "PostgreSQL WAL"]
        clusters = [[0, 1, 2]]

        tags = _generate_l2_tags(clusters, l1_texts, mock_llm)

        assert len(tags) == 1
        assert tags[0] == "Database Persistence"
        assert mock_llm.call_count == 1

    def test_multiple_clusters(self):
        from prism_rag.ingest.splitters.benchmark.multi_granularity import _generate_l2_tags

        call_count = 0
        def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            return f"Tag {call_count}"

        clusters = [[0, 1], [2, 3]]
        l1_texts = ["A", "B", "C", "D"]

        tags = _generate_l2_tags(clusters, l1_texts, mock_llm)

        assert len(tags) == 2
        assert tags[0] == "Tag 1"
        assert tags[1] == "Tag 2"

    def test_llm_failure_returns_general(self):
        from prism_rag.ingest.splitters.benchmark.multi_granularity import _generate_l2_tags

        def failing_llm(prompt):
            raise RuntimeError("LLM unavailable")

        tags = _generate_l2_tags([[0]], ["text"], failing_llm)
        assert tags == ["general"]


# ---------------------------------------------------------------------------
# Tests: Retrieval functions
# ---------------------------------------------------------------------------

class TestRetrieval:
    """Test retrieval functions with known vectors."""

    @pytest.fixture
    def simple_index(self):
        """Build a small index with predictable vectors.

        L0: 6 knots, 3-dim vectors (unit vectors cycling through dims)
        L1: 3 groups of 2
        L2: 1 cluster (all L1)
        """
        dim = 3
        return MultiGranularityIndex(
            l0_texts=["k0", "k1", "k2", "k3", "k4", "k5"],
            l0_vectors=[
                _make_vec(1, 0, 0),  # k0
                _make_vec(0, 1, 0),  # k1
                _make_vec(0, 0, 1),  # k2
                _make_vec(1, 1, 0),  # k3 (close to k0 and k1)
                _make_vec(0, 1, 1),  # k4 (close to k1 and k2)
                _make_vec(1, 0, 1),  # k5 (close to k0 and k2)
            ],
            l0_source_idx=[0, 0, 0, 1, 1, 1],
            l1_texts=["k0 k1", "k2 k3", "k4 k5"],
            l1_vectors=[
                _make_vec(1, 1, 0),   # g0: close to dim 0,1
                _make_vec(1, 1, 1),   # g1: close to everything
                _make_vec(1, 0, 1),   # g2: close to dim 0,2
            ],
            l1_members=[[0, 1], [2, 3], [4, 5]],
            l1_source_idx=[0, 0, 1],
            l2_tags=["topic alpha"],
            l2_vectors=[_make_vec(1, 1, 1)],
            l2_members=[[0, 1, 2]],
            source_texts=["source0", "source1"],
            l1_to_l2=[0, 0, 0],
        )

    def test_flat_l0(self, simple_index):
        query = _make_vec(1, 0, 0)  # closest to k0
        results = retrieve_flat_l0(query, simple_index, top_k=2)
        assert len(results) == 2
        assert results[0] == "k0"  # exact match

    def test_flat_l1(self, simple_index):
        query = _make_vec(1, 1, 0)  # exact match to g0
        results = retrieve_flat_l1(query, simple_index, top_k=1)
        assert results == ["k0 k1"]

    def test_parent_l0(self, simple_index):
        query = _make_vec(1, 0, 0)  # closest L0 is k0, parent L1 is g0
        results = retrieve_parent_l0(query, simple_index, top_k=1)
        assert len(results) == 1
        assert results[0] == "k0 k1"  # parent of k0

    def test_multi_layer_keyword_match(self, simple_index):
        query_vec = _make_vec(1, 0, 0)
        # Query with "alpha" should match L2 tag "topic alpha"
        results = retrieve_multi_layer(query_vec, "what about alpha", simple_index, top_k=2)
        assert len(results) == 2
        # Should return L1 texts from the matched cluster (all of them)

    def test_multi_layer_fallback_to_vector(self, simple_index):
        query_vec = _make_vec(1, 0, 0)
        # Query with no keyword match -> fallback to L2 vector cosine
        results = retrieve_multi_layer(query_vec, "xyz unrelated", simple_index, top_k=2)
        assert len(results) == 2

    def test_collapsed(self, simple_index):
        query = _make_vec(1, 0, 0)
        results = retrieve_collapsed(query, simple_index, top_k=3)
        assert len(results) == 3
        # Should contain both L0 and L1 texts
        assert results[0] == "k0"  # exact L0 match

    def test_empty_index(self):
        empty = MultiGranularityIndex(
            l0_texts=[], l0_vectors=[], l0_source_idx=[],
            l1_texts=[], l1_vectors=[], l1_members=[], l1_source_idx=[],
            l2_tags=[], l2_vectors=[], l2_members=[],
            source_texts=[], l1_to_l2=[],
        )
        assert retrieve_flat_l0(_make_vec(1, 0, 0), empty) == []
        assert retrieve_flat_l1(_make_vec(1, 0, 0), empty) == []
        assert retrieve_parent_l0(_make_vec(1, 0, 0), empty) == []
        assert retrieve_multi_layer(_make_vec(1, 0, 0), "test", empty) == []
        assert retrieve_collapsed(_make_vec(1, 0, 0), empty) == []


# ---------------------------------------------------------------------------
# Tests: End-to-end index building with mocks
# ---------------------------------------------------------------------------

class TestBuildIndex:

    def test_end_to_end_with_mocks(self):
        """Full pipeline with mock splitter, embedder, and LLM."""
        texts = [
            "Redis uses RDB snapshots. Redis uses AOF logs.",
            "PostgreSQL uses MVCC. VACUUM cleans dead tuples.",
        ]

        # Mock splitter: splits on period
        splitter = MockSplitter()

        # Mock embedder: returns sequential unit vectors
        dim = 8
        call_counter = {"n": 0}
        def mock_embed(batch):
            vecs = []
            for _ in batch:
                v = [0.0] * dim
                v[call_counter["n"] % dim] = 1.0
                call_counter["n"] += 1
                vecs.append(v)
            return vecs

        # Mock LLM: returns a tag
        mock_llm = MagicMock(return_value="Database Systems")

        index = build_multi_granularity_index(
            texts, splitter, mock_embed, mock_llm, l1_window=3
        )

        # L0: 4 knots (2 per source, split on period)
        assert len(index.l0_texts) == 4
        assert index.l0_source_idx == [0, 0, 1, 1]

        # L1: 2 groups (one per source, window=3 fits both)
        assert len(index.l1_texts) == 2
        assert len(index.l1_members) == 2

        # L2: 1 cluster (< 10 L1 groups -> single cluster)
        assert len(index.l2_tags) == 1
        assert index.l2_tags[0] == "Database Systems"
        assert len(index.l2_members) == 1
        assert sorted(index.l2_members[0]) == [0, 1]

        # Vectors populated
        assert len(index.l0_vectors) == 4
        assert len(index.l1_vectors) == 2
        assert len(index.l2_vectors) == 1

        # l1_to_l2 mapping
        assert index.l1_to_l2 == [0, 0]

    def test_empty_texts(self):
        splitter = MockSplitter()
        index = build_multi_granularity_index(
            [], splitter, lambda b: [], MagicMock(), l1_window=3
        )
        assert index.l0_texts == []
        assert index.l1_texts == []
        assert index.l2_tags == []


# ---------------------------------------------------------------------------
# Tests: Keyword extraction
# ---------------------------------------------------------------------------

class TestKeywordExtraction:

    def test_removes_stopwords(self):
        kw = _extract_keywords("what is the capital of France")
        assert "france" in kw
        assert "capital" in kw
        assert "what" not in kw
        assert "is" not in kw
        assert "the" not in kw

    def test_empty_string(self):
        assert _extract_keywords("") == set()


# ---------------------------------------------------------------------------
# Tests: Cosine similarity
# ---------------------------------------------------------------------------

class TestCosine:

    def test_identical_vectors(self):
        assert abs(_cosine([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        assert abs(_cosine([1, 0, 0], [0, 1, 0])) < 1e-9

    def test_zero_vector(self):
        assert _cosine([0, 0, 0], [1, 2, 3]) == 0.0
