"""Tests for entity-based L1 grouping in multi-granularity knot retrieval.

All LLM calls are mocked — no Ollama required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from prism_rag.ingest.splitters.benchmark.multi_granularity import (
    atomize_with_entities,
    build_l1_groups_by_entity,
    build_multi_granularity_index_entity,
    MultiGranularityIndex,
)


# ---------------------------------------------------------------------------
# Tests: Entity extraction parsing
# ---------------------------------------------------------------------------

class TestAtomizeWithEntities:

    def test_basic_extraction(self):
        """Mock LLM returns JSON with body + entities; verify parsing."""
        llm_response = json.dumps([
            {"body": "Redis uses RDB snapshots for persistence.",
             "entities": ["Redis", "RDB snapshots"]},
            {"body": "AOF logs every write operation.",
             "entities": ["AOF"]},
        ])

        def mock_llm(prompt):
            return llm_response

        texts = ["Redis persistence uses RDB and AOF."]
        knot_texts, knot_entities, source_indices = atomize_with_entities(texts, mock_llm)

        assert len(knot_texts) == 2
        assert knot_texts[0] == "Redis uses RDB snapshots for persistence."
        assert knot_texts[1] == "AOF logs every write operation."
        # Entities normalized to lowercase
        assert knot_entities[0] == ["redis", "rdb snapshots"]
        assert knot_entities[1] == ["aof"]
        assert source_indices == [0, 0]

    def test_missing_entities_field(self):
        """If entities field is missing, treat as empty list."""
        llm_response = json.dumps([
            {"body": "Some fact without entities."},
        ])

        def mock_llm(prompt):
            return llm_response

        knot_texts, knot_entities, source_indices = atomize_with_entities(
            ["test text"], mock_llm
        )

        assert len(knot_texts) == 1
        assert knot_entities[0] == []

    def test_entities_not_a_list(self):
        """If entities is not a list, treat as empty."""
        llm_response = json.dumps([
            {"body": "A fact.", "entities": "not a list"},
        ])

        def mock_llm(prompt):
            return llm_response

        _, knot_entities, _ = atomize_with_entities(["text"], mock_llm)
        assert knot_entities[0] == []

    def test_empty_body_skipped(self):
        """Items with empty body are skipped."""
        llm_response = json.dumps([
            {"body": "", "entities": ["X"]},
            {"body": "Real fact.", "entities": ["Y"]},
        ])

        def mock_llm(prompt):
            return llm_response

        knot_texts, _, _ = atomize_with_entities(["text"], mock_llm)
        assert len(knot_texts) == 1
        assert knot_texts[0] == "Real fact."

    def test_llm_failure_graceful(self):
        """If LLM raises, no knots produced for that text."""
        call_count = 0

        def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("LLM down")
            return json.dumps([{"body": "Second text fact.", "entities": ["X"]}])

        knot_texts, _, source_indices = atomize_with_entities(
            ["first text", "second text"], mock_llm
        )

        assert len(knot_texts) == 1
        assert knot_texts[0] == "Second text fact."
        assert source_indices == [1]

    def test_multiple_sources(self):
        """Knots from multiple source texts get correct source_indices."""
        def mock_llm(prompt):
            if "first" in prompt:
                return json.dumps([
                    {"body": "Fact A.", "entities": ["A"]},
                    {"body": "Fact B.", "entities": ["B"]},
                ])
            else:
                return json.dumps([
                    {"body": "Fact C.", "entities": ["C"]},
                ])

        knot_texts, _, source_indices = atomize_with_entities(
            ["first text", "second text"], mock_llm
        )

        assert len(knot_texts) == 3
        assert source_indices == [0, 0, 1]

    def test_thinking_tags_stripped(self):
        """LLM output with <think>...</think> tags is cleaned."""
        llm_response = '<think>reasoning here</think>' + json.dumps([
            {"body": "A fact.", "entities": ["X"]},
        ])

        def mock_llm(prompt):
            return llm_response

        knot_texts, _, _ = atomize_with_entities(["text"], mock_llm)
        assert len(knot_texts) == 1


# ---------------------------------------------------------------------------
# Tests: Entity-based L1 grouping
# ---------------------------------------------------------------------------

class TestBuildL1GroupsByEntity:

    def test_shared_entity_groups_together(self):
        """4 knots: [A,B], [B,C], [D], [D,E] -> groups [0,1], [2,3]."""
        l0_texts = ["fact0", "fact1", "fact2", "fact3"]
        l0_entities = [
            ["a", "b"],
            ["b", "c"],
            ["d"],
            ["d", "e"],
        ]
        source_indices = [0, 0, 0, 0]

        groups = build_l1_groups_by_entity(l0_texts, l0_entities, source_indices)

        assert len(groups) == 2
        # Group 0: knots 0,1 (share entity "b")
        assert groups[0][1] == [0, 1]
        assert groups[0][2] == 0
        # Group 1: knots 2,3 (share entity "d")
        assert groups[1][1] == [2, 3]
        assert groups[1][2] == 0

    def test_source_boundary_breaks_group(self):
        """Source boundary breaks group even with shared entities."""
        l0_texts = ["fact0", "fact1", "fact2", "fact3"]
        l0_entities = [
            ["redis"],
            ["redis"],
            ["redis"],  # same entity but different source
            ["redis"],
        ]
        source_indices = [0, 0, 1, 1]

        groups = build_l1_groups_by_entity(l0_texts, l0_entities, source_indices)

        assert len(groups) == 2
        assert groups[0][1] == [0, 1]
        assert groups[0][2] == 0
        assert groups[1][1] == [2, 3]
        assert groups[1][2] == 1

    def test_max_group_size_enforced(self):
        """Group stops at max_group_size even with shared entities."""
        l0_texts = [f"fact{i}" for i in range(6)]
        l0_entities = [["common"]] * 6
        source_indices = [0] * 6

        groups = build_l1_groups_by_entity(
            l0_texts, l0_entities, source_indices, max_group_size=3,
        )

        # Should produce 2 groups of 3
        assert len(groups) == 2
        assert groups[0][1] == [0, 1, 2]
        assert groups[1][1] == [3, 4, 5]

    def test_singleton_merges_with_previous(self):
        """A singleton group merges with previous group (same source)."""
        l0_texts = ["fact0", "fact1", "fact2"]
        l0_entities = [
            ["a", "b"],
            ["b"],      # shares entity with fact0 -> group with fact0
            ["c"],      # no shared entity -> singleton -> merge with previous
        ]
        source_indices = [0, 0, 0]

        groups = build_l1_groups_by_entity(l0_texts, l0_entities, source_indices)

        # fact0+fact1 grouped (shared "b"), then fact2 singleton merges back
        assert len(groups) == 1
        assert groups[0][1] == [0, 1, 2]

    def test_singleton_merges_with_next_if_no_previous(self):
        """Singleton at start merges with next group (same source)."""
        l0_texts = ["fact0", "fact1", "fact2"]
        l0_entities = [
            ["x"],      # unique entity -> singleton
            ["a", "b"],
            ["b", "c"], # shares entity with fact1
        ]
        source_indices = [0, 0, 0]

        groups = build_l1_groups_by_entity(l0_texts, l0_entities, source_indices)

        # fact0 is singleton, merges forward into fact1+fact2 group
        assert len(groups) == 1
        assert sorted(groups[0][1]) == [0, 1, 2]

    def test_no_shared_entities_all_singletons(self):
        """No shared entities: each becomes singleton, merges with neighbors."""
        l0_texts = ["fact0", "fact1", "fact2"]
        l0_entities = [["a"], ["b"], ["c"]]
        source_indices = [0, 0, 0]

        groups = build_l1_groups_by_entity(
            l0_texts, l0_entities, source_indices, max_group_size=5,
        )

        # All singletons merge together (same source, under max)
        assert len(groups) == 1
        assert groups[0][1] == [0, 1, 2]

    def test_empty_entities_no_overlap(self):
        """Empty entity lists produce no overlap -> singletons."""
        l0_texts = ["fact0", "fact1"]
        l0_entities = [[], []]
        source_indices = [0, 0]

        groups = build_l1_groups_by_entity(l0_texts, l0_entities, source_indices)

        # Both are singletons; first merges into second (or vice versa)
        assert len(groups) == 1
        assert sorted(groups[0][1]) == [0, 1]

    def test_empty_input(self):
        groups = build_l1_groups_by_entity([], [], [])
        assert groups == []

    def test_single_knot(self):
        groups = build_l1_groups_by_entity(
            ["only"], [["entity"]], [0],
        )
        assert len(groups) == 1
        assert groups[0] == ("only", [0], 0)

    def test_l1_text_is_joined(self):
        """L1 text is space-joined from member L0 texts."""
        l0_texts = ["Alpha is great.", "Beta is fast."]
        l0_entities = [["alpha", "beta"], ["beta"]]
        source_indices = [0, 0]

        groups = build_l1_groups_by_entity(l0_texts, l0_entities, source_indices)

        assert len(groups) == 1
        assert groups[0][0] == "Alpha is great. Beta is fast."

    def test_max_group_size_prevents_singleton_merge(self):
        """Singleton cannot merge with previous if previous is at max size."""
        l0_texts = ["f0", "f1", "f2", "f3"]
        l0_entities = [["a"], ["a"], ["b"], ["c"]]
        source_indices = [0, 0, 0, 0]

        groups = build_l1_groups_by_entity(
            l0_texts, l0_entities, source_indices, max_group_size=2,
        )

        # f0+f1 grouped (shared "a", size 2=max), f2 singleton, f3 singleton
        # f2 can't merge with f0+f1 (at max), tries next (f3)
        # f3 can't merge with f0+f1 (at max), merges with f2
        assert len(groups) == 2
        assert groups[0][1] == [0, 1]
        assert sorted(groups[1][1]) == [2, 3]


# ---------------------------------------------------------------------------
# Tests: End-to-end entity index builder
# ---------------------------------------------------------------------------

class TestBuildEntityIndex:

    def test_end_to_end_with_mocks(self):
        """Full entity-based pipeline with mock LLM and embedder."""
        texts = [
            "Redis uses RDB snapshots. Redis uses AOF logs.",
            "PostgreSQL uses MVCC. VACUUM cleans dead tuples.",
        ]

        # Mock LLM: returns entity-enriched knots
        def mock_llm(prompt):
            if "Redis" in prompt:
                return json.dumps([
                    {"body": "Redis uses RDB snapshots for persistence.",
                     "entities": ["Redis", "RDB"]},
                    {"body": "Redis uses AOF to log writes.",
                     "entities": ["Redis", "AOF"]},
                ])
            elif "PostgreSQL" in prompt:
                return json.dumps([
                    {"body": "PostgreSQL uses MVCC for concurrency.",
                     "entities": ["PostgreSQL", "MVCC"]},
                    {"body": "VACUUM reclaims dead tuples in PostgreSQL.",
                     "entities": ["VACUUM", "PostgreSQL"]},
                ])
            else:
                # Tag generation
                return "Database Systems"

        # Mock embedder
        dim = 8
        counter = {"n": 0}

        def mock_embed(batch):
            vecs = []
            for _ in batch:
                v = [0.0] * dim
                v[counter["n"] % dim] = 1.0
                counter["n"] += 1
                vecs.append(v)
            return vecs

        index = build_multi_granularity_index_entity(
            texts, mock_embed, mock_llm, max_group_size=5,
        )

        # L0: 4 knots total
        assert len(index.l0_texts) == 4
        assert index.l0_source_idx == [0, 0, 1, 1]

        # L1: Redis knots share "redis" -> 1 group;
        #     PostgreSQL knots share "postgresql" -> 1 group
        assert len(index.l1_texts) == 2
        assert len(index.l1_members) == 2

        # Vectors populated
        assert len(index.l0_vectors) == 4
        assert len(index.l1_vectors) == 2

    def test_empty_texts(self):
        index = build_multi_granularity_index_entity(
            [], lambda b: [], lambda p: "[]", max_group_size=5,
        )
        assert index.l0_texts == []
        assert index.l1_texts == []
