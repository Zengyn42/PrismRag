"""Tests for HotpotQA loader and PPR retrieval.

All LLM and embedding calls are mocked -- no Ollama or HuggingFace required.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import pytest

from prism_rag.ingest.splitters.benchmark.hotpotqa import (
    HotpotQACase,
    load_hotpotqa,
)
from prism_rag.ingest.splitters.benchmark.ppr_retrieval import (
    AtomEntityGraph,
    build_atom_entity_graph,
    extract_query_entities,
    retrieve_ppr,
    retrieve_ppr_l1,
    _cosine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(dim: int, idx: int) -> list[float]:
    """Unit vector with 1.0 at position idx."""
    v = [0.0] * dim
    v[idx % dim] = 1.0
    return v


def _make_vec(*values: float) -> list[float]:
    return list(values)


# ---------------------------------------------------------------------------
# HotpotQA loader tests
# ---------------------------------------------------------------------------


class TestHotpotQALoader:

    def _make_mock_dataset(self):
        """Create a mock HF dataset with 2 examples."""
        return [
            {
                "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
                "answer": "yes",
                "type": "comparison",
                "level": "hard",
                "supporting_facts": {
                    "title": ["Scott Derrickson", "Ed Wood"],
                    "sent_id": [0, 0],
                },
                "context": {
                    "title": [
                        "Scott Derrickson", "Ed Wood",
                        "Fake1", "Fake2", "Fake3",
                        "Fake4", "Fake5", "Fake6", "Fake7", "Fake8",
                    ],
                    "sentences": [
                        ["Scott Derrickson is an American director.", " He is known for horror films."],
                        ["Ed Wood was an American filmmaker.", " He directed Plan 9."],
                        ["Fake paragraph one."],
                        ["Fake paragraph two."],
                        ["Fake paragraph three."],
                        ["Fake paragraph four."],
                        ["Fake paragraph five."],
                        ["Fake paragraph six."],
                        ["Fake paragraph seven."],
                        ["Fake paragraph eight."],
                    ],
                },
            },
            {
                "question": "What government position was held by the woman who portrayed Nora Batty?",
                "answer": "Member of Parliament",
                "type": "bridge",
                "level": "medium",
                "supporting_facts": {
                    "title": ["Nora Batty", "Kathy Staff"],
                    "sent_id": [0, 1],
                },
                "context": {
                    "title": [
                        "Nora Batty", "Kathy Staff",
                        "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
                    ],
                    "sentences": [
                        ["Nora Batty is a character from Last of the Summer Wine.", " She was portrayed by Kathy Staff."],
                        ["Kathy Staff was a British actress.", " She served as Member of Parliament."],
                        ["Distractor 1."],
                        ["Distractor 2."],
                        ["Distractor 3."],
                        ["Distractor 4."],
                        ["Distractor 5."],
                        ["Distractor 6."],
                        ["Distractor 7."],
                        ["Distractor 8."],
                    ],
                },
            },
        ]

    @patch("datasets.load_dataset")
    def test_basic_loading(self, mock_load_dataset):
        """Verify parsing of context, supporting facts, types."""
        mock_ds = self._make_mock_dataset()
        mock_load_dataset.return_value = mock_ds

        cases = load_hotpotqa(split="validation", max_cases=2)

        assert len(cases) == 2

        # Case 1: comparison
        c1 = cases[0]
        assert c1.question_type == "comparison"
        assert c1.level == "hard"
        assert c1.answer == "yes"
        assert len(c1.context_titles) == 10
        assert len(c1.context_paragraphs) == 10
        # Paragraphs are joined sentences
        assert "Scott Derrickson is an American director." in c1.context_paragraphs[0]
        assert "He is known for horror films." in c1.context_paragraphs[0]
        # Supporting titles (set, so check membership)
        assert "Scott Derrickson" in c1.supporting_titles
        assert "Ed Wood" in c1.supporting_titles

    @patch("datasets.load_dataset")
    def test_max_cases_limit(self, mock_load_dataset):
        """max_cases=1 returns only 1 case."""
        mock_ds = self._make_mock_dataset()
        mock_load_dataset.return_value = mock_ds

        cases = load_hotpotqa(max_cases=1)
        assert len(cases) == 1

    @patch("datasets.load_dataset")
    def test_paragraph_joining(self, mock_load_dataset):
        """Sentences within a paragraph are joined with spaces."""
        mock_ds = self._make_mock_dataset()
        mock_load_dataset.return_value = mock_ds

        cases = load_hotpotqa(max_cases=1)
        # First paragraph: two sentences joined
        para = cases[0].context_paragraphs[0]
        assert "Scott Derrickson is an American director." in para
        assert "He is known for horror films." in para

    @patch("datasets.load_dataset")
    def test_bridge_type(self, mock_load_dataset):
        """Bridge-type question is correctly loaded."""
        mock_ds = self._make_mock_dataset()
        mock_load_dataset.return_value = mock_ds

        cases = load_hotpotqa(max_cases=2)
        c2 = cases[1]
        assert c2.question_type == "bridge"
        assert c2.level == "medium"
        assert c2.answer == "Member of Parliament"


# ---------------------------------------------------------------------------
# AtomEntityGraph construction tests
# ---------------------------------------------------------------------------


class TestAtomEntityGraph:

    def test_basic_construction(self):
        """4 atoms, 3 entities, verify nodes and containment edges."""
        texts = [
            "Redis supports RDB snapshots.",
            "Redis AOF logs every write.",
            "PostgreSQL uses MVCC.",
            "PostgreSQL VACUUM reclaims dead tuples.",
        ]
        entities = [
            ["redis", "rdb"],
            ["redis", "aof"],
            ["postgresql", "mvcc"],
            ["postgresql", "vacuum"],
        ]
        vectors = [
            _unit_vec(4, 0),
            _unit_vec(4, 1),
            _unit_vec(4, 2),
            _unit_vec(4, 3),
        ]

        aeg = build_atom_entity_graph(texts, entities, vectors)

        assert len(aeg.atom_nodes) == 4
        # Entities: redis, rdb, aof, postgresql, mvcc, vacuum = 6
        assert len(aeg.entity_nodes) == 6
        assert aeg.atom_texts["a_0"] == "Redis supports RDB snapshots."
        assert aeg.atom_texts["a_3"] == "PostgreSQL VACUUM reclaims dead tuples."

        # Check containment edges
        G = aeg.graph
        assert G.has_edge("a_0", "e_redis")
        assert G.has_edge("a_0", "e_rdb")
        assert G.has_edge("a_1", "e_redis")
        assert G.has_edge("a_1", "e_aof")
        assert G.has_edge("a_2", "e_postgresql")
        assert G.has_edge("a_3", "e_postgresql")

        # No edge between atoms directly
        assert not G.has_edge("a_0", "a_1")

    def test_entity_to_atoms_mapping(self):
        """entity_to_atoms correctly maps shared entities."""
        texts = ["A uses X.", "B uses X.", "C uses Y."]
        entities = [["x"], ["x"], ["y"]]
        vectors = [_unit_vec(3, i) for i in range(3)]

        aeg = build_atom_entity_graph(texts, entities, vectors)

        assert sorted(aeg.entity_to_atoms["x"]) == ["a_0", "a_1"]
        assert aeg.entity_to_atoms["y"] == ["a_2"]

    def test_synonym_edges(self):
        """Synonym edges added when embedder detects similar entities."""
        texts = ["A uses Redis.", "B uses redis-server."]
        entities = [["redis"], ["redis-server"]]
        vectors = [_unit_vec(4, 0), _unit_vec(4, 1)]

        # Embedder returns very similar vectors for both entities
        def mock_embedder(texts_list):
            return [_make_vec(1.0, 0.0, 0.0, 0.0)] * len(texts_list)

        aeg = build_atom_entity_graph(
            texts, entities, vectors,
            synonym_threshold=0.8,
            embedder_fn=mock_embedder,
        )

        # Should have synonym edge between redis and redis-server
        assert aeg.graph.has_edge("e_redis", "e_redis-server")
        edge_data = aeg.graph.edges["e_redis", "e_redis-server"]
        assert edge_data["edge_type"] == "synonym"

    def test_empty_entities(self):
        """Atoms with no entities still appear in graph."""
        texts = ["Some fact."]
        entities = [[]]
        vectors = [_unit_vec(4, 0)]

        aeg = build_atom_entity_graph(texts, entities, vectors)
        assert len(aeg.atom_nodes) == 1
        assert len(aeg.entity_nodes) == 0
        assert aeg.graph.number_of_edges() == 0


# ---------------------------------------------------------------------------
# PPR seed setup tests
# ---------------------------------------------------------------------------


class TestPPRSeedSetup:

    def _build_simple_aeg(self):
        """Build a simple AEG with 4 atoms and 3 entities."""
        texts = [
            "Redis uses RDB.",
            "Redis uses AOF.",
            "PostgreSQL uses MVCC.",
            "Kubernetes uses etcd.",
        ]
        entities = [
            ["redis", "rdb"],
            ["redis", "aof"],
            ["postgresql", "mvcc"],
            ["kubernetes", "etcd"],
        ]
        vectors = [
            _make_vec(1, 0, 0, 0),
            _make_vec(0.9, 0.1, 0, 0),
            _make_vec(0, 0, 1, 0),
            _make_vec(0, 0, 0, 1),
        ]
        return build_atom_entity_graph(texts, entities, vectors), vectors

    def test_entity_seeds_match(self):
        """Query entities matching graph entities seed the PPR walk."""
        aeg, vectors = self._build_simple_aeg()

        # Query about Redis -- should prioritize Redis atoms
        results = retrieve_ppr(
            query="How does Redis persist data?",
            query_entities=["redis"],
            query_vector=_make_vec(1, 0, 0, 0),
            aeg=aeg,
            top_k=4,
        )

        assert len(results) == 4
        # Redis atoms should rank higher due to entity seed
        assert "Redis uses RDB." in results[0] or "Redis uses AOF." in results[0]

    def test_no_matching_entities_uses_cosine_fallback(self):
        """When no query entities match, falls back to embedding cosine seeds."""
        aeg, vectors = self._build_simple_aeg()

        results = retrieve_ppr(
            query="Something about databases",
            query_entities=["nonexistent_entity"],
            query_vector=_make_vec(0, 0, 1, 0),  # closest to postgresql
            aeg=aeg,
            top_k=2,
        )

        assert len(results) == 2
        # Should still find postgresql-related atom via cosine seed
        assert "PostgreSQL uses MVCC." in results[0]

    def test_empty_graph_returns_empty(self):
        """Empty graph returns empty results."""
        aeg = build_atom_entity_graph([], [], [])
        results = retrieve_ppr("test", [], _make_vec(1, 0), aeg)
        assert results == []


# ---------------------------------------------------------------------------
# PPR retrieval ranking tests
# ---------------------------------------------------------------------------


class TestPPRRetrieval:

    def test_top_k_limit(self):
        """retrieve_ppr respects top_k limit."""
        texts = [f"Fact {i}." for i in range(10)]
        entities = [["common_entity"] for _ in range(10)]
        vectors = [_unit_vec(10, i) for i in range(10)]

        aeg = build_atom_entity_graph(texts, entities, vectors)

        results = retrieve_ppr(
            "test", ["common_entity"], _unit_vec(10, 0), aeg,
            top_k=3,
        )
        assert len(results) == 3

    def test_multi_hop_connectivity(self):
        """PPR should discover atoms connected through shared entities.

        Graph: a0 -redis- e_redis -redis- a1
               a1 -persistence- e_persistence -persistence- a2

        Query seeds on 'redis' entity. PPR should reach a2 through
        the entity chain redis -> a1 -> persistence -> a2.
        """
        texts = [
            "Redis is an in-memory database.",
            "Redis persistence uses RDB and AOF.",
            "RDB persistence creates snapshots at intervals.",
        ]
        entities = [
            ["redis"],
            ["redis", "persistence"],
            ["persistence", "rdb"],
        ]
        # Orthogonal vectors so cosine alone can't connect them
        vectors = [
            _make_vec(1, 0, 0),
            _make_vec(0, 1, 0),
            _make_vec(0, 0, 1),
        ]

        aeg = build_atom_entity_graph(texts, entities, vectors)

        # Query about Redis -- PPR should propagate to a2 via entity chain
        results = retrieve_ppr(
            "How does Redis work?",
            ["redis"],
            _make_vec(1, 0, 0),  # only cosine-similar to a0
            aeg,
            top_k=3,
        )

        assert len(results) == 3
        # All three should be reachable through graph traversal
        assert "Redis is an in-memory database." in results
        assert "Redis persistence uses RDB and AOF." in results
        assert "RDB persistence creates snapshots at intervals." in results


# ---------------------------------------------------------------------------
# PPR -> L1 retrieval tests
# ---------------------------------------------------------------------------


class TestPPRL1Retrieval:

    def test_ppr_l1_deduplication(self):
        """PPR L1 returns deduplicated L1 groups."""
        texts = ["A.", "B.", "C.", "D."]
        entities = [["x"], ["x"], ["y"], ["y"]]
        vectors = [_unit_vec(4, i) for i in range(4)]

        aeg = build_atom_entity_graph(texts, entities, vectors)

        l1_members = [[0, 1], [2, 3]]
        l1_texts = ["A. B.", "C. D."]

        results = retrieve_ppr_l1(
            "test", ["x"], _unit_vec(4, 0), aeg,
            l1_members, l1_texts,
            top_k=5,
        )

        # Should return at most 2 L1 groups (we only have 2)
        assert len(results) <= 2
        # No duplicates
        assert len(results) == len(set(results))


# ---------------------------------------------------------------------------
# Entity extraction tests
# ---------------------------------------------------------------------------


class TestEntityExtraction:

    def test_basic_extraction(self):
        """LLM returns JSON array of entities."""
        def mock_llm(prompt):
            return '["Redis", "PostgreSQL", "MVCC"]'

        entities = extract_query_entities("How does Redis compare to PostgreSQL?", mock_llm)

        assert entities == ["redis", "postgresql", "mvcc"]

    def test_extraction_with_thinking_tags(self):
        """Thinking tags are stripped before parsing."""
        def mock_llm(prompt):
            return '<think>Let me analyze...</think>["Redis", "AOF"]'

        entities = extract_query_entities("How does Redis AOF work?", mock_llm)
        assert entities == ["redis", "aof"]

    def test_extraction_fallback_on_invalid_json(self):
        """Falls back to capitalized words on JSON parse failure."""
        def mock_llm(prompt):
            return "I cannot parse this properly"

        entities = extract_query_entities(
            "How does Redis handle Persistence?", mock_llm,
        )
        # Fallback: capitalized words
        assert "redis" in entities
        assert "persistence" in entities

    def test_extraction_empty_result(self):
        """Empty LLM response falls back gracefully."""
        def mock_llm(prompt):
            return "[]"

        entities = extract_query_entities("what?", mock_llm)
        assert entities == []
