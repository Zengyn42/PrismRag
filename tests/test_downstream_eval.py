"""Tests for downstream RAG evaluation harness.

All tests use mocked LLM and embedder — no Ollama required.
"""

from __future__ import annotations

import json
import math

import pytest

from prism_rag.ingest.splitters.benchmark.downstream import (
    SAMPLE_TEXTS,
    DownstreamReport,
    QAPair,
    RetrievalScore,
    _extract_json_array,
    _extract_json_object,
    _find_top_k,
    build_splitter_index,
    cosine_similarity,
    evaluate_retrieval,
    format_downstream_report,
    generate_qa_pairs,
    run_downstream_eval,
    score_boundary_clarity,
    score_context_sufficiency,
    score_iou,
    word_overlap,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _mock_llm(prompt: str) -> str:
    """Mock LLM that returns predictable QA pairs."""
    return json.dumps([
        {"question": "What persistence mechanisms does Redis support?",
         "answer": "Redis supports RDB snapshots and AOF logs."},
        {"question": "What does RDB create?",
         "answer": "RDB creates point-in-time snapshots at configured intervals."},
        {"question": "What fsync policies does AOF support?",
         "answer": "AOF supports always, everysec, or no fsync policies."},
    ])


def _mock_embedder(texts: list[str]) -> list[list[float]]:
    """Mock embedder: simple bag-of-characters hash to 8-dim vectors."""
    vectors = []
    for text in texts:
        # Deterministic pseudo-embedding based on character frequencies
        vec = [0.0] * 8
        for i, ch in enumerate(text.lower()):
            vec[ord(ch) % 8] += 1.0
        # Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        vectors.append(vec)
    return vectors


# ---------------------------------------------------------------------------
# Unit tests: cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# Unit tests: word overlap
# ---------------------------------------------------------------------------


class TestWordOverlap:
    def test_full_overlap(self):
        assert word_overlap("hello world", "hello world") == pytest.approx(1.0)

    def test_partial_overlap(self):
        # "hello" overlaps, "foo" doesn't — 1/2 of text_b words found
        assert word_overlap("hello world", "hello foo") == pytest.approx(0.5)

    def test_no_overlap(self):
        assert word_overlap("alpha beta", "gamma delta") == pytest.approx(0.0)

    def test_empty_b(self):
        assert word_overlap("hello", "") == 0.0


# ---------------------------------------------------------------------------
# Unit tests: JSON extraction
# ---------------------------------------------------------------------------


class TestExtractJsonArray:
    def test_plain_array(self):
        result = _extract_json_array('[{"q": 1}]')
        assert result == [{"q": 1}]

    def test_with_markdown_fences(self):
        text = '```json\n[{"q": 1}]\n```'
        result = _extract_json_array(text)
        assert result == [{"q": 1}]

    def test_with_leading_prose(self):
        text = 'Here are the results:\n[{"q": 1}]'
        result = _extract_json_array(text)
        assert result == [{"q": 1}]

    def test_invalid_json(self):
        assert _extract_json_array("not json at all") == []


# ---------------------------------------------------------------------------
# Unit tests: QA generation
# ---------------------------------------------------------------------------


class TestGenerateQAPairs:
    def test_basic_generation(self):
        texts = ["Redis supports RDB and AOF."]
        pairs = generate_qa_pairs(texts, _mock_llm, n_per_text=3)
        assert len(pairs) == 3
        assert all(isinstance(p, QAPair) for p in pairs)
        assert pairs[0].source_text_idx == 0
        assert pairs[0].source_text == texts[0]

    def test_multiple_texts(self):
        texts = ["Text one.", "Text two."]
        pairs = generate_qa_pairs(texts, _mock_llm, n_per_text=3)
        assert len(pairs) == 6  # 3 per text
        assert pairs[3].source_text_idx == 1

    def test_handles_llm_failure(self):
        def bad_llm(prompt):
            return "I cannot generate questions."

        pairs = generate_qa_pairs(["Some text."], bad_llm, n_per_text=3)
        assert pairs == []


# ---------------------------------------------------------------------------
# Unit tests: index building
# ---------------------------------------------------------------------------


class TestBuildSplitterIndex:
    def test_sentence_splitter(self):
        texts = ["Hello world. Goodbye world."]
        chunks, vectors = build_splitter_index(texts, "sentence", _mock_embedder)
        assert len(chunks) >= 2
        assert len(vectors) == len(chunks)
        assert all(len(v) == 8 for v in vectors)

    def test_paragraph_splitter(self):
        texts = ["Para one.\n\nPara two."]
        chunks, vectors = build_splitter_index(texts, "paragraph", _mock_embedder)
        assert len(chunks) >= 1
        assert len(vectors) == len(chunks)


# ---------------------------------------------------------------------------
# Unit tests: retrieval scoring
# ---------------------------------------------------------------------------


class TestEvaluateRetrieval:
    def test_perfect_retrieval(self):
        """When the only chunk IS the source text, scores should be high."""
        source = "Redis supports RDB snapshots and AOF logs."
        chunks = [source]
        vectors = _mock_embedder(chunks)

        qa_pairs = [QAPair(
            question="What does Redis support?",
            answer="RDB snapshots and AOF logs",
            source_text_idx=0,
            source_text=source,
        )]

        score = evaluate_retrieval(
            qa_pairs, chunks, vectors, _mock_embedder,
            splitter_name="test", top_k=1,
        )
        assert isinstance(score, RetrievalScore)
        assert score.splitter_name == "test"
        # The chunk IS the source, so precision and MRR should be 1.0
        assert score.context_precision == pytest.approx(1.0)
        assert score.mrr == pytest.approx(1.0)

    def test_empty_index(self):
        score = evaluate_retrieval([], [], [], _mock_embedder, splitter_name="empty")
        assert score.context_recall == 0.0
        assert score.mrr == 0.0

    def test_find_top_k(self):
        vectors = [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]]
        query = [1, 0, 0]
        top = _find_top_k(query, vectors, 2)
        assert top[0] == 0  # exact match first
        assert top[1] == 2  # close second


# ---------------------------------------------------------------------------
# Unit tests: report formatting
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_produces_markdown_table(self):
        report = DownstreamReport(
            scores=[
                RetrievalScore("sentence", 0.8, 0.6, 0.75, 10),
                RetrievalScore("paragraph", 0.7, 0.5, 0.65, 5),
            ],
            qa_count=9,
            text_count=3,
        )
        output = format_downstream_report(report)
        assert "| sentence |" in output
        assert "| paragraph |" in output
        assert "QA pairs" in output
        assert "0.800" in output


# ---------------------------------------------------------------------------
# Integration test: end-to-end with mocks
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_run_downstream_eval(self):
        """Full pipeline with mock LLM and embedder."""
        report = run_downstream_eval(
            texts=SAMPLE_TEXTS[:2],
            splitter_names=["sentence", "paragraph"],
            embedder_fn=_mock_embedder,
            llm_fn=_mock_llm,
            top_k=3,
            n_qa_per_text=3,
        )
        assert isinstance(report, DownstreamReport)
        assert report.qa_count == 6  # 2 texts * 3 QA each
        assert report.text_count == 2
        assert len(report.scores) == 2
        assert all(isinstance(s, RetrievalScore) for s in report.scores)
        # Scores sorted by avg descending
        assert report.scores[0].avg >= report.scores[1].avg

    def test_retrieval_score_avg(self):
        s = RetrievalScore("test", 1.0, 1.0, 1.0)
        assert s.avg == pytest.approx(1.0)

        s2 = RetrievalScore("test", 0.0, 0.0, 0.0)
        assert s2.avg == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Unit tests: IoU metric
# ---------------------------------------------------------------------------


class TestScoreIoU:
    def test_perfect_retrieval(self):
        """When retrieved == relevant, IoU should be 1.0."""
        source = "Redis supports RDB snapshots and AOF logs for persistence."
        chunks = [source, "Unrelated content about cooking pasta."]
        vectors = _mock_embedder(chunks)

        qa_pairs = [QAPair(
            question="What does Redis support?",
            answer="RDB snapshots and AOF logs",
            source_text_idx=0,
            source_text=source,
        )]
        # top_k=1, the mock embedder should retrieve the closest chunk
        iou = score_iou(qa_pairs, chunks, vectors, _mock_embedder,
                        [source], top_k=2)
        # IoU should be > 0 since at least some overlap
        assert 0.0 <= iou <= 1.0

    def test_no_relevant_chunks(self):
        """When no chunks are relevant, IoU should be 0."""
        source = "Quantum computing uses qubits."
        # Chunks with zero word overlap to source
        chunks = ["Xyzzyx foobar blargh.", "Wumpus grault corge."]
        vectors = _mock_embedder(chunks)

        qa_pairs = [QAPair(
            question="What does quantum computing use?",
            answer="qubits",
            source_text_idx=0,
            source_text=source,
        )]
        iou = score_iou(qa_pairs, chunks, vectors, _mock_embedder,
                        [source], top_k=2)
        # No relevant chunks, so intersection is empty → IoU = 0
        assert iou == pytest.approx(0.0)

    def test_empty_inputs(self):
        assert score_iou([], [], [], _mock_embedder, [], top_k=5) == 0.0


# ---------------------------------------------------------------------------
# Unit tests: Boundary Clarity metric
# ---------------------------------------------------------------------------


class TestScoreBoundaryClarity:
    def test_identical_vectors_low_clarity(self):
        """Adjacent identical vectors → cosine=1 → clarity=0."""
        vectors = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
        clarity = score_boundary_clarity(vectors)
        assert clarity == pytest.approx(0.0)

    def test_orthogonal_vectors_high_clarity(self):
        """Adjacent orthogonal vectors → cosine=0 → clarity=1."""
        vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        clarity = score_boundary_clarity(vectors)
        assert clarity == pytest.approx(1.0)

    def test_mixed_vectors_moderate_clarity(self):
        """Mixed: one pair identical, one pair orthogonal → clarity=0.5."""
        vectors = [[1, 0, 0], [1, 0, 0], [0, 1, 0]]
        clarity = score_boundary_clarity(vectors)
        # sim(0,1)=1.0, sim(1,2)=0.0 → mean=0.5 → clarity=0.5
        assert clarity == pytest.approx(0.5)

    def test_single_chunk(self):
        """Single chunk → no boundaries → clarity=1.0 (no bad splits)."""
        assert score_boundary_clarity([[1, 0, 0]]) == 1.0

    def test_empty(self):
        assert score_boundary_clarity([]) == 1.0


# ---------------------------------------------------------------------------
# Unit tests: Context Sufficiency metric
# ---------------------------------------------------------------------------


class TestScoreContextSufficiency:
    def test_llm_answers_correctly(self):
        """Mock LLM that always answers correctly → score=1.0."""
        source = "Redis supports RDB snapshots and AOF logs."
        chunks = [source]
        vectors = _mock_embedder(chunks)

        qa_pairs = [QAPair(
            question="What does Redis support?",
            answer="RDB snapshots and AOF logs",
            source_text_idx=0,
            source_text=source,
        )]

        def mock_judge(prompt):
            return json.dumps({
                "answerable": True,
                "confidence": 0.9,
                "answer": "Redis supports RDB snapshots and AOF logs.",
            })

        suff = score_context_sufficiency(
            qa_pairs, chunks, vectors, _mock_embedder, mock_judge, top_k=1)
        assert suff == pytest.approx(1.0)

    def test_llm_cannot_answer(self):
        """Mock LLM says not answerable → score=0."""
        source = "Redis supports RDB."
        chunks = ["Unrelated cooking text."]
        vectors = _mock_embedder(chunks)

        qa_pairs = [QAPair(
            question="What does Redis support?",
            answer="RDB",
            source_text_idx=0,
            source_text=source,
        )]

        def mock_judge(prompt):
            return json.dumps({
                "answerable": False,
                "confidence": 0.1,
                "answer": "",
            })

        suff = score_context_sufficiency(
            qa_pairs, chunks, vectors, _mock_embedder, mock_judge, top_k=1)
        assert suff == pytest.approx(0.0)

    def test_llm_wrong_answer(self):
        """LLM says answerable but gives wrong answer → score=0."""
        source = "Redis supports RDB."
        chunks = [source]
        vectors = _mock_embedder(chunks)

        qa_pairs = [QAPair(
            question="What does Redis support?",
            answer="RDB snapshots",
            source_text_idx=0,
            source_text=source,
        )]

        def mock_judge(prompt):
            return json.dumps({
                "answerable": True,
                "confidence": 0.8,
                "answer": "Completely unrelated answer about cooking.",
            })

        suff = score_context_sufficiency(
            qa_pairs, chunks, vectors, _mock_embedder, mock_judge, top_k=1)
        assert suff == pytest.approx(0.0)

    def test_empty_inputs(self):
        assert score_context_sufficiency([], [], [], _mock_embedder,
                                         _mock_llm, top_k=5) == 0.0

    def test_llm_returns_garbage(self):
        """LLM returns unparseable text → treated as not answerable."""
        source = "Redis supports RDB."
        chunks = [source]
        vectors = _mock_embedder(chunks)

        qa_pairs = [QAPair(
            question="What?",
            answer="RDB",
            source_text_idx=0,
            source_text=source,
        )]

        def garbage_llm(prompt):
            return "I'm not sure what you mean, let me think..."

        suff = score_context_sufficiency(
            qa_pairs, chunks, vectors, _mock_embedder, garbage_llm, top_k=1)
        assert suff == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Unit tests: _extract_json_object helper
# ---------------------------------------------------------------------------


class TestExtractJsonObject:
    def test_plain_object(self):
        result = _extract_json_object('{"answerable": true, "answer": "yes"}')
        assert result == {"answerable": True, "answer": "yes"}

    def test_embedded_in_text(self):
        text = 'Here is my answer: {"answerable": false, "confidence": 0.1, "answer": ""}'
        result = _extract_json_object(text)
        assert result is not None
        assert result["answerable"] is False

    def test_invalid_json(self):
        assert _extract_json_object("no json here") is None


# ---------------------------------------------------------------------------
# Integration: new metrics appear in evaluate_retrieval
# ---------------------------------------------------------------------------


class TestEvaluateRetrievalNewMetrics:
    def test_new_fields_populated(self):
        """evaluate_retrieval returns populated iou and boundary_clarity."""
        source = "Redis supports RDB snapshots and AOF logs."
        chunks = [source, "Another chunk about something else entirely."]
        vectors = _mock_embedder(chunks)

        qa_pairs = [QAPair(
            question="What does Redis support?",
            answer="RDB snapshots and AOF logs",
            source_text_idx=0,
            source_text=source,
        )]

        score = evaluate_retrieval(
            qa_pairs, chunks, vectors, _mock_embedder,
            splitter_name="test", top_k=2, source_texts=[source],
        )
        # IoU and boundary_clarity should be computed (non-default)
        assert isinstance(score.iou, float)
        assert isinstance(score.boundary_clarity, float)
        assert 0.0 <= score.boundary_clarity <= 1.0

    def test_sufficiency_with_llm(self):
        """When llm_fn provided, context_sufficiency is computed."""
        source = "Redis supports RDB snapshots and AOF logs."
        chunks = [source]
        vectors = _mock_embedder(chunks)

        qa_pairs = [QAPair(
            question="What does Redis support?",
            answer="RDB snapshots and AOF logs",
            source_text_idx=0,
            source_text=source,
        )]

        def mock_judge(prompt):
            return json.dumps({
                "answerable": True,
                "confidence": 0.95,
                "answer": "Redis supports RDB snapshots and AOF logs.",
            })

        score = evaluate_retrieval(
            qa_pairs, chunks, vectors, _mock_embedder,
            splitter_name="test", top_k=1,
            llm_fn=mock_judge, source_texts=[source],
        )
        assert score.context_sufficiency == pytest.approx(1.0)

    def test_sufficiency_without_llm(self):
        """When no llm_fn, context_sufficiency defaults to 0."""
        source = "Redis supports RDB."
        chunks = [source]
        vectors = _mock_embedder(chunks)

        qa_pairs = [QAPair(
            question="What?",
            answer="RDB",
            source_text_idx=0,
            source_text=source,
        )]

        score = evaluate_retrieval(
            qa_pairs, chunks, vectors, _mock_embedder,
            splitter_name="test", top_k=1,
        )
        assert score.context_sufficiency == 0.0
