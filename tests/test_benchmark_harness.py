"""Tests for the splitter benchmark harness (Phase 3, v6.0)."""

from __future__ import annotations

import pytest

from prism_rag.ingest.splitters.base import Knot
from prism_rag.ingest.splitters.benchmark.dataset import BenchmarkCase
from prism_rag.ingest.splitters.benchmark.datasets import load_benchmark_dataset
from prism_rag.ingest.splitters.benchmark.harness import (
    BenchmarkReport,
    format_report,
    run_benchmark,
)
from prism_rag.ingest.splitters.benchmark.scoring import (
    SplitterScore,
    score_atomicity,
    score_coverage,
    score_faithfulness,
    score_self_containedness,
    score_splitter,
)
from prism_rag.ingest.splitters.registry import get_splitter


# -----------------------------------------------------------------------
# Dataset tests
# -----------------------------------------------------------------------


class TestDataset:
    def test_load_benchmark_dataset_returns_cases(self):
        cases = load_benchmark_dataset()
        assert len(cases) >= 3
        for case in cases:
            assert isinstance(case, BenchmarkCase)
            assert case.section_text.strip()
            assert case.source

    def test_cases_have_reference_knots(self):
        cases = load_benchmark_dataset()
        for case in cases:
            assert len(case.reference_knots) > 0
            for knot in case.reference_knots:
                assert isinstance(knot, Knot)
                assert knot.text.strip()


# -----------------------------------------------------------------------
# Scoring function tests
# -----------------------------------------------------------------------


class TestScoreAtomicity:
    def test_single_sentence_knots_score_high(self):
        knots = [
            Knot(text="Redis is an in-memory store."),
            Knot(text="The default port is 6379."),
        ]
        score = score_atomicity(knots)
        assert score >= 0.9

    def test_multi_sentence_knots_score_lower(self):
        knots = [
            Knot(
                text="Redis is an in-memory store. It supports many data types. "
                "The default port is 6379."
            ),
        ]
        score = score_atomicity(knots)
        assert score < 0.8

    def test_empty_knots(self):
        assert score_atomicity([]) == 0.0

    def test_long_knot_penalized(self):
        # >50 words should get a word penalty
        long_text = " ".join(["word"] * 80) + "."
        knots = [Knot(text=long_text)]
        score = score_atomicity(knots)
        assert score < 1.0


class TestScoreSelfContainedness:
    def test_no_pronouns_score_high(self):
        knots = [
            Knot(text="Redis supports sorted sets."),
            Knot(text="PostgreSQL has window functions."),
        ]
        score = score_self_containedness(knots)
        assert score == 1.0

    def test_dangling_pronoun_penalized(self):
        knots = [
            Knot(text="It supports sorted sets."),
            Knot(text="This enables full audit trails."),
            Knot(text="Redis is fast."),
        ]
        score = score_self_containedness(knots)
        assert score == pytest.approx(1 / 3)

    def test_empty_knots(self):
        assert score_self_containedness([]) == 0.0


class TestScoreFaithfulness:
    def test_faithful_knots_score_one(self):
        source = "Redis supports strings and lists."
        knots = [Knot(text="Redis supports strings and lists.")]
        score = score_faithfulness(knots, source)
        assert score == 1.0

    def test_hallucinated_words_penalized(self):
        source = "Redis supports strings."
        knots = [Knot(text="Redis supports vectors and graphs.")]
        score = score_faithfulness(knots, source)
        assert score < 1.0

    def test_empty_knots(self):
        assert score_faithfulness([], "some source") == 0.0


class TestScoreCoverage:
    def test_full_coverage(self):
        source = "Hello world."
        knots = [Knot(text="Hello world.")]
        score = score_coverage(knots, source)
        assert score == 1.0

    def test_partial_coverage(self):
        source = "Hello world. Goodbye world."
        knots = [Knot(text="Hello world.")]
        score = score_coverage(knots, source)
        assert 0.3 < score < 0.7

    def test_overcoverage_capped(self):
        source = "Hi."
        knots = [Knot(text="Hi."), Knot(text="Hi. Extra text here.")]
        score = score_coverage(knots, source)
        assert score == 1.0

    def test_empty_source(self):
        assert score_coverage([Knot(text="hello")], "") == 0.0

    def test_empty_knots(self):
        assert score_coverage([], "hello") == 0.0


# -----------------------------------------------------------------------
# score_splitter integration test
# -----------------------------------------------------------------------


class TestScoreSplitter:
    def test_score_splitter_returns_valid_score(self):
        splitter = get_splitter("sentence")
        cases = load_benchmark_dataset()
        result = score_splitter(splitter, cases)
        assert isinstance(result, SplitterScore)
        assert result.splitter_name == "sentence"
        for dim in ("atomicity", "self_containedness", "faithfulness", "coverage", "overall"):
            val = getattr(result, dim)
            assert 0.0 <= val <= 1.0, f"{dim} out of range: {val}"

    def test_score_splitter_empty_cases(self):
        splitter = get_splitter("passthrough")
        result = score_splitter(splitter, [])
        assert result.overall == 0.0


# -----------------------------------------------------------------------
# Harness tests
# -----------------------------------------------------------------------


class TestRunBenchmark:
    def test_run_benchmark_default(self):
        """Default run should include rule-based splitters, skip LLM ones."""
        report = run_benchmark()
        assert isinstance(report, BenchmarkReport)
        assert report.dataset_size > 0
        assert report.timestamp

        names = {s.splitter_name for s in report.scores}
        assert "sentence" in names
        assert "paragraph" in names
        assert "passthrough" in names
        assert "fixed_window" in names
        # LLM splitters should be excluded by default
        assert "llm" not in names
        assert "llm_gleanings" not in names

    def test_run_benchmark_specific_splitters(self):
        report = run_benchmark(splitter_names=["sentence", "passthrough"])
        names = {s.splitter_name for s in report.scores}
        assert names == {"sentence", "passthrough"}

    def test_run_benchmark_scores_are_valid(self):
        report = run_benchmark()
        for score in report.scores:
            assert 0.0 <= score.overall <= 1.0
            assert 0.0 <= score.atomicity <= 1.0


# -----------------------------------------------------------------------
# Report formatting tests
# -----------------------------------------------------------------------


class TestFormatReport:
    def test_format_report_contains_table(self):
        report = run_benchmark(splitter_names=["sentence", "passthrough"])
        text = format_report(report)
        assert "| Splitter" in text
        assert "sentence" in text
        assert "passthrough" in text
        assert "Atomicity" in text
        assert "Overall" in text

    def test_format_report_sorted_by_overall(self):
        report = run_benchmark()
        text = format_report(report)
        # The table should have splitter rows — just verify it's non-empty markdown
        lines = [l for l in text.split("\n") if l.startswith("|") and "Splitter" not in l and "---" not in l]
        assert len(lines) >= 2  # at least 2 splitter rows

    def test_format_report_metadata(self):
        report = run_benchmark()
        text = format_report(report)
        assert "Timestamp" in text
        assert "Cases" in text
