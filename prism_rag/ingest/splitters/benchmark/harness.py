"""Benchmark harness — run all splitters on a dataset and produce a report."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from prism_rag.ingest.splitters.benchmark.dataset import BenchmarkCase
from prism_rag.ingest.splitters.benchmark.datasets import load_benchmark_dataset
from prism_rag.ingest.splitters.benchmark.scoring import SplitterScore, score_splitter
from prism_rag.ingest.splitters.registry import get_splitter, list_splitters

# Splitters that require a live LLM and are skipped by default.
_LLM_SPLITTERS = {"llm", "llm_gleanings"}


@dataclass
class BenchmarkReport:
    """Results of a benchmark run.

    Attributes:
        scores: Per-splitter quality scores.
        timestamp: When the benchmark was run (UTC ISO string).
        dataset_size: Number of benchmark cases evaluated.
        dataset_sources: List of case source labels.
    """

    scores: list[SplitterScore]
    timestamp: str = ""
    dataset_size: int = 0
    dataset_sources: list[str] = field(default_factory=list)


def run_benchmark(
    *,
    splitter_names: list[str] | None = None,
    dataset: list[BenchmarkCase] | None = None,
    include_llm: bool = False,
) -> BenchmarkReport:
    """Run the benchmark harness.

    Args:
        splitter_names: Specific splitters to evaluate. If *None*, all
            registered splitters are used (minus LLM-based ones unless
            *include_llm* is True).
        dataset: Benchmark cases to use. If *None*, the built-in dataset
            is loaded via :func:`load_benchmark_dataset`.
        include_llm: If False (default), ``llm`` and ``llm_gleanings``
            splitters are skipped even if they appear in *splitter_names*
            or the registry.

    Returns:
        A :class:`BenchmarkReport` with scores for each evaluated splitter.
    """
    if dataset is None:
        dataset = load_benchmark_dataset()

    if splitter_names is None:
        splitter_names = list_splitters()

    if not include_llm:
        splitter_names = [n for n in splitter_names if n not in _LLM_SPLITTERS]

    scores: list[SplitterScore] = []
    for name in sorted(splitter_names):
        splitter = get_splitter(name)
        score = score_splitter(splitter, dataset)
        scores.append(score)

    return BenchmarkReport(
        scores=scores,
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_size=len(dataset),
        dataset_sources=[c.source for c in dataset],
    )


def format_report(report: BenchmarkReport) -> str:
    """Format a benchmark report as a markdown table.

    Args:
        report: The report to format.

    Returns:
        A human-readable markdown string.
    """
    lines: list[str] = []
    lines.append(f"# Splitter Benchmark Report")
    lines.append(f"")
    lines.append(f"- **Timestamp**: {report.timestamp}")
    lines.append(f"- **Cases**: {report.dataset_size}")
    lines.append(f"")
    has_gold = any(s.gold_alignment > 0 for s in report.scores)

    if has_gold:
        lines.append(
            "| Splitter | Atomicity | Self-Cont. | Faithful | Coverage | Gold-F1 | Overall |"
        )
        lines.append(
            "|----------|-----------|------------|----------|----------|---------|---------|"
        )
        for s in sorted(report.scores, key=lambda x: x.overall, reverse=True):
            lines.append(
                f"| {s.splitter_name:<14s} "
                f"| {s.atomicity:9.4f} "
                f"| {s.self_containedness:10.4f} "
                f"| {s.faithfulness:8.4f} "
                f"| {s.coverage:8.4f} "
                f"| {s.gold_alignment:7.4f} "
                f"| {s.overall:7.4f} |"
            )
    else:
        lines.append(
            "| Splitter | Atomicity | Self-Contained | Faithfulness | Coverage | Overall |"
        )
        lines.append(
            "|----------|-----------|----------------|--------------|----------|---------|"
        )
        for s in sorted(report.scores, key=lambda x: x.overall, reverse=True):
            lines.append(
                f"| {s.splitter_name:<14s} "
                f"| {s.atomicity:9.4f} "
                f"| {s.self_containedness:14.4f} "
                f"| {s.faithfulness:12.4f} "
                f"| {s.coverage:8.4f} "
                f"| {s.overall:7.4f} |"
            )

    lines.append("")
    return "\n".join(lines)
