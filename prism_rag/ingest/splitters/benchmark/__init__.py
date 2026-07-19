"""Benchmark harness for comparing splitter quality.

Run the full benchmark via CLI::

    python3 -m prism_rag.cli_benchmark

Or programmatically::

    from prism_rag.ingest.splitters.benchmark import run_benchmark, format_report
    report = run_benchmark()
    print(format_report(report))
"""

from prism_rag.ingest.splitters.benchmark.dataset import BenchmarkCase
from prism_rag.ingest.splitters.benchmark.datasets import load_benchmark_dataset
from prism_rag.ingest.splitters.benchmark.harness import (
    BenchmarkReport,
    format_report,
    run_benchmark,
)
from prism_rag.ingest.splitters.benchmark.propositionizer import (
    load_propositionizer_dataset,
)
from prism_rag.ingest.splitters.benchmark.scoring import (
    SplitterScore,
    score_gold_alignment,
    score_splitter,
)

__all__ = [
    "BenchmarkCase",
    "BenchmarkReport",
    "SplitterScore",
    "format_report",
    "load_benchmark_dataset",
    "load_propositionizer_dataset",
    "run_benchmark",
    "score_gold_alignment",
    "score_splitter",
]
