"""CLI entry point for the splitter benchmark harness.

Usage::

    python3 -m prism_rag.cli_benchmark
    python3 -m prism_rag.cli_benchmark --include-llm
    python3 -m prism_rag.cli_benchmark --splitters sentence paragraph
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the PrismRag splitter benchmark harness.",
    )
    parser.add_argument(
        "--splitters",
        nargs="*",
        default=None,
        help="Specific splitter names to benchmark (default: all rule-based).",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        default=False,
        help="Include LLM-based splitters (llm, llm_gleanings). Requires live LLM.",
    )
    args = parser.parse_args(argv)

    # Lazy import to keep CLI startup fast and avoid circular imports.
    from prism_rag.ingest.splitters.benchmark.harness import (
        format_report,
        run_benchmark,
    )

    report = run_benchmark(
        splitter_names=args.splitters,
        include_llm=args.include_llm,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
