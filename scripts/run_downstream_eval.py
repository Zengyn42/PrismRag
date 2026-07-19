#!/usr/bin/env python3
"""Run downstream RAG evaluation — CLI entry point.

Usage::

    # Use built-in sample texts
    python3 scripts/run_downstream_eval.py --texts builtin

    # Use custom text files
    python3 scripts/run_downstream_eval.py --texts file1.txt file2.txt

    # Specify splitters
    python3 scripts/run_downstream_eval.py --texts builtin --splitters sentence paragraph fixed_window
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prism_rag.ingest.splitters.benchmark.downstream import (
    SAMPLE_TEXTS,
    format_downstream_report,
    ollama_chat,
    ollama_embed,
    run_downstream_eval,
)


def main():
    parser = argparse.ArgumentParser(description="Downstream RAG evaluation")
    parser.add_argument(
        "--texts",
        nargs="+",
        default=["builtin"],
        help="'builtin' for sample texts, or paths to text files",
    )
    parser.add_argument(
        "--splitters",
        nargs="*",
        default=None,
        help="Splitter names to evaluate (default: all non-slow)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval")
    parser.add_argument("--n-qa", type=int, default=3, help="QA pairs per text")
    parser.add_argument(
        "--embed-model", default="qwen3-embedding:8b", help="Ollama embedding model"
    )
    parser.add_argument(
        "--llm-model", default="gemma4:e4b", help="Ollama chat model for QA generation"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Load texts
    if args.texts == ["builtin"]:
        texts = SAMPLE_TEXTS
        print(f"Using {len(texts)} built-in sample texts")
    else:
        texts = []
        for path in args.texts:
            p = Path(path)
            if not p.exists():
                print(f"ERROR: File not found: {path}", file=sys.stderr)
                sys.exit(1)
            texts.append(p.read_text().strip())
        print(f"Loaded {len(texts)} texts from files")

    # Create bound embedder/llm functions
    def embedder_fn(batch: list[str]) -> list[list[float]]:
        return ollama_embed(batch, model=args.embed_model)

    def llm_fn(prompt: str) -> str:
        return ollama_chat(prompt, model=args.llm_model)

    # Run eval
    report = run_downstream_eval(
        texts=texts,
        splitter_names=args.splitters,
        embedder_fn=embedder_fn,
        llm_fn=llm_fn,
        top_k=args.top_k,
        n_qa_per_text=args.n_qa,
    )

    print()
    print(format_downstream_report(report))


if __name__ == "__main__":
    main()
