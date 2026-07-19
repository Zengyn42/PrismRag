# PrismRag Splitter Evaluation System

> Version: 2026-07-19
> Status: Operational (rule-based splitters verified; LLM splitters pending downstream eval)

## Overview

Two-layer evaluation: **upstream quality** (splitting 本身好不好) + **downstream retrieval** (切出来的东西能不能检索到).

两层互补——上游评的是"拆得像不像 GPT-4"，下游评的是"拆完能不能帮你找到答案"。两者方向可能冲突（切太细上游分高但下游反而差）。

---

## Layer 1: Upstream — Splitting Quality Benchmark

### Data Source

- **Dataset**: `chentong00/propositionizer-wiki-data` (HuggingFace)
- **Origin**: Dense X Retrieval 论文 (Chen et al., EMNLP 2024)
- **Size**: 42,857 train / 1,000 validation / 1,000 test
- **Format**: `{sources: "Title: X. Section: Y. Content: Z", targets: "[prop1, prop2, ...]"}`
- **Gold standard**: GPT-4 generated propositions (not human-annotated)
- **Domain**: Wikipedia (encyclopedic text only — NOT technical documentation)

### What Gets Evaluated

Each splitter receives the same input text and produces `list[Knot]`. Scoring compares these knots against gold propositions AND checks intrinsic quality.

### Scoring Dimensions (5)

| Dimension | Weight (with gold) | Weight (no gold) | Algorithm |
|-----------|-------------------|------------------|-----------|
| **atomicity** | 20% | 25% | Per-knot: penalize >1 sentence (0.3/extra) + >50 words (gradual). Score = max(0, 1 - penalties). Average across all knots. |
| **self_containedness** | 15% | 25% | Per-knot: check if text starts with dangling pronoun (it/this/that/these/those/they/he/she via regex). Score = clean_count / total_count. |
| **faithfulness** | 20% | 25% | Per-knot: extract word sets from knot and source, compute `len(knot_words & source_words) / len(knot_words)`. Average across knots. Measures: no hallucinated words. |
| **coverage** | 15% | 25% | `sum(knot_text_length) / source_text_length`, capped at 1.0. Measures: information preserved. |
| **gold_alignment** | 30% | N/A | Greedy best-match word-level F1. See below. |

### Gold Alignment Algorithm (most important dimension)

```
For each predicted knot P_i:
    best_f1_i = max over all gold G_j of: word_F1(P_i, G_j)
Precision = mean(best_f1_i for all i)

For each gold proposition G_j:
    best_f1_j = max over all predicted P_i of: word_F1(G_j, P_i)
Recall = mean(best_f1_j for all j)

Gold-F1 = 2 * Precision * Recall / (Precision + Recall)
```

Where `word_F1(a, b)` = standard F1 over word sets (lowercased, `\w+` tokenized).

### Known Limitations

1. **All heuristic / surface-level** — no semantic understanding
2. **faithfulness** is word-overlap, not BertScore (semantic). "Redis uses RDB" → "Redis employs RDB" gets penalized
3. **gold_alignment** is word-level F1, not semantic F1. Paraphrases get penalized
4. **No minimality dimension** — can't detect "this knot could be split further"
5. **Domain mismatch** — gold is Wikipedia; our actual use is technical documentation

### How to Run

```bash
# Rule-based splitters only (instant)
python3 -m prism_rag.cli_benchmark

# With LLM splitters (requires Ollama)
OLLAMA_HOST=http://localhost:11434 python3 scripts/run_llm_benchmark.py --cases 100 --model gemma4:e4b

# Large model comparison
OLLAMA_HOST=http://localhost:11434 python3 scripts/run_llm_benchmark.py --cases 20 --model gemma4:31b
```

### Results (2026-07-19, gemma4:e4b, 100 cases)

| Splitter | Atomicity | Self-Cont. | Faithful | Coverage | Gold-F1 | Overall |
|----------|-----------|------------|----------|----------|---------|---------|
| v2_propositions | 0.979 | 0.962 | 0.913 | 0.973 | **0.780** | **0.903** |
| llm (V1) | 0.980 | 0.988 | 0.923 | 0.954 | 0.750 | 0.897 |
| v2_molecular | 0.950 | 0.958 | 0.902 | 0.916 | 0.764 | 0.880 |
| sentence | 0.998 | 0.892 | 1.000 | 0.995 | 0.657 | 0.880 |
| v2_decontext | 0.971 | 0.975 | 0.864 | 0.960 | 0.735 | 0.877 |

### Model Size Effect (20 cases)

| Method | Gold-F1 | Overall |
|--------|---------|---------|
| v2_propositions @ gemma4:31b | **0.821** | **0.926** |
| v2_decontext @ gemma4:31b | 0.809 | 0.920 |
| v2_decontext @ gemma4:e4b | 0.769 | 0.910 |
| sentence (baseline) | 0.654 | 0.881 |

---

## Layer 2: Downstream — Retrieval Quality Evaluation

### Purpose

Measures whether splitting actually helps find answers. A splitter with perfect Gold-F1 is useless if the resulting chunks can't be retrieved.

### Pipeline

```
Step 1: Input texts (technical docs or builtin samples)
           ↓
Step 2: LLM generates QA pairs from each text
        (gemma4:e4b, think:false, 3 QA per text)
           ↓
Step 3: For each splitter:
        - Split all texts → chunks
        - Embed all chunks (qwen3-embedding:8b)
           ↓
Step 4: For each QA pair:
        - Embed the question
        - Cosine similarity search → top-k chunks
        - Score against ground truth
           ↓
Step 5: Aggregate → per-splitter scores
```

### Scoring Metrics (3)

| Metric | Algorithm | Measures |
|--------|-----------|----------|
| **context_recall** | For each QA: does ANY top-k chunk have word_overlap >= 0.5 with the gold answer? Binary per QA, averaged. | Can we find the answer at all? |
| **context_precision** | For each QA: what fraction of top-k chunks have word_overlap >= 0.3 with the source text? Averaged. | How much noise in results? |
| **MRR** (Mean Reciprocal Rank) | For each QA: 1/rank of first chunk with word_overlap >= 0.3 with source. 0 if none in top-k. Averaged. | How quickly do we find relevant content? |

Overall = weighted average: recall 40% + precision 30% + MRR 30%

### QA Pair Generation

LLM prompt generates factual questions from source text:
```
Given this text, generate N question-answer pairs.
Requirements:
- Questions must be answerable ONLY from this text
- Answers must be short (1-2 sentences)
- Cover different facts in the text
Output: JSON array [{question, answer}]
```

### Key Insight: Granularity Trade-off

| Granularity | Upstream (Gold-F1) | Downstream (Recall) | Why |
|-------------|-------------------|--------------------|----|
| passthrough (no split) | 0.42 (worst) | 1.00 (best) | Whole text always contains answer, but embedding is diluted |
| sentence | 0.66 (good) | 0.62 (worst) | Too fine: answer scattered across chunks, individual embeddings lose context |
| LLM atomic | 0.78 (best) | **TBD** | Self-contained propositions should balance both |

The optimal splitter maximizes BOTH layers. This is why LLM-based proposition splitting exists — it aims to produce chunks that are atomic (good embeddings) AND self-contained (answer not split across chunks).

### How to Run

```bash
# Built-in sample texts (7 texts, ~21 QA pairs, ~2 min)
OLLAMA_HOST=http://localhost:11434 python3 scripts/run_downstream_eval.py --texts builtin

# Custom texts
OLLAMA_HOST=http://localhost:11434 python3 scripts/run_downstream_eval.py --texts /path/to/doc1.md /path/to/doc2.md

# Specific splitters only
OLLAMA_HOST=http://localhost:11434 python3 scripts/run_downstream_eval.py --texts builtin --splitters sentence,paragraph,fixed_window
```

### Results (2026-07-19, builtin 7 texts, 21 QA pairs, rule-based only)

| Splitter | Context Recall | Context Precision | MRR | Avg |
|----------|---------------|-------------------|-----|-----|
| paragraph | 1.000 | 0.400 | 1.000 | **0.800** |
| passthrough | 1.000 | 0.400 | 1.000 | **0.800** |
| fixed_window | 1.000 | 0.381 | 0.995 | 0.792 |
| sentence | 0.905 | 0.257 | 0.705 | 0.622 |

---

## Combined Interpretation

Neither layer alone tells the full story:

- **Upstream only**: v2_propositions wins → but does it actually help retrieval?
- **Downstream only**: passthrough wins → but that's because with 7 texts the whole doc is one chunk (trivial retrieval)

The right conclusion comes from running downstream eval on a **larger corpus** (50+ texts) where passthrough chunks become too diluted in embedding space. At scale, atomic propositions should win both layers.

---

## File Locations

| Component | Path |
|-----------|------|
| Upstream harness | `prism_rag/ingest/splitters/benchmark/harness.py` |
| Upstream scoring | `prism_rag/ingest/splitters/benchmark/scoring.py` |
| Propositionizer dataset loader | `prism_rag/ingest/splitters/benchmark/propositionizer.py` |
| Downstream harness | `prism_rag/ingest/splitters/benchmark/downstream.py` |
| LLM benchmark script | `scripts/run_llm_benchmark.py` |
| Downstream eval script | `scripts/run_downstream_eval.py` |
| CLI (rule-based) | `prism_rag/cli_benchmark.py` |
| Tests (upstream) | `tests/test_benchmark_harness.py` |
| Tests (downstream) | `tests/test_downstream_eval.py` |

---

## Future Improvements

1. **Semantic scoring**: Replace word-level F1 with BertScore or embedding cosine for gold_alignment and faithfulness
2. **Minimality dimension**: LLM-as-judge "can this knot be split further?"
3. **Domain-specific eval set**: Annotate 100-200 QA pairs from actual vault technical documents
4. **Scale downstream eval**: 50+ texts to properly stress embedding-based retrieval
5. **LLM splitters in downstream**: Run v2_propositions through the retrieval pipeline
6. **Confidence intervals**: Bootstrap sampling over cases to report statistical significance
