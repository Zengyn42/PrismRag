"""Downstream RAG evaluation harness — measures how splitters affect retrieval.

Lightweight end-to-end pipeline: split docs -> embed -> retrieve -> score.
No external eval frameworks (RAGAS etc.) — built from scratch using stdlib.

Usage::

    from prism_rag.ingest.splitters.benchmark.downstream import (
        run_downstream_eval, format_downstream_report,
        ollama_embed, ollama_chat,
        score_iou, score_context_sufficiency, score_boundary_clarity,
    )

    report = run_downstream_eval(
        texts=SAMPLE_TEXTS,
        splitter_names=["sentence", "paragraph", "fixed_window"],
        embedder_fn=ollama_embed,
        llm_fn=ollama_chat,
    )
    print(format_downstream_report(report))
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

from prism_rag.ingest.splitters.registry import get_splitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample texts for quick built-in evaluation
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    (
        "Redis supports two persistence mechanisms: RDB snapshots and AOF logs. "
        "RDB creates point-in-time snapshots at configured intervals using fork(). "
        "AOF logs every write operation and can be configured with different fsync "
        "policies: always, everysec, or no. The everysec policy provides a good "
        "balance between durability and performance, losing at most one second of data."
    ),
    (
        "The Leiden algorithm improves upon Louvain by guaranteeing well-connected "
        "communities. It uses a refinement phase that moves nodes between communities "
        "to optimize modularity. The resolution parameter gamma controls community "
        "granularity: higher values produce smaller communities. Leiden runs in "
        "O(n log n) time on sparse graphs."
    ),
    (
        "PostgreSQL MVCC uses transaction IDs (XIDs) to determine row visibility. "
        "Each row version has xmin (creating transaction) and xmax (deleting transaction). "
        "A row is visible if xmin is committed and xmax is either invalid or from an "
        "aborted transaction. VACUUM reclaims dead tuples whose xmax is older than "
        "the oldest active transaction."
    ),
    (
        "gRPC uses HTTP/2 as its transport protocol, enabling multiplexed streams "
        "over a single TCP connection. Protocol Buffers (protobuf) define service "
        "interfaces and message types. gRPC supports four communication patterns: "
        "unary, server streaming, client streaming, and bidirectional streaming. "
        "Deadlines propagate across service boundaries via metadata headers."
    ),
    (
        "The CAP theorem states that a distributed system cannot simultaneously "
        "provide Consistency, Availability, and Partition tolerance. In practice, "
        "network partitions are inevitable, so systems must choose between CP "
        "(consistent but may reject requests during partitions) and AP (available "
        "but may return stale data). DynamoDB chose AP with eventual consistency "
        "as the default, offering optional strongly consistent reads."
    ),
    (
        "B+ trees store all values in leaf nodes connected by sibling pointers. "
        "Internal nodes contain only keys and child pointers, maximizing fanout. "
        "A B+ tree with branching factor b and n keys has height O(log_b(n)). "
        "Range queries are efficient because leaf nodes form a linked list. "
        "Most database indexes use B+ trees due to their excellent disk I/O patterns."
    ),
    (
        "Kubernetes uses etcd as its distributed key-value store for cluster state. "
        "The API server is the only component that directly communicates with etcd. "
        "Controllers watch the API server for changes and reconcile desired state "
        "with actual state. The scheduler assigns pods to nodes based on resource "
        "requests, affinity rules, and taints/tolerations."
    ),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QAPair:
    """A question-answer pair generated from a source text."""

    question: str
    answer: str
    source_text_idx: int
    source_text: str


@dataclass
class RetrievalScore:
    """Retrieval quality metrics for one splitter."""

    splitter_name: str
    context_recall: float  # fraction of QA pairs where answer found in top-k
    context_precision: float  # fraction of top-k chunks relevant to source
    mrr: float  # mean reciprocal rank of first relevant chunk
    chunk_count: int = 0
    iou: float = 0.0  # intersection-over-union of retrieved vs relevant chunk sets
    context_sufficiency: float = 0.0  # LLM-as-judge: can LLM answer from retrieved?
    boundary_clarity: float = 0.0  # 1 - mean(adjacent cosine similarities)

    @property
    def avg(self) -> float:
        """Weighted average: recall 40%, MRR 35%, precision 25%."""
        return 0.4 * self.context_recall + 0.35 * self.mrr + 0.25 * self.context_precision


@dataclass
class DownstreamReport:
    """Full downstream evaluation report."""

    scores: list[RetrievalScore]
    qa_count: int
    text_count: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------


def ollama_embed(
    texts: list[str],
    model: str = "qwen3-embedding:8b",
    host: str | None = None,
) -> list[list[float]]:
    """Embed texts via Ollama /api/embed endpoint."""
    if host is None:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["embeddings"]


def ollama_chat(
    prompt: str,
    model: str = "gemma4:e4b",
    host: str | None = None,
) -> str:
    """Chat with Ollama model (non-streaming, thinking disabled)."""
    if host is None:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "think": False,
        "stream": False,
        "options": {"num_predict": 4096, "num_ctx": 8192},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("message", {}).get("content", "")


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Word overlap scoring
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization."""
    return set(re.findall(r"\w+", text.lower()))


def word_overlap(text_a: str, text_b: str) -> float:
    """Fraction of words in text_b that appear in text_a."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_b)


# ---------------------------------------------------------------------------
# New metrics: IoU, Context Sufficiency, Boundary Clarity
# ---------------------------------------------------------------------------


def score_iou(
    qa_pairs: list[QAPair],
    chunk_texts: list[str],
    vectors: list[list[float]],
    embedder_fn,
    source_texts: list[str],
    top_k: int = 5,
) -> float:
    """Intersection-over-Union of retrieved vs relevant chunk sets.

    For each QA pair:
      - relevant_chunks = chunks with word_overlap >= 0.3 with source_text
      - retrieved_set = top-k chunk indices by embedding similarity
      - IoU = |retrieved & relevant| / |retrieved | relevant|

    Returns mean IoU across all QA pairs.
    """
    if not qa_pairs or not vectors:
        return 0.0

    # Embed all questions
    questions = [qa.question for qa in qa_pairs]
    q_vecs = embedder_fn(questions)

    iou_sum = 0.0
    for i, qa in enumerate(qa_pairs):
        # Relevant set: all chunks with sufficient overlap to source
        relevant_set = set()
        for idx, chunk in enumerate(chunk_texts):
            if word_overlap(chunk, qa.source_text) >= 0.3:
                relevant_set.add(idx)

        # Retrieved set: top-k by cosine similarity
        retrieved_set = set(_find_top_k(q_vecs[i], vectors, top_k))

        # IoU
        intersection = retrieved_set & relevant_set
        union = retrieved_set | relevant_set
        if union:
            iou_sum += len(intersection) / len(union)
        # If both sets empty, contribute 0

    return iou_sum / len(qa_pairs)


_SUFFICIENCY_PROMPT = """\
Given ONLY the following context (retrieved chunks), can you answer this question?
Question: {question}
Context: {context}

Answer with JSON: {{"answerable": true/false, "confidence": 0.0-1.0, "answer": "..."}}
"""


def score_context_sufficiency(
    qa_pairs: list[QAPair],
    chunk_texts: list[str],
    vectors: list[list[float]],
    embedder_fn,
    llm_fn,
    top_k: int = 5,
) -> float:
    """LLM-as-judge context sufficiency (Google ICLR 2025 inspired).

    For each QA pair, ask the LLM if it can answer from retrieved chunks only.
    Score = fraction of pairs where the LLM produces a correct answer
    (word overlap >= 0.5 with gold answer).
    """
    if not qa_pairs or not vectors:
        return 0.0

    questions = [qa.question for qa in qa_pairs]
    q_vecs = embedder_fn(questions)

    correct = 0
    for i, qa in enumerate(qa_pairs):
        top_indices = _find_top_k(q_vecs[i], vectors, top_k)
        context = "\n\n".join(chunk_texts[idx] for idx in top_indices)

        prompt = _SUFFICIENCY_PROMPT.format(question=qa.question, context=context)
        try:
            response = llm_fn(prompt)
            # Try to parse JSON from response
            parsed = _extract_json_object(response)
            if parsed and parsed.get("answerable"):
                llm_answer = parsed.get("answer", "")
                if llm_answer and word_overlap(llm_answer, qa.answer) >= 0.5:
                    correct += 1
        except Exception:
            # LLM failure — treat as not answerable
            pass

    return correct / len(qa_pairs)


def _extract_json_object(text: str) -> dict | None:
    """Parse first JSON object from LLM output."""
    text = text.strip()
    # Try whole text
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Find first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def score_boundary_clarity(vectors: list[list[float]]) -> float:
    """Boundary clarity metric (MoC, ACL 2025 inspired).

    Measures whether chunk boundaries align with semantic transitions.
    Score = 1 - mean(cosine_similarity between adjacent chunks).
    Near 1.0 = boundaries at semantic breaks (good).
    Near 0.0 = boundaries split coherent text (bad).
    """
    if len(vectors) < 2:
        return 1.0  # Single chunk or empty — no bad boundaries

    sim_sum = 0.0
    n_pairs = len(vectors) - 1
    for i in range(n_pairs):
        sim_sum += cosine_similarity(vectors[i], vectors[i + 1])

    mean_sim = sim_sum / n_pairs
    return 1.0 - mean_sim


# ---------------------------------------------------------------------------
# 1. QA pair generation
# ---------------------------------------------------------------------------

_QA_PROMPT = """\
Given the following text, generate {n} factual question-answer pairs.
Each question must be answerable ONLY from information in this specific text.
Answers should be concise (1-2 sentences).

Text:
{text}

Return ONLY a JSON array (no markdown fences, no commentary):
[
  {{"question": "...", "answer": "..."}},
  ...
]
"""


def _extract_json_array(text: str) -> list:
    """Parse first JSON array from LLM output."""
    # Try the whole text first
    text = text.strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Find first [ ... ]
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def generate_qa_pairs(
    texts: list[str],
    llm_fn,
    n_per_text: int = 3,
) -> list[QAPair]:
    """Generate QA pairs from source texts using an LLM."""
    pairs = []
    for idx, text in enumerate(texts):
        prompt = _QA_PROMPT.format(n=n_per_text, text=text)
        response = llm_fn(prompt)
        items = _extract_json_array(response)
        for item in items:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                pairs.append(QAPair(
                    question=item["question"],
                    answer=item["answer"],
                    source_text_idx=idx,
                    source_text=text,
                ))
    logger.info("Generated %d QA pairs from %d texts", len(pairs), len(texts))
    return pairs


# ---------------------------------------------------------------------------
# 2. Index builder
# ---------------------------------------------------------------------------


def build_splitter_index(
    texts: list[str],
    splitter_name: str,
    embedder_fn,
) -> tuple[list[str], list[list[float]]]:
    """Split texts and embed chunks. Returns (chunk_texts, vectors)."""
    splitter = get_splitter(splitter_name)
    chunks: list[str] = []
    for text in texts:
        knots = splitter.split(text)
        for knot in knots:
            if knot.text.strip():
                chunks.append(knot.text)

    if not chunks:
        return [], []

    # Embed in batches of 32
    vectors: list[list[float]] = []
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vecs = embedder_fn(batch)
        vectors.extend(vecs)

    logger.info("Splitter %r: %d chunks embedded", splitter_name, len(chunks))
    return chunks, vectors


def build_splitter_index_with_parents(
    texts: list[str],
    splitter,
    embedder_fn,
) -> tuple[list[str], list[list[float]], list[str]]:
    """Split texts, embed chunks, AND track parent text for each chunk.

    Like :func:`build_splitter_index` but also returns a parallel list of
    parent texts — the original text each chunk was derived from. This
    enables **parent-retrieval mode**: match on fine-grained chunk vectors,
    evaluate/return coarse-grained parent paragraphs.

    Args:
        texts: Source documents.
        splitter: A Splitter instance (not a name string).
        embedder_fn: Embedding function.

    Returns:
        (chunk_texts, vectors, parent_texts) where ``parent_texts[i]``
        is the source text that ``chunk_texts[i]`` was split from.
    """
    chunks: list[str] = []
    parents: list[str] = []
    for text in texts:
        knots = splitter.split(text)
        for knot in knots:
            if knot.text.strip():
                chunks.append(knot.text)
                parents.append(text)

    if not chunks:
        return [], [], []

    # Embed in batches of 32
    vectors: list[list[float]] = []
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vecs = embedder_fn(batch)
        vectors.extend(vecs)

    logger.info("Splitter %r: %d chunks embedded (with parents)", splitter.name, len(chunks))
    return chunks, vectors, parents


# ---------------------------------------------------------------------------
# 3. Retrieval evaluation
# ---------------------------------------------------------------------------


def _find_top_k(
    query_vec: list[float],
    vectors: list[list[float]],
    top_k: int,
) -> list[int]:
    """Return indices of top-k most similar vectors."""
    sims = [(i, cosine_similarity(query_vec, v)) for i, v in enumerate(vectors)]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in sims[:top_k]]


def evaluate_retrieval(
    qa_pairs: list[QAPair],
    chunk_texts: list[str],
    vectors: list[list[float]],
    embedder_fn,
    splitter_name: str = "unknown",
    top_k: int = 5,
    llm_fn=None,
    source_texts: list[str] | None = None,
    parent_texts: list[str] | None = None,
) -> RetrievalScore:
    """Evaluate retrieval quality for a set of QA pairs against an index.

    Args:
        parent_texts: When provided, enables **parent-retrieval mode**:
            retrieval uses chunk vectors (fine-grained matching), but
            evaluation uses parent_texts[idx] (coarse-grained context).
            ``parent_texts`` must be the same length as ``chunk_texts``,
            mapping each chunk to its parent paragraph/document.
            This implements the Dense X "proposition indexing +
            paragraph retrieval" pattern.
    """
    if not qa_pairs or not vectors:
        return RetrievalScore(
            splitter_name=splitter_name,
            context_recall=0.0,
            context_precision=0.0,
            mrr=0.0,
            chunk_count=len(chunk_texts),
        )

    # Embed all questions at once
    questions = [qa.question for qa in qa_pairs]
    q_vecs = embedder_fn(questions)

    recall_hits = 0
    precision_sum = 0.0
    mrr_sum = 0.0

    for i, qa in enumerate(qa_pairs):
        top_indices = _find_top_k(q_vecs[i], vectors, top_k)

        if parent_texts is not None:
            # Parent-retrieval: match on chunk vectors, evaluate on parent texts
            # Deduplicate parents (multiple chunks may share same parent)
            seen_parents = set()
            top_chunks = []
            for idx in top_indices:
                p = parent_texts[idx]
                if p not in seen_parents:
                    seen_parents.add(p)
                    top_chunks.append(p)
        else:
            top_chunks = [chunk_texts[idx] for idx in top_indices]

        # Context recall: does any top-k chunk contain the answer?
        answer_found = any(
            word_overlap(chunk, qa.answer) >= 0.5 for chunk in top_chunks
        )
        if answer_found:
            recall_hits += 1

        # Context precision: fraction of top-k relevant to source text
        relevant_count = sum(
            1 for chunk in top_chunks
            if word_overlap(chunk, qa.source_text) >= 0.3
        )
        precision_sum += relevant_count / top_k

        # MRR: reciprocal rank of first relevant chunk
        for rank, chunk in enumerate(top_chunks, 1):
            if word_overlap(chunk, qa.source_text) >= 0.3:
                mrr_sum += 1.0 / rank
                break

    n = len(qa_pairs)

    # New metrics
    iou = score_iou(qa_pairs, chunk_texts, vectors, embedder_fn,
                    source_texts or [], top_k)
    boundary = score_boundary_clarity(vectors)

    sufficiency = 0.0
    if llm_fn is not None:
        sufficiency = score_context_sufficiency(
            qa_pairs, chunk_texts, vectors, embedder_fn, llm_fn, top_k)

    return RetrievalScore(
        splitter_name=splitter_name,
        context_recall=recall_hits / n,
        context_precision=precision_sum / n,
        mrr=mrr_sum / n,
        chunk_count=len(chunk_texts),
        iou=iou,
        context_sufficiency=sufficiency,
        boundary_clarity=boundary,
    )


# ---------------------------------------------------------------------------
# 4. End-to-end runner
# ---------------------------------------------------------------------------

# Splitters that are too slow for eval loops by default
_SLOW_SPLITTERS = {"llm", "llm_gleanings"}


def run_downstream_eval(
    texts: list[str],
    splitter_names: list[str] | None = None,
    embedder_fn=None,
    llm_fn=None,
    top_k: int = 5,
    n_qa_per_text: int = 3,
) -> DownstreamReport:
    """Run full downstream RAG evaluation pipeline.

    Args:
        texts: Source documents to evaluate on.
        splitter_names: Which splitters to test. Defaults to all non-slow.
        embedder_fn: Embedding function (texts -> vectors). Defaults to ollama_embed.
        llm_fn: LLM function (prompt -> text). Defaults to ollama_chat.
        top_k: Number of chunks to retrieve per query.
        n_qa_per_text: Number of QA pairs to generate per text.

    Returns:
        DownstreamReport with scores for each splitter.
    """
    if embedder_fn is None:
        embedder_fn = ollama_embed
    if llm_fn is None:
        llm_fn = ollama_chat

    if splitter_names is None:
        from prism_rag.ingest.splitters.registry import list_splitters
        splitter_names = [s for s in list_splitters() if s not in _SLOW_SPLITTERS]

    # Step 1: Generate QA pairs (shared across all splitters)
    logger.info("Generating QA pairs from %d texts...", len(texts))
    qa_pairs = generate_qa_pairs(texts, llm_fn, n_per_text=n_qa_per_text)

    if not qa_pairs:
        logger.warning("No QA pairs generated — aborting eval")
        return DownstreamReport(scores=[], qa_count=0, text_count=len(texts))

    # Step 2: For each splitter, build index and evaluate
    scores: list[RetrievalScore] = []
    for name in splitter_names:
        logger.info("Evaluating splitter: %s", name)
        try:
            chunk_texts, vectors = build_splitter_index(texts, name, embedder_fn)
            score = evaluate_retrieval(
                qa_pairs, chunk_texts, vectors, embedder_fn,
                splitter_name=name, top_k=top_k,
                llm_fn=llm_fn, source_texts=texts,
            )
            scores.append(score)
        except Exception as e:
            logger.error("Splitter %r failed: %s", name, e)
            scores.append(RetrievalScore(
                splitter_name=name,
                context_recall=0.0,
                context_precision=0.0,
                mrr=0.0,
            ))

    # Sort by avg descending
    scores.sort(key=lambda s: s.avg, reverse=True)

    return DownstreamReport(
        scores=scores,
        qa_count=len(qa_pairs),
        text_count=len(texts),
    )


# ---------------------------------------------------------------------------
# 5. Report formatting
# ---------------------------------------------------------------------------


def format_downstream_report(report: DownstreamReport) -> str:
    """Format report as a markdown table."""
    lines = [
        "# Downstream RAG Evaluation Report",
        "",
        f"- **Texts**: {report.text_count}",
        f"- **QA pairs**: {report.qa_count}",
        f"- **Timestamp**: {report.timestamp}",
        "",
        "| Splitter | Chunks | Recall | Precision | MRR | IoU | Sufficiency | Boundary | Avg |",
        "|----------|--------|--------|-----------|-----|-----|-------------|----------|-----|",
    ]
    for s in report.scores:
        lines.append(
            f"| {s.splitter_name} | {s.chunk_count} | "
            f"{s.context_recall:.3f} | {s.context_precision:.3f} | "
            f"{s.mrr:.3f} | {s.iou:.3f} | {s.context_sufficiency:.3f} | "
            f"{s.boundary_clarity:.3f} | {s.avg:.3f} |"
        )
    lines.append("")
    lines.append("*Avg = 0.4*Recall + 0.35*MRR + 0.25*Precision*")
    return "\n".join(lines)
