"""Multi-granularity Knot retrieval — 3-layer index (L0/L1/L2) + 5 retrieval strategies.

Builds a hierarchical index from atomized knots:
  L0: Individual knots (atomic facts)
  L1: Adjacent knot groups from same source (window-based)
  L2: Leiden-clustered L1 groups with LLM-generated tags

Retrieval strategies:
  flat_l0       — baseline cosine on L0 vectors
  flat_l1       — cosine on L1 group vectors
  parent_l0     — L0 cosine match, return parent L1 text
  multi_layer   — L2 tag filter -> L1 cosine within scope
  collapsed     — all L0+L1 vectors searched together (RAPTOR style)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords for keyword extraction
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once here there when where why how all each every both few "
    "more most other some such no nor not only own same so than too very and "
    "but or if while that what which who whom this these those it its he his "
    "she her they them their we our you your i me my".split()
)

# ---------------------------------------------------------------------------
# Tag generation prompt
# ---------------------------------------------------------------------------

TAG_PROMPT = """Given these related knowledge statements, generate ONE concise topic tag (2-5 words) that captures their shared theme. Return ONLY the tag, nothing else.

Statements:
{statements}

Tag:"""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MultiGranularityIndex:
    """Complete multi-layer index for retrieval."""

    # L0
    l0_texts: list[str]
    l0_vectors: list[list[float]]
    l0_source_idx: list[int]

    # L1
    l1_texts: list[str]
    l1_vectors: list[list[float]]
    l1_members: list[list[int]]  # L0 indices in each L1 group
    l1_source_idx: list[int]

    # L2
    l2_tags: list[str]
    l2_vectors: list[list[float]]
    l2_members: list[list[int]]  # L1 indices in each L2 cluster

    # Source texts (for evaluation)
    source_texts: list[str]

    # Mapping: L1 index -> L2 cluster index
    l1_to_l2: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cosine similarity (local copy to avoid circular imports)
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _find_top_k(
    query_vec: list[float],
    vectors: list[list[float]],
    top_k: int,
    candidate_indices: list[int] | None = None,
) -> list[int]:
    """Return indices of top-k most similar vectors.

    If candidate_indices is provided, only search within those indices.
    """
    if candidate_indices is not None:
        sims = [(i, _cosine(query_vec, vectors[i])) for i in candidate_indices]
    else:
        sims = [(i, _cosine(query_vec, v)) for i, v in enumerate(vectors)]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in sims[:top_k]]


# ---------------------------------------------------------------------------
# Step 2: L1 grouping
# ---------------------------------------------------------------------------


def build_l1_groups(
    l0_texts: list[str],
    l0_source_idx: list[int],
    window: int = 3,
) -> list[tuple[str, list[int], int]]:
    """Group consecutive L0 knots from same source into L1 groups.

    Returns list of (l1_text, member_l0_indices, source_idx).
    """
    groups: list[tuple[str, list[int], int]] = []
    i = 0
    while i < len(l0_texts):
        src = l0_source_idx[i]
        members: list[int] = []
        while i < len(l0_texts) and l0_source_idx[i] == src and len(members) < window:
            members.append(i)
            i += 1
        l1_text = " ".join(l0_texts[j] for j in members)
        groups.append((l1_text, members, src))
    return groups


# ---------------------------------------------------------------------------
# Step 3: L2 clustering via Leiden
# ---------------------------------------------------------------------------


def _cluster_l1_leiden(
    l1_vectors: list[list[float]],
    threshold: float = 0.5,
    seed: int = 42,
) -> list[list[int]]:
    """Cluster L1 groups using Leiden on a cosine similarity graph.

    Returns list of clusters, each a list of L1 indices.
    """
    n = len(l1_vectors)
    if n < 10:
        # Too few groups — single cluster
        return [list(range(n))]

    import igraph as ig
    import leidenalg

    # Build similarity graph
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine(l1_vectors[i], l1_vectors[j])
            if sim > threshold:
                edges.append((i, j))
                weights.append(sim)

    g = ig.Graph(n=n, edges=edges, directed=False)
    if weights:
        g.es["weight"] = weights

    try:
        partition = leidenalg.find_partition(
            g,
            leidenalg.ModularityVertexPartition,
            weights="weight" if weights else None,
            n_iterations=-1,
            seed=seed,
        )
        clusters = [list(members) for members in partition]
    except Exception:
        clusters = [list(range(n))]

    # Remove empty clusters
    clusters = [c for c in clusters if c]
    if not clusters:
        clusters = [list(range(n))]

    return clusters


def _generate_l2_tags(
    clusters: list[list[int]],
    l1_texts: list[str],
    llm_fn,
) -> list[str]:
    """Generate a topic tag for each L2 cluster using LLM."""
    tags: list[str] = []
    for cluster in clusters:
        # Collect member texts (cap at 10 for prompt length)
        statements = "\n".join(
            f"- {l1_texts[i]}" for i in cluster[:10]
        )
        prompt = TAG_PROMPT.format(statements=statements)
        try:
            tag = llm_fn(prompt).strip().strip('"').strip("'")
            # Take first line only
            tag = tag.split("\n")[0].strip()
            if not tag:
                tag = "general"
        except Exception:
            tag = "general"
        tags.append(tag)
    return tags


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------


def build_multi_granularity_index(
    texts: list[str],
    splitter,
    embedder_fn,
    llm_fn,
    l1_window: int = 3,
) -> MultiGranularityIndex:
    """Build a 3-layer multi-granularity index.

    Args:
        texts: Source documents.
        splitter: A Splitter instance for L0 atomization.
        embedder_fn: Function(list[str]) -> list[list[float]].
        llm_fn: Function(str) -> str for tag generation.
        l1_window: Number of consecutive L0 knots per L1 group.

    Returns:
        MultiGranularityIndex with all layers populated.
    """
    # Step 1: L0 atomization
    l0_texts: list[str] = []
    l0_source_idx: list[int] = []
    for idx, text in enumerate(texts):
        knots = splitter.split(text)
        for knot in knots:
            if knot.text.strip():
                l0_texts.append(knot.text)
                l0_source_idx.append(idx)

    logger.info("L0: %d knots from %d sources", len(l0_texts), len(texts))

    if not l0_texts:
        return MultiGranularityIndex(
            l0_texts=[], l0_vectors=[], l0_source_idx=[],
            l1_texts=[], l1_vectors=[], l1_members=[], l1_source_idx=[],
            l2_tags=[], l2_vectors=[], l2_members=[],
            source_texts=texts, l1_to_l2=[],
        )

    # Step 2: L1 grouping
    l1_groups = build_l1_groups(l0_texts, l0_source_idx, window=l1_window)
    l1_texts = [g[0] for g in l1_groups]
    l1_members = [g[1] for g in l1_groups]
    l1_source_idx = [g[2] for g in l1_groups]
    logger.info("L1: %d groups (window=%d)", len(l1_texts), l1_window)

    # Embed L0 and L1 in batches
    batch_size = 32
    all_texts_to_embed = l0_texts + l1_texts
    all_vectors: list[list[float]] = []
    for i in range(0, len(all_texts_to_embed), batch_size):
        batch = all_texts_to_embed[i : i + batch_size]
        vecs = embedder_fn(batch)
        all_vectors.extend(vecs)

    l0_vectors = all_vectors[: len(l0_texts)]
    l1_vectors = all_vectors[len(l0_texts) :]

    # Step 3: L2 clustering + tag generation
    clusters = _cluster_l1_leiden(l1_vectors)
    l2_tags = _generate_l2_tags(clusters, l1_texts, llm_fn)
    logger.info("L2: %d clusters", len(clusters))

    # Build l1_to_l2 mapping
    l1_to_l2 = [0] * len(l1_texts)
    for cluster_idx, members in enumerate(clusters):
        for l1_idx in members:
            l1_to_l2[l1_idx] = cluster_idx

    # Embed L2 tags
    if l2_tags:
        l2_vectors = embedder_fn(l2_tags)
    else:
        l2_vectors = []

    return MultiGranularityIndex(
        l0_texts=l0_texts,
        l0_vectors=l0_vectors,
        l0_source_idx=l0_source_idx,
        l1_texts=l1_texts,
        l1_vectors=l1_vectors,
        l1_members=l1_members,
        l1_source_idx=l1_source_idx,
        l2_tags=l2_tags,
        l2_vectors=l2_vectors,
        l2_members=clusters,
        source_texts=texts,
        l1_to_l2=l1_to_l2,
    )


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------


def retrieve_flat_l0(
    query_vec: list[float],
    index: MultiGranularityIndex,
    top_k: int = 5,
) -> list[str]:
    """Flat L0 search -- baseline."""
    if not index.l0_vectors:
        return []
    indices = _find_top_k(query_vec, index.l0_vectors, top_k)
    return [index.l0_texts[i] for i in indices]


def retrieve_flat_l1(
    query_vec: list[float],
    index: MultiGranularityIndex,
    top_k: int = 5,
) -> list[str]:
    """Flat L1 search."""
    if not index.l1_vectors:
        return []
    indices = _find_top_k(query_vec, index.l1_vectors, top_k)
    return [index.l1_texts[i] for i in indices]


def retrieve_parent_l0(
    query_vec: list[float],
    index: MultiGranularityIndex,
    top_k: int = 5,
) -> list[str]:
    """L0 search, return parent L1 text (deduplicated)."""
    if not index.l0_vectors:
        return []
    l0_indices = _find_top_k(query_vec, index.l0_vectors, top_k * 2)

    # Map L0 -> L1 parent
    l0_to_l1: dict[int, int] = {}
    for l1_idx, members in enumerate(index.l1_members):
        for l0_idx in members:
            l0_to_l1[l0_idx] = l1_idx

    seen_l1: set[int] = set()
    results: list[str] = []
    for l0_idx in l0_indices:
        l1_idx = l0_to_l1.get(l0_idx)
        if l1_idx is not None and l1_idx not in seen_l1:
            seen_l1.add(l1_idx)
            results.append(index.l1_texts[l1_idx])
            if len(results) >= top_k:
                break
    return results


def _extract_keywords(text: str) -> set[str]:
    """Extract non-stopword keywords from text."""
    words = set(re.findall(r"\w+", text.lower()))
    return words - _STOPWORDS


def retrieve_multi_layer(
    query_vec: list[float],
    query_text: str,
    index: MultiGranularityIndex,
    top_k: int = 5,
) -> list[str]:
    """L2 tag filter -> L1 vector search within scope."""
    if not index.l1_vectors:
        return []

    query_keywords = _extract_keywords(query_text)

    # Stage 1: Match L2 tags by keyword overlap
    matched_l2: list[int] = []
    if query_keywords:
        for i, tag in enumerate(index.l2_tags):
            tag_words = set(re.findall(r"\w+", tag.lower()))
            if not tag_words:
                continue
            overlap = len(query_keywords & tag_words) / len(tag_words)
            if overlap >= 0.3:
                matched_l2.append(i)

    # Fallback: L2 vector cosine top-3
    if not matched_l2 and index.l2_vectors:
        matched_l2 = _find_top_k(query_vec, index.l2_vectors, min(3, len(index.l2_vectors)))

    if not matched_l2:
        # Ultimate fallback: search all L1
        return retrieve_flat_l1(query_vec, index, top_k)

    # Stage 2: L1 search within matched L2 clusters
    matched_l2_set = set(matched_l2)
    candidate_l1 = [
        i for i, cluster_idx in enumerate(index.l1_to_l2)
        if cluster_idx in matched_l2_set
    ]

    if not candidate_l1:
        return retrieve_flat_l1(query_vec, index, top_k)

    result_indices = _find_top_k(query_vec, index.l1_vectors, top_k, candidate_indices=candidate_l1)
    return [index.l1_texts[i] for i in result_indices]


def retrieve_collapsed(
    query_vec: list[float],
    index: MultiGranularityIndex,
    top_k: int = 5,
) -> list[str]:
    """All L0+L1 vectors flat-searched together (RAPTOR style)."""
    all_texts = index.l0_texts + index.l1_texts
    all_vectors = index.l0_vectors + index.l1_vectors
    if not all_vectors:
        return []
    indices = _find_top_k(query_vec, all_vectors, top_k)
    return [all_texts[i] for i in indices]
