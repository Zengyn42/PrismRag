# Multi-Granularity Knot Algorithm — v0.1

> Date: 2026-07-19
> Status: First implementation spec

---

## Algorithm Overview

```
Input: list of source texts (documents/sections)
Output: 3-layer knot hierarchy with retrieval indices

Pipeline:
  Step 1: L0 — Atomize texts into propositions (existing v2_propositions splitter)
  Step 2: L1 — Group L0 knots by source + adjacency (2-4 knots per group)
  Step 3: L2 — Cluster L1 groups via Leiden → LLM generates tag per cluster
  Step 4: Build retrieval indices:
          - L0/L1: embedding vectors (cosine search)
          - L2: tag + fallback embedding (keyword match + cosine fallback)
  Step 5: Retrieval: L2 tag filter → L1/L0 vector search within scope
```

---

## Step 1: L0 Atomization

Existing pipeline, no changes:
- Splitter: v2_propositions prompt via Ollama
- Output: list[Knot] — each is a self-contained atomic fact
- Each knot tracks `source_text_idx` (which input text it came from)

## Step 2: L1 Grouping

Group adjacent L0 knots from the same source text into "topic groups":
- Window: 2-4 consecutive L0 knots per L1 group
- Boundary: source text boundary always breaks a group
- L1 text = concatenation of member L0 knot texts
- L1 embedding = embed(L1 text)

```python
def build_l1_groups(l0_knots, source_indices, window=3):
    """Group consecutive L0 knots from same source into L1 groups."""
    groups = []  # list of (l1_text, member_l0_indices, source_idx)
    i = 0
    while i < len(l0_knots):
        src = source_indices[i]
        # Collect up to `window` consecutive knots from same source
        members = []
        while i < len(l0_knots) and source_indices[i] == src and len(members) < window:
            members.append(i)
            i += 1
        l1_text = " ".join(l0_knots[j].text for j in members)
        groups.append((l1_text, members, src))
    return groups
```

## Step 3: L2 Tag Generation

Two sub-steps:

### 3a: Cluster L1 groups via Leiden

- Build similarity graph: L1 embeddings → pairwise cosine → edge if cosine > 0.5
- Run Leiden algorithm (already have `prism_rag/cluster/leiden.py`)
- Output: each L1 group assigned to a community

### 3b: LLM generates tag for each cluster

- For each cluster: collect member L1 texts (or their L0 knots)
- Prompt LLM: "Given these knowledge statements, generate a concise topic tag (2-5 words)"
- Output: L2 tag string per cluster
- Also embed the tag text → L2 fallback embedding

```python
TAG_PROMPT = """Given these related knowledge statements, generate ONE concise topic tag (2-5 words) that captures their shared theme. Return ONLY the tag, nothing else.

Statements:
{statements}

Tag:"""
```

## Step 4: Build Retrieval Indices

| Layer | Index Type | Content |
|-------|-----------|---------|
| L0 | Vector (cosine) | Individual knot embeddings |
| L1 | Vector (cosine) | Group text embeddings |
| L2 | Tag (keyword match) + Vector (cosine fallback) | Tag string + tag embedding |

## Step 5: Multi-Layer Retrieval

```python
def multi_granularity_retrieve(query, l0_vecs, l1_vecs, l2_tags, l2_vecs, 
                                l1_to_l2, l0_to_l1, embedder_fn, top_k=5):
    q_vec = embedder_fn([query])[0]
    
    # Stage 1: L2 tag matching (keyword filter)
    # Extract keywords from query, match against L2 tags
    matched_l2 = keyword_match(query, l2_tags)
    
    # If no tag match → fallback to L2 embedding search
    if not matched_l2:
        matched_l2 = cosine_top_k(q_vec, l2_vecs, k=3)
    
    # Stage 2: L1 vector search within matched L2 clusters
    candidate_l1 = [i for i, cluster in enumerate(l1_to_l2) if cluster in matched_l2]
    l1_results = cosine_top_k_filtered(q_vec, l1_vecs, candidate_l1, k=top_k)
    
    # Stage 3: Return L1 texts (with L0 members for drill-down)
    return l1_results
```

---

## Data Structures

```python
@dataclass
class MultiGranularityIndex:
    """Complete multi-layer index for retrieval."""
    # L0
    l0_texts: list[str]           # knot texts
    l0_vectors: list[list[float]] # knot embeddings
    l0_source_idx: list[int]      # which source text each L0 came from
    
    # L1
    l1_texts: list[str]           # group texts (concatenated L0s)
    l1_vectors: list[list[float]] # group embeddings
    l1_members: list[list[int]]   # L0 indices in each L1 group
    l1_source_idx: list[int]      # which source text each L1 came from
    
    # L2
    l2_tags: list[str]            # cluster tags
    l2_vectors: list[list[float]] # tag embeddings (fallback)
    l2_members: list[list[int]]   # L1 indices in each L2 cluster
    
    # Source texts (for evaluation)
    source_texts: list[str]
```

---

## Benchmark Plan

Compare 5 retrieval strategies on the same 50-text Propositionizer dataset:

| Strategy | Description |
|----------|-------------|
| `flat_l0` | L0 knots only, flat cosine search (baseline — known MRR 0.414) |
| `flat_l1` | L1 groups only, flat cosine search |
| `parent_l0` | L0 cosine search, return parent L1 text |
| `multi_layer` | L2 tag filter → L1 cosine search (the full pipeline) |
| `collapsed` | All L0+L1+L2 vectors flat-searched together (RAPTOR style) |

Metrics: Recall, MRR, IoU, Context Sufficiency, Boundary Clarity (all existing).
