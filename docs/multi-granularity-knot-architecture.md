# Multi-Granularity Knot Architecture — Design Notes

> Date: 2026-07-19
> Status: Early design exploration (not converged)
> Participants: Boss + Hani
> Prerequisites: v6.0 benchmark results, AtomicRAG / RAPTOR / SiReRAG research

---

## 1. Problem Statement

Benchmark results prove a fundamental tension:

| Approach | Upstream (Gold-F1) | Downstream (MRR) | Why |
|----------|-------------------|-------------------|-----|
| Proposition (v2_propositions, L0) | **0.780** best | **0.414** worst | Too fragmented — embedding info density too low |
| Fixed window (400 char) | N/A | **0.823** best | Right granularity for embedding, but no semantic awareness |
| Parent-retrieval (proposition index → paragraph return) | 0.780 | 0.980 | Fixes MRR but sufficiency 0.700 (top-k deduplication reduces context) |
| Paragraph (passthrough) | 0.421 worst | 0.800 | Not atomic at all, but good for retrieval |

**Core contradiction**: Better atomization (upstream) = worse retrieval (downstream). Optimizing one layer hurts the other.

**Boss's solution**: Don't compromise — keep ALL granularities simultaneously, let the query decide which layer to search.

---

## 2. Proposed Architecture: 4-Layer Knot Hierarchy

```
L3: Knowledge Domain (category/tag)        ← "数据库", "网络", "算法"
L2: Topic (keywords/tags)                   ← "Redis持久化", "Redis集群"
L1: Topic Description (2-5 related facts)   ← complete description of one topic
L0: Atomic Fact (single proposition)        ← "RDB creates snapshots every 5 minutes"
```

### Key Design Decisions

#### 2.1 Layer definitions are SEMANTIC, not structural

Layer assignment is based on **content granularity**, NOT document position.

BAD: "This is H2 heading → therefore L2"
GOOD: "This describes a complete topic with 3 related facts → therefore L1"

This ensures the same knowledge is at the same layer regardless of which document it comes from. A well-structured document and a poorly-structured one produce the same layer assignments.

#### 2.2 Intra-layer relationships: existing graph edges

Within each layer, knots connect via existing PrismRag edge types:
- L0: `NEXT` (sequential), `SIMILARITY` (embedding cosine)
- L1: `SIMILARITY`, `MENTIONS` (shared entities)
- L2: tag co-occurrence, entity overlap
- L3: high-level ontology relations

#### 2.3 Inter-layer relationships: CONTAINS tree

```
L3(数据库) -[CONTAINS]-> L2(Redis持久化) -[CONTAINS]-> L1(RDB机制描述) -[CONTAINS]-> L0(RDB每5分钟快照)
```

The CONTAINS tree is derived from document structure when available, or from semantic clustering when structure is absent.

---

## 3. Open Questions (Recorded 2026-07-19)

### Q1: High layers use tags/keywords, not embedding — is this right?

**Boss's proposal**: L2/L3 are tags and category labels, not text embeddings.

**Arguments FOR tags/keywords**:
- Tags are discrete, precise, and interpretable — "Redis持久化" means exactly that
- Tag matching is exact (BM25/keyword) — no embedding space distortion
- Tags are naturally hierarchical — taxonomy/ontology structure
- Avoids the problem of long-text embedding quality degradation (L2/L3 text too long for embedding model's optimal window)
- Storage efficient — a few tags vs a 768-dim vector

**Arguments AGAINST (or rather, considerations)**:
- Tags miss semantic similarity — "Redis persistence" won't match tag "Redis持久化" without a synonym layer
- Tag vocabulary needs management — who defines allowed tags? LLM may generate inconsistent tags
- Hybrid may be best — tags for filtering/routing + embedding for ranking within filtered set

**Hani's current view**: Boss is probably right for L3 (pure categories), partially right for L2 (tags + lightweight embedding). L2 might need both:
- Tags for **routing** (which topic cluster to search)
- Embedding for **ranking** (which L1/L0 within that cluster is most relevant)

**Implication for retrieval**: If L2/L3 are tags, then RAPTOR's collapsed tree (all layers in one vector index) does NOT apply. The retrieval path becomes:

```
Query → LLM extracts keywords + classifies category
     → L3 category match (classifier/exact match)
     → L2 keyword/tag match (BM25 or tag intersection)
     → L1/L0 vector search (cosine, within narrowed scope)
```

This is closer to GraphRAG's community-based routing than RAPTOR's flat vector search.

### Q2: How to quantify granularity uniformly?

The problem: "Is this knot L0 or L1?" needs an operational rule, not LLM subjective judgment.

**No known paper directly addresses this.** Adjacent work:
- Dense X defines "proposition" qualitatively (atomic, self-contained) but no formal granularity metric
- RAPTOR uses cluster size as implicit granularity (leaf=fine, root=coarse) but no cross-document normalization
- MoC (ACL 2025) defines "boundary clarity" and "chunk stickiness" but these measure chunking quality, not granularity level
- Topic segmentation literature (TextTiling, C99) segments by topic shifts but doesn't assign granularity levels

**Possible quantitative criteria** (to be researched/designed):
- **Word count bands**: L0 (10-30 words), L1 (50-150 words), L2 (tags, <10 words), L3 (single label)
- **Entity count**: L0 mentions 1-2 entities, L1 mentions 3-5, L2 summarizes a cluster of entities
- **Information density**: L0 = one fact, L1 = one topic (multiple facts about same subject), L2 = one domain (multiple topics sharing a theme)
- **Composability test**: If a knot can be split into 2+ independently meaningful statements → it's at least L1, not L0

**Action needed**: Research or design a formal granularity metric. This is possibly a novel contribution — no existing paper provides one.

### Q3: Where do L2 tags come from?

**Boss's view**: L2 tags should emerge from L0/L1 clustering (bottom-up), not from document structure.

**Three possible sources**:

| Source | Method | Pros | Cons |
|--------|--------|------|------|
| **Bottom-up clustering** (Boss preference) | Cluster L0/L1 knots by entity/embedding → LLM generates tag for each cluster | Data-driven, cross-document consistent | Depends on clustering quality |
| **Leiden community labels** (already have) | Leiden clusters on the knot graph → god-node labels | Zero additional cost, already implemented | Labels are derived from node names (e.g., "Node A · Node B"), not semantically meaningful tags |
| **LLM extraction at ingest time** | During atomization, LLM also assigns topic tags to each knot | Simple, inline with existing pipeline | Per-knot tags may be inconsistent across documents |

**Hani's recommendation**: Bottom-up clustering is the right approach. Two-step:
1. Cluster L0/L1 knots (by shared entities or embedding similarity)
2. LLM generates a concise tag/label for each cluster

This is essentially what GraphRAG community reports do, but producing a tag instead of a full report.

**Leiden community label accuracy**: Currently LOW for our purposes. Leiden labels are concatenated god-node labels like "Redis RDB · AOF Configuration · fsync policy" — machine-generated from node names, not semantically curated. They work as identifiers but not as meaningful topic tags. The v6.0 `generate_community_reports` tool produces better labels (LLM-generated titles like "Redis Persistence Mechanisms") — these would be more suitable as L2 tags.

---

## 4. Relationship to Existing Research

| Paper/System | Relevance to this architecture |
|-------------|-------------------------------|
| **RAPTOR** (ICLR 2024) | Hierarchical tree from leaf chunks. Their "collapsed tree" (all layers flat-searched) works when all layers are embeddings. Does NOT apply if L2/L3 are tags. |
| **AtomicRAG** (Apr 2026) | Entity-based dynamic grouping at query time. Their Atom-Entity Graph is relevant for L0↔L2 connections (knots linked via shared entities = tags). |
| **SiReRAG** (Dec 2024) | Dual tree: similarity + relatedness (entity co-occurrence). Relatedness tree is closest to Boss's L1 concept. |
| **GraphRAG** (Microsoft) | Community hierarchy = L2/L3. Community reports = L2 tag generation. Map-reduce global_ask = L3 routing. Already partially implemented in PrismRag. |
| **Anthropic Contextual Retrieval** | Enriching each chunk with context prefix. Could apply to L0/L1 to improve embedding quality without changing structure. |
| **Six-Space Theory** (王延章) | K-space Nm/Am/Rm maps to Knot name/attributes/relations. Am attributes (maturity/confidence/actionability) can filter within layers. Rm dimension directly corresponds to inter/intra-layer edges. |

---

## 5. Knowledge Update in Multi-Granularity System

### Update flow: bottom-up bubble

```
Document edited
    ↓
Step 1: Re-atomize affected sections → new L0 knots
    ↓
Step 2: L0 change detection (Mem0-style LLM Judge)
    - REUSE: no change needed
    - UPDATE: old L0 → superseded, new L0 created
    - SUPERSEDE: contradiction detected, old replaced
    - NEW: genuinely new knowledge
    ↓
Step 3: Affected L1 knots → re-aggregate member L0s → re-embed
    ↓  
Step 4: If L1 changed significantly (>30% members changed) → L2 tag may need update
    ↓
Step 5: L3 categories are stable (rarely change)
```

### Granularity inconsistency problem (Boss identified)

If layers are defined by document structure, the same knowledge may be L1 in one doc and L2 in another. Solution: layers are defined by **semantic granularity** (see Q2), not by document position. The atomization step assigns layers based on content, normalized across all documents.

### Cross-document dedup at each layer

| Layer | Dedup signal | Action |
|-------|-------------|--------|
| L0 | Embedding cosine > 0.95 or exact text match | REUSE (link to existing) |
| L1 | >50% member L0 knots overlap with existing L1 | MERGE or REUSE |
| L2 | Same tag/keyword | MERGE tag clusters |
| L3 | Same category | No action needed (categories are global) |

---

## 6. Retrieval Strategy (Not Converged)

Two candidates:

### Option A: Tag routing + scoped vector search (Boss's intuition)
```
Query → LLM extracts keywords + category
     → L3 filter (category match)
     → L2 filter (tag/keyword match)  
     → L1/L0 vector search (cosine, within filtered scope)
```
Pro: Efficient (search space reduced before vector search)
Con: LLM routing errors → wrong scope → miss results

### Option B: Collapsed tree (RAPTOR style)
```
All layers in one flat vector index → cosine search → top-k
```
Pro: No routing errors, query naturally finds right granularity
Con: Only works if all layers are embeddings (breaks if L2/L3 are tags)

### Option C: Hybrid (Hani's suggestion)
```
L0/L1: vector index (cosine search)
L2: tag index (BM25 / exact match)
L3: classifier
Merge results from vector search + tag match → re-rank
```
Pro: Best of both worlds
Con: More complex to implement and tune

**Not decided yet. Needs benchmark validation.**

---

## 7. Next Steps (When Resumed)

1. Research formal granularity metrics (Q2) — is there any paper?
2. Prototype L2 tag generation from L0/L1 clustering
3. Benchmark: multi-layer retrieval vs flat vs parent-retrieval
4. Design the update pipeline with cross-layer consistency
5. Debate subgraph: converge on retrieval strategy (A vs B vs C)

---

*— Hani · 无垠智穹*
