# GraphRAG vs PrismRag — Comparison & Borrowable Mechanisms

> Date: 2026-07-18
> Source basis: microsoft/graphrag source code (shallow clone at `Resource/graphrag`,
> new modular layout `packages/graphrag/graphrag/` + `graphrag-llm`, `graphrag-chunking`),
> analyzed module-by-module — not README claims.

## 1. How GraphRAG Organizes Knowledge Sources

Fully automated "mechanical chunking → entity-centric reorganization" pipeline
(verified from `index/workflows/`):

```
Raw documents (txt/csv/json)
  │ load_input_documents
  ▼
TextUnit chunking — pure token-count split (~1200 tokens + overlap), no semantic boundaries
  │ create_base_text_units        ← provenance anchor: all downstream artifacts carry text_unit_ids
  ▼
LLM entity/relationship extraction — per chunk: (entity, type, description) + (src, dst, desc, strength)
  │ extract_graph + gleanings loop
  ▼
Cross-chunk entity merge by name — same entity in N chunks → N accumulated descriptions
  │ summarize_descriptions: LLM merges into one, "resolve contradictions"
  ▼
Claims extraction (optional) — (subject, object, type, status, temporal bounds, source_quotes)
  │ extract_covariates
  ▼
Leiden hierarchical clustering — entity graph → multi-level communities (L0 coarse → L1 fine)
  │ create_communities
  ▼
Community reports — LLM writes per-community: title + summary + rating + cited findings
  │ create_community_reports
  ▼
Storage: parquet tables (documents / text_units / entities / relationships /
         communities / community_reports / covariates) + LanceDB vector store
```

## 2. Fundamental Philosophical Difference

| Dimension | GraphRAG | PrismRag |
|---|---|---|
| Organizing center | **Entities** (nouns auto-merged across chunks) | **KNOTs** (self-contained knowledge atoms) |
| Split criterion | Token count (mechanical) | Semantic atomicity (LLM-judged) |
| Unit of knowledge | entity + auto-merged description | human-reviewed title + body + ontology_type |
| Human role | None — fully automated, no correction loop | Inbox review is a mandatory gate |
| Storage form | Parquet tables (not human-editable) | Obsidian markdown (directly human-editable) |
| Provenance | text_unit_ids (points to mechanical chunks) | KNOW-ID + section provenance (points to semantic locations) |
| Assumed corpus | Static, one-shot ingestion | Living, continuously human-revised |

**One-liner: GraphRAG turns documents into a database; PrismRag turns documents
into a maintainable knowledge base.**

## 3. Where PrismRag Is Already Ahead

- **Semantic dedup** — ours: 0.90 cosine over embeddings; theirs: exact-title match only.
- **Human-in-the-loop curation** — they have no review/correction path at all;
  our inbox review + Obsidian editability is the audit trail they lack.
- **Code↔doc drift checking** — no equivalent in GraphRAG. Unique moat.
- **Hybrid search** — theirs is plain LanceDB vector lookup.
- **Evaluation** — they ship only smoke/integration tests (`tests/smoke`),
  no benchmark harness. Confirms our benchmark plan is real differentiation.

## 4. Borrowable Mechanisms (ranked)

### 4.1 Community reports + map-reduce global search — BIGGEST GAP
- Refs: `prompts/index/community_report.py`,
  `query/structured_search/global_search/search.py`
- We have Leiden clusters but no LLM-written per-community reports. Their report
  schema (title, summary, rating 0–10 + explanation, 5–10 findings each with
  grounded `[Data: Entities (ids)]` citations) turns clusters into queryable
  artifacts. Global search maps the question over all reports in parallel
  (asyncio semaphore, 32 concurrent), each map emits
  `{points: [{description, score 0-100}]}`, reduce sorts by score and synthesizes.
- PrismRag today cannot answer "what are the main themes of this vault."
- Cost: one indexing step + one new MCP tool (`global_ask`).

### 4.2 Claims schema → Knot fields — LOW COST, HIGH LEVERAGE
- Ref: `prompts/index/extract_claims.py`
- Their tuple: `(subject, object|NONE, claim_type, status TRUE/FALSE/SUSPECTED,
  start_date, end_date, description, source_quotes)`.
- Borrow into `Knot`:
  - **`status` field** (confirmed / suspected / superseded) — pairs directly with
    our drift-checker: when code changes, mark linked KNOTs `suspected`.
  - **Prompt trick**: "name the claim_type so it can be repeated across multiple
    text inputs" — one line that stabilizes the ontology_type vocabulary across
    atomize runs.
- Also validates our "text is canonical, structure is projection" position:
  their claims too keep a text description as the body with structured fields
  around it.

### 4.3 Gleanings loop — RECALL BOOSTER FOR LlmSplitter
- Refs: `index/operations/extract_graph/graph_extractor.py:85-122`,
  `prompts/index/extract_graph.py`
- After first extraction, re-prompt in the same conversation: "MANY entities
  were missed... Add them below" (CONTINUE_PROMPT), then a forced single-letter
  Y/N gate (LOOP_PROMPT), up to `max_gleanings` rounds.
- Drops straight into our Splitter interface as a wrapper around any LLM-based
  method; existing 0.90 dedup absorbs repeats. Registerable as its own splitter
  (`llm_gleanings`) → directly comparable in the B1 benchmark.
- Side note: their delimiter protocol (`<|>`, `##`, `<|COMPLETE|>`) is more
  robust than JSON for long extraction outputs from small local models.

### 4.4 Incremental update: delta + selective recompute
- Refs: `index/update/incremental_index.py` (title-based `InputDelta` new/deleted
  diff), `update/entities.py:_group_and_resolve_entities` (groupby title, keep
  first id, append descriptions, re-summarize, remap delta ids → old ids),
  `update_communities.py` (only communities containing touched entities get
  reports regenerated).
- Our KNOW-ID + content-hash setup can adopt the same
  "join → id remap → selective community re-report" pattern once we have
  community reports (4.1).

### 4.5 Description summarization on merge
- Ref: `prompts/index/summarize_descriptions.py`
- When one entity accumulates N descriptions, LLM-merge into one and "resolve
  contradictions." Complements our dedup: 0.90-similar pairs become
  merge candidates instead of drop-or-keep.

## 5. Explicitly Skipped

| Mechanism | Why skip |
|---|---|
| DRIFT search (`drift_search/`) | Primer + follow-up-query tree over local searches — expensive multi-hop machinery; hybrid search + KNOW-ID routing covers vault-scale needs |
| `prompt_tune/` auto prompt tuning | Generates domain/persona-adapted prompts from sample docs; overkill for a personal/team vault |
| Their evaluation setup | Nothing shippable — smoke tests only, no benchmark harness |
| NLP noun-graph fast path (`build_noun_graph/`) | Noun-phrase co-occurrence graph, English-centric, strictly lower quality than LLM atomization |

## 6. Suggested Adoption Order

1. **Community reports + `global_ask`** — fills the biggest capability gap
   (vault-level thematic questions).
2. **`status` field on Knot** — near-zero cost, links drift-checker to atom
   lifecycle (confirmed → suspected → superseded).
3. **Gleanings wrapper splitter** — enters the B1 benchmark as a candidate
   method; measurable before committing.
4. Incremental selective recompute — after (1) exists.
5. Merge-on-dedup — after benchmark tells us how often 0.90 pairs are
   true merges vs distinct atoms.
