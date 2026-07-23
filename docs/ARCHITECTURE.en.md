# PrismRag v5.7 Architecture

> Historical version: [v4.0 Architecture](archive/ARCHITECTURE-v4.0.en.md)

---

## Core Proposition

> **Clustering is graph-topology-based. Retrieval is graph traversal. Embedding only builds edges.**

Key differences from traditional RAG:

| Dimension | Traditional RAG | PrismRag v5.7 |
|---|---|---|
| Primary storage | Vector database | NetworkX graph + JSON |
| Primary retrieval | Vector similarity search (query-time) | Graph traversal BFS / DFS (query-time) |
| Embedding role | Core query-time path | **Index-time only** (builds similarity edges) |
| Clustering | Optional / skipped | Leiden community detection (topology-only, no LLM) |
| Explainability | Vector distance numbers | EXTRACTED / INFERRED / AMBIGUOUS confidence labels |
| Incremental updates | Full rebuild | SHA256 file cache, only re-processes changed files |
| Data sources | Single document set | Markdown vault + Python codebase (unified graph) |

---

## Input Types

| Type | Command | Description |
|---|---|---|
| Markdown vault (Obsidian) | `prism-rag ingest` | Document-only graph |
| Python codebase | `prism-rag ingest-code` | Code-only graph |
| Code + docs unified graph | `prism-rag ingest-project` | code + docs, bridged via symbol links |

Output convention: `<target>/.prismrag/<namespace>/`

---

## Pipeline Overview (7 Passes)

```
Vault (.md) + Repo (.py)
       │
       ├─ Pass 1a: Markdown AST extraction
       │
       ├─ Pass 1b: Python code AST (Tree-sitter)
       │
       ├─ Pass 2: Leiden community detection
       │
       ├─ Pass 3a: Embedding (Ollama / Gemini)
       │
       ├─ Pass 3b: Similarity edge generation
       │
       ├─ Pass 3c: Symbol links (doc→code)
       │
       └─ Pass 4: Persist + visualize
```

---

## Pass 1a: Markdown AST Extraction (Deterministic, Zero LLM)

**Input**: `.md` files under the vault

**Processing**:
- `python-frontmatter` parses YAML frontmatter → metadata
- `markdown-it-py` parses markdown into AST
- Extracts structured signals:

| Signal | Edge type | Confidence |
|---|---|---|
| `[[wikilink]]` | `links_to` | EXTRACTED (1.0) |
| `[[note#heading]]` | `links_to_section` | EXTRACTED (1.0) |
| `#tag` | `tagged_as` | EXTRACTED (1.0) |
| frontmatter `aliases:` | `aliased_as` | EXTRACTED (1.0) |
| frontmatter `category:` | `categorized_as` | EXTRACTED (1.0) |

**Output**: All **EXTRACTED** edges (`confidence_score = 1.0`), zero LLM cost.

**KNOW-ID routing (v5.4+)**: Nodes with `knowledge_id` in frontmatter use stable IDs. Label resolved via three-layer fallback: `title` → `clean_slug` → stem.

---

## Pass 1b: Python Code AST (Tree-sitter)

**Input**: `.py` files under the repo

**Processing**:
- `tree-sitter` parses Python AST
- Extracts code structures:

| Structure | Node prefix | Example |
|---|---|---|
| Module | `code::module::` | `code::module::prism_rag.cli` |
| Class | `code::class::` | `code::class::KnowledgeGraph` |
| Function | `code::func::` | `code::func::search_knowledge` |
| Import | — | generates `imports` edge |

- `defines` edges: module → class/function
- `calls` edges: caller → callee (static call graph)
- `imports` edges: module → imported symbol

**Output**: `code::` nodes + call graph edges, all EXTRACTED.

---

## Pass 2: Leiden Community Detection

**Input**: Complete graph from Pass 1 (doc nodes + code nodes)

**Processing**:
- `python-igraph` converts NetworkX to igraph representation
- `leidenalg.find_partition(ModularityVertexPartition)` runs community detection
- Edge weight = `confidence_score × weight` (EXTRACTED edges have highest weight)
- Each community identifies a **hub node** (highest-degree node in the community)
- Hub node label becomes the community name (used in visualization legend, e.g. `#LangGraph (36)`)

**Output**: Each node tagged with `community_id`; community metadata (hub node, member count)

---

## Pass 3a: Embedding (INDEX-TIME ONLY)

**Input**: All text nodes

**Model options**:

| Model | Backend | Dimensions | Use case |
|---|---|---|---|
| `bge-m3` | Ollama (local) | 1024 | Default, bilingual |
| `qwen3-embedding:8b` | Ollama (local) | 1024 | Chinese-heavy content |
| `text-embedding-004` | Gemini API | 768 | Cloud fallback |

**Processing**:
1. Compute vector for each node
2. Write to LanceDB (`embed_cache.lance`) as cache only
3. SHA256 content hash prevents redundant recomputation

**Key constraint**: Vectors are **NOT used at query time** (only used in Pass 3b to build edges).

---

## Pass 3b: Similarity Edge Generation

**Input**: LanceDB vector cache

**Processing**:
- Global top-K ANN search (default K=10)
- Generate `semantically_similar_to` edges for each pair
- Confidence rules:
  - `confidence = INFERRED`
  - `confidence_score = cosine similarity`
  - Pairs below threshold (default 0.5) are skipped

**Cross-type edges**:
- doc ↔ doc
- code ↔ code
- doc ↔ code (requires `--cross-modal`)

---

## Pass 3c: Symbol Links (doc → code)

**Input**: Doc node text + code symbol set

**Processing**:
- Scans doc node content for code symbol names (exact string match)
- Generates `mentions_symbol` edges (doc → code)
- Confidence = EXTRACTED

**Visualization behavior**: `mentions_symbol` edges are hidden by default; shown on node click (avoids cluttering the graph).

---

## Pass 4: Persist + Visualize

**Output files** (stored in `.prismrag/<namespace>/`):

| File | Description |
|---|---|
| `graph.json` | Complete knowledge graph (nodes + edges + communities + metadata) |
| `GRAPH_REPORT.md` | Text statistics report (community overview, hub nodes, edge stats) |
| `graph.html` | force-graph WebGL interactive visualization |
| `embed_cache.lance/` | LanceDB vector cache |
| `bm25_index/` | BM25 keyword index |

### graph.html Visualization Features (v5.7)

Powered by [force-graph](https://github.com/vasturiano/force-graph) (WebGL Canvas):

| Feature | Description |
|---|---|
| Node focus | Click node → show only that node + direct neighbors |
| Multi-select legend | Click color swatch → show all nodes in cluster; stackable |
| 3-click cycle | ×1 focus node → ×2 select node's cluster → ×3 clear |
| Semantic cluster names | Legend shows hub node label (e.g. `#LangGraph (36)`) |
| LOD labels | Labels fade in as you zoom in — no clutter at low zoom |
| Keyboard controls | WASD to pan, `+`/`-` to zoom, Escape to reset |
| On-demand edges | `mentions_symbol` hidden by default, shown on node click |
| Right-click Obsidian | Right-click doc node → `obsidian://` opens original note |

---

## Incremental Updates

- Each file gets a SHA256 hash stored in `file_hashes.json`
- Subsequent ingest runs only reprocess files whose hash changed
- Embedding cache is keyed by `(node_id, content_hash)` — unchanged nodes reuse existing vectors
- Leiden reruns fully after graph changes (incremental Leiden is future work)

---

## Query Time (Zero Embedding)

```
User query
    │
    ▼
Entry node resolution
    ├─ label / alias exact match
    ├─ BM25 keyword search
    └─ ANN vector search (fallback, top-1 only)
    │
    ▼
Graph traversal
    ├─ BFS (default — broad context)
    ├─ DFS (single-chain depth)
    └─ path(a→b) (shortest path)
    │
    ▼
Token budget pruning (--budget N)
    │
    ▼
Return: [nodes], [edges], community_info
```

**Key properties**:
- **No vector search** at query time (except entry point fallback, top-1 only)
- **No re-ranking**
- **Deterministic output**: same query always returns same result
- **Hard budget cap**: `--budget 4000` returns at most 4000 tokens of node content

---

## Atomize Pipeline (v5.3+)

For notes that contain multiple independent knowledge points:

```
prism-rag atomize propose --node <id>
    │
    ▼ LLM (Gemini / Claude) splits into atomic proposals
    │
    ▼ Proposals written to inbox for review
    │
prism-rag atomize inbox          # human review
    │
prism-rag atomize promote <id>   # approve → write to graph
```

- Atomic nodes use `knowledge_id` frontmatter as stable ID
- Semantic deduplication (v5.5): checks for equivalent existing nodes before generating, avoids redundancy

---

## MCP Server (30 tools)

Started with `prism-rag serve`. Supports both stdio and SSE transports. See `docs/MCP_TOOLS.md` for full reference.

### Graph Query Tools (7)

| Tool | Purpose |
|---|---|
| `search_knowledge` | Main retrieval (hybrid: BM25 + embedding + exact, BFS/DFS traversal) |
| `explain_node` | Node details + all edges + community info |
| `trace_path` | Shortest path between two nodes (incl. cross-namespace) |
| `communities` | Community list / member details (merged list/explore) |
| `impact` | Change impact analysis (blast radius) |
| `list_namespaces` | Federated graph namespace stats |
| `generate_graph` | Generate interactive HTML visualization |

### Knowledge Atomization Tools (5)

| Tool | Purpose |
|---|---|
| `atomize_scan` | Scan document structure (step 1 of scan/propose/apply) |
| `atomize_propose` | Submit atomization claims (with semantic dedup) |
| `atomize_apply` | Execute proposal, create knowledge/*.md files |
| `alloc_knowledge_id` | Allocate globally unique KNOW-IDs |
| `list_knowledge_nodes` | List knowledge nodes in the graph |

### Edge Management + Drift Detection Tools (6)

| Tool | Purpose |
|---|---|
| `pending_edges` | List / inspect pending cross-namespace edges |
| `review_pending_edge` | Approve or reject a pending edge |
| `check_drift` | Detect stale mentions_symbol edges |
| `flag_drift` | Auto-flag affected KNOTs as suspected |
| `rollback_dedup` | Roll back graph effects of a dedup decision |
| `list_dedup_log` | View dedup decision log |

### Community Intelligence Tools (2)

| Tool | Purpose |
|---|---|
| `generate_community_reports` | Generate LLM reports for Leiden communities |
| `global_ask` | Map-reduce cross-community Q&A |

### Vault CRUD Tools (10)

Read, create, update, delete notes in an Obsidian vault; supports frontmatter operations, tag management, keyword search, wikilink queries, etc.

---

## Data Storage Layout

```
<target>/.prismrag/<namespace>/
├── graph.json            # complete knowledge graph
├── GRAPH_REPORT.md       # statistics report
├── graph.html            # interactive visualization
├── file_hashes.json      # SHA256 incremental cache
├── embed_cache.lance/    # LanceDB vector cache
└── bm25_index/           # BM25 index
```

Each target (vault or repo) has its own isolated `.prismrag/` directory.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Graph storage | NetworkX + JSON |
| Community detection | `leidenalg` + `python-igraph` |
| Code parsing | `tree-sitter` |
| Markdown AST | `markdown-it-py` + `python-frontmatter` |
| Embedding | Ollama `bge-m3` / `qwen3-embedding:8b` (default); Gemini API (fallback) |
| Vector cache | LanceDB |
| Keyword index | BM25 (`rank_bm25`) |
| Visualization | [force-graph](https://github.com/vasturiano/force-graph) (WebGL Canvas) |
| MCP Server | `mcp` official SDK (FastMCP) |
| Configuration | `pydantic-settings` (.env / environment variables) |

---

## v6.0 Additions: Pluggable Atomization & Knowledge Benchmarks

### Splitter Framework (`prism_rag/ingest/splitters/`)

Pluggable document splitting interface where all methods output `list[Knot]`. 7 built-in strategies: `sentence`, `llm` (versioned prompts), `llm_gleanings` (GraphRAG gleaning rounds), `fixed_window`, etc. LLM backend is injectable (Ollama / Claude CLI).

### Knot Lifecycle

KNOT (Knowledge Ontology Token) nodes now carry a `status` field:

| Status | Meaning |
|--------|---------|
| `confirmed` | Verified knowledge (default) |
| `suspected` | Associated code has changed; needs human review |
| `superseded` | Replaced by a newer version of the knowledge |

The `flag_drift` MCP tool automatically marks KNOTs as `suspected` when their linked code symbols no longer exist.

### Benchmark Harness (`prism_rag/ingest/splitters/benchmark/`)

- **Upstream evaluation**: Based on Propositionizer-wiki-data (42k gold propositions), 5-dimension scoring (atomicity, self-containedness, faithfulness, coverage, gold-alignment F1)
- **Downstream evaluation**: 6 metrics (recall, MRR, IoU, context_sufficiency, boundary_clarity)
- 5 prompt strategy comparison; `v2_propositions` (Dense X style) performs best

### Community Reports + global_ask

- `generate_community_reports` MCP tool: generates LLM community reports (title/summary/rating/findings) for all Leiden communities, cached to `community_reports.json`
- `global_ask` MCP tool: map-reduce cross-community Q&A, synthesizes answers from all relevant communities

### Multi-Granularity Knot Architecture (Evolving Design)

Three-layer retrieval architecture (still under development):

| Layer | Meaning | Retrieval Role |
|-------|---------|----------------|
| L0 | Atomic proposition (Knot) | Storage / reasoning unit |
| L1 | Group of 2-4 adjacent knots | Optimal retrieval unit (MRR 0.927) |
| L2 | Leiden cluster + LLM-generated tag | Routing / filtering layer |

See `docs/multi-granularity-knot-architecture.md` and `docs/multi-granularity-algorithm.md` for details.

---

## v7.0 Outlook: Federated Meta-Graph

> See [v7.0-design.en.md](v7.0-design.en.md) and [v7.0-implementation-plan.en.md](v7.0-implementation-plan.en.md)

Core goal: unified global view across multiple namespaces (different vaults / repos).

- **manifest.json**: metadata registry per namespace
- **FederatedGraph**: runtime federation over multiple KnowledgeGraph instances
- **Cross-namespace bridge edges**: shared tag + import (deterministic) → hub node ANN (semantic)
- **Federated visualization**: namespace super-node meta-graph + drill-down to single namespace

---

*— Zengyn42 · Zenith Horizon*
