# PrismRag

> Zengyn42 · Graph-First RAG System
> Builds a knowledge graph from Markdown vaults and code repositories, providing graph-traversal-based semantic retrieval via an MCP Server.

**Status**: ✅ v5.7 complete, v6.0 in progress (Pluggable Atomization & Knowledge Benchmark)

---

## Core Proposition

> **Clustering is graph-topology-based. Retrieval is graph traversal. Embedding only builds edges.**

How PrismRag differs from traditional RAG:

| Dimension | Traditional RAG | PrismRag |
|---|---|---|
| Primary storage | Vector database | NetworkX graph + JSON |
| Primary retrieval | Vector similarity search | Graph traversal (BFS / DFS) |
| Embedding role | Core query-time path | **Index-time only** (builds similarity edges) |
| Clustering | Optional / skipped | Leiden community detection (topology-driven) |
| Explainability | Vector distance numbers | EXTRACTED / INFERRED / AMBIGUOUS confidence labels |
| Incremental updates | Full re-embedding | SHA256 file cache, only re-processes changed files |

---

## Version History

| Version | Status | Highlights |
|---------|--------|-----------|
| v4.0 | ✅ | Graph-first foundation: NetworkX + Leiden + BFS/DFS + MCP Server |
| v5.0 | ✅ | Multi-namespace support, federated graph, incremental ingest |
| v5.1 | ✅ | BM25 hybrid retrieval |
| v5.2 | ✅ | Edge classifier (EXTRACTED / INFERRED / AMBIGUOUS confidence) |
| v5.3 | ✅ | `atomize_document`: LLM splits long documents into atomic knowledge fragments + inbox review |
| v5.4 | ✅ | KNOW-ID routing + label resolver (stable node IDs) |
| v5.5 | ✅ | Atomize semantic deduplication (reuse existing nodes, reduce redundancy) |
| v5.6 | ✅ | Visualization upgrade: Obsidian URI deep-links, portal cross-namespace nodes |
| **v5.7** | ✅ | **force-graph WebGL renderer** (replaces pyvis), unified code+docs graph, ego-graph focus, multi-select legend, semantic cluster naming |
| v6.0 | 🔧 | Pluggable atomization (unified Knot type, Splitter interface) + GraphRAG absorption (status field, gleanings, community reports, global_ask) + B1 benchmark |
| v7.0 | 🔜 | Federated meta-graph (global view across multiple namespaces) |

---

## v5.7 Visualization Features

`graph.html` is powered by [force-graph](https://github.com/vasturiano/force-graph) (WebGL Canvas):

- **Node focus**: click a node → show only that node + direct neighbors, dim others
- **Multi-select legend**: click a color swatch → show all nodes in that cluster; stack multiple selections
- **3-click cycle**: ×1 focus node → ×2 select node's cluster → ×3 clear
- **Semantic cluster names**: legend shows the hub node's label (e.g. `#LangGraph (36)`) instead of `doc group`
- **LOD labels**: labels fade in as you zoom — no clutter at low zoom
- **Keyboard controls**: WASD to pan, `+`/`-` to zoom, `Escape` to reset
- **On-demand edges**: `mentions_symbol` (doc→code references) hidden by default, shown on node click
- **Right-click to open Obsidian**: right-click a doc node → `obsidian://` protocol opens the original note (requires `--vault` flag)

---

## Pipeline Overview

```
Vault (.md) + Repo (.py)
       │
       ├─ Pass 1a: Markdown AST extraction
       │    └─ wikilinks / tags / frontmatter → EXTRACTED edges (confidence=1.0)
       │
       ├─ Pass 1b: Python code AST (Tree-sitter)
       │    └─ module / class / function / import → code:: nodes + call edges
       │
       ├─ Pass 2: Leiden community detection
       │    └─ topology-only clustering, identifies communities + hub nodes
       │
       ├─ Pass 3a: Embedding (Ollama bge-m3 / Gemini)
       │    └─ vectorize each node, write to Lance index
       │
       ├─ Pass 3b: Similarity edge generation
       │    └─ doc↔doc / code↔code / doc↔code semantic similarity edges
       │
       ├─ Pass 3c: Symbol links
       │    └─ code symbols mentioned in doc text → mentions_symbol edges
       │
       └─ Pass 4: Persist
            ├─ graph.json          # knowledge graph
            ├─ GRAPH_REPORT.md     # community overview
            └─ graph.html          # force-graph interactive visualization
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  MCP Server (prism-rag serve) │
                    │                              │
                    │  search_knowledge  (hybrid)  │
                    │  explain_node                │
                    │  trace_path                  │
                    │  list_communities            │
                    │  explore_community           │
                    │  atomize_document            │
                    │  inbox_review / inbox_promote│
                    │  + 11 Vault CRUD tools       │
                    └──────────────────────────────┘
                                   │
                                   ▼
              ZenithLoom agents (Hani / Asa / Jei / ...)
```

---

## Quick Start

```bash
# Install (development mode)
git clone git@github.com:Zengyn42/PrismRag.git
cd PrismRag
pip install -e ".[dev]"

# Configure (.env or environment variables)
cp .prism_env .env
# Edit .env: PRISM_VAULT_PATH, GEMINI_API_KEY (optional)
```

### Ingest a documentation vault

```bash
prism-rag ingest --vault ~/Projects/MyVault --namespace my_project
# → output to ~/Projects/MyVault/.prismrag/my_project/
```

### Ingest code + docs as a unified graph

```bash
prism-rag ingest-project --repo ~/Projects/Pulsify
# → output to ~/Projects/Pulsify/.prismrag/pulsify/
```

### View visualization

```bash
prism-rag visualize
# → opens .prismrag/<ns>/graph.html
```

### Start MCP Server

```bash
prism-rag serve              # stdio mode (Claude Code / Cursor)
prism-rag serve --transport sse  # SSE mode (network access)
```

### Atomize (split long documents into atomic knowledge fragments)

```bash
prism-rag atomize propose --node "my-note-id"
prism-rag atomize inbox          # review proposals
prism-rag atomize promote <id>   # approve and write to graph
```

---

## Directory Structure

```
PrismRag/
├── prism_rag/
│   ├── cli.py                  # CLI entrypoint (prism-rag command set)
│   ├── cli_atomize.py          # atomize subcommands
│   ├── config.py               # PrismRagSettings
│   ├── ingest/                 # ingestion pipeline
│   │   ├── vault_loader.py     # Markdown file scanning
│   │   ├── obsidian_parser.py  # Obsidian wikilinks/frontmatter
│   │   ├── ast_extractor.py    # graph node/edge extraction
│   │   ├── code_parser.py      # Tree-sitter Python parsing
│   │   ├── atomize.py          # LLM atomic document splitting
│   │   ├── embedder.py         # Ollama / Gemini vectorization
│   │   ├── similarity_linker.py# semantic similarity edges
│   │   ├── symbol_linker.py    # doc→code symbol reference edges
│   │   ├── edge_classifier.py  # edge confidence classification
│   │   ├── label_resolver.py   # KNOW-ID label resolution
│   │   ├── dedup_log.py        # deduplication log
│   │   └── incremental.py      # incremental updates
│   ├── store/                  # graph database layer
│   │   ├── graph.py            # KnowledgeGraph core class
│   │   ├── networkx_backend.py # NetworkX implementation
│   │   ├── embedding_store.py  # Lance vector index
│   │   ├── bm25_index.py       # BM25 keyword index
│   │   ├── federated.py        # multi-namespace federated graph
│   │   └── registry.py         # KNOW-ID namespace registry
│   ├── cluster/
│   │   └── leiden.py           # Leiden community detection
│   ├── retrieve/               # retrieval engine
│   │   ├── entry.py            # retrieval entrypoint
│   │   ├── hybrid.py           # hybrid retrieval (vector + BM25)
│   │   ├── bfs.py / dfs.py     # graph traversal
│   │   └── impact.py           # impact analysis
│   ├── inbox/                  # atomize review inbox
│   ├── report/
│   │   ├── visualize.py        # force-graph WebGL HTML
│   │   └── graph_report.py     # text statistics report
│   ├── vault_ops/              # vault file read/write operations
│   └── mcp_server/             # MCP Server (18 tools)
├── docs/                       # architecture docs + design plans
│   ├── ARCHITECTURE.md
│   ├── v7.0-implementation-plan.md
│   └── v7.0-design.md
├── pyproject.toml
└── README.md                   # Chinese version
└── README.en.md                # This file (English version)
```

---

## Tech Stack

| Layer | Choice |
|---|---|
| Graph storage | NetworkX + JSON |
| Community detection | `leidenalg` + `python-igraph` |
| Code parsing | `tree-sitter` |
| Markdown AST | `markdown-it-py` + `python-frontmatter` |
| Embedding | Ollama `bge-m3` (default) / Gemini Embedding |
| Vector index | LanceDB (index-time only) |
| Visualization | [force-graph](https://github.com/vasturiano/force-graph) (WebGL Canvas) |
| MCP Server | `mcp` official SDK (FastMCP) |

---

## Design Principles

1. **Graph-first** — no vector search at query time, only graph traversal
2. **Embedding as index-only** — embeddings used once at index time to build similarity edges
3. **Transparent confidence** — every edge labeled EXTRACTED / INFERRED / AMBIGUOUS
4. **Token budget** — all queries accept `--budget N` hard cap
5. **Incrementally friendly** — SHA256 file cache, only re-processes changed files
6. **Zero ops** — `pip install -e .` and go; no Neo4j / Qdrant / Docker required

---

## Related Repositories

| Repo | Role |
|---|---|
| [Zengyn42/ZenithLoom](https://github.com/Zengyn42/ZenithLoom) | Agent orchestration engine (LangGraph) |
| [Zengyn42/NimbusVault](https://github.com/Zengyn42/NimbusVault) | Obsidian knowledge base (one of PrismRag's data sources) |
| **Zengyn42/PrismRag** | **This repository** |

---

## License

Proprietary — internal use only.

---

*— Zengyn42 · Zenith Horizon*
