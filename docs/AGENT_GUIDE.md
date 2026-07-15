# PrismRag — Agent Integration Guide

How to use PrismRag as a knowledge retrieval backend. Written for AI agents (Claude, Gemini, Ollama-based) that need to query codebases or Obsidian vaults.

---

## What PrismRag Does

PrismRag turns source code and markdown notes into a **queryable knowledge graph**. You give it a codebase or vault; it parses the structure (AST for code, wikilinks for notes), clusters related nodes via Leiden community detection, embeds everything into vectors, and exposes the graph through CLI or MCP tools.

**Retrieval pipeline** (what happens when you query):

```
Your question
  → 3-way parallel search:
      ① Exact match    — substring on node ID / label
      ② BM25           — keyword relevance (TF-IDF variant)
      ③ Embedding ANN  — semantic similarity (cosine in vector space)
  → Reciprocal Rank Fusion (k=60) — merges 3 ranked lists into one
  → Top-K seed nodes selected
  → BFS/DFS graph traversal from seeds (token-budget bounded)
  → Formatted context returned to you
```

---

## Two Ways to Use PrismRag

### Option A: MCP Server (Recommended for Agents)

Start the server, then call tools via MCP protocol.

```bash
# Start (stdio transport — for Claude Code / agent SDK)
prism-rag serve

# Start (SSE transport — for network access)
prism-rag serve --transport sse --port 8102
```

#### Core Query Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `search_knowledge` | Graph traversal from a query | **Primary tool.** Ask any question about the codebase/vault |
| `explain_node` | Full details of one node + neighbors | You know the exact symbol/file name |
| `trace_path` | Shortest path between two nodes | "How does A relate to B?" |
| `list_communities` | Overview of all Leiden clusters | Understand codebase structure |
| `explore_community` | Drill into one community | Deep-dive a module/subsystem |
| `list_namespaces` | Show all indexed graphs | Check what's available |
| `check_drift` | Detect stale cross-references | Maintenance |

#### search_knowledge — The Main Tool

```json
{
  "tool": "search_knowledge",
  "arguments": {
    "query": "How does hybrid search work?",
    "scope": "code",
    "mode": "bfs",
    "budget": 4000
  }
}
```

Parameters:
- `query` (required): Natural language or keyword. Works in English and Chinese.
- `scope`: Namespace to search (`"code"`, `"nimbus"`, `"pulsify"`, or `""` for all).
- `mode`: `"bfs"` (broad context, default) or `"dfs"` (trace a specific path).
- `budget`: Max tokens in response (default 4000). Lower = faster, less context.

#### explain_node

```json
{
  "tool": "explain_node",
  "arguments": {
    "node": "hybrid_search",
    "scope": "code"
  }
}
```

Returns: node label, type, source file, source location, community membership, all edges (incoming + outgoing), and node content.

#### trace_path

```json
{
  "tool": "trace_path",
  "arguments": {
    "from_node": "hybrid_search",
    "to_node": "BM25Index",
    "max_length": 5
  }
}
```

Returns: shortest relationship chain between two nodes.

### Option B: CLI (Quick Queries / Scripting)

```bash
# Query a graph
prism-rag query "How does BFS traversal work?" -s code -b 4000

# Query with DFS mode
prism-rag query "embedding store nearest neighbor" -s code -m dfs

# Show node content
prism-rag query "hybrid_search" -s code --content
```

---

## Setting Up a New Codebase

### Ingest Source Code

```bash
# Basic: parse repo → build graph + clusters + embeddings
prism-rag ingest-code --repo /path/to/your/repo

# With custom output dir and namespace
prism-rag ingest-code \
  --repo /path/to/your/repo \
  --data-dir /path/to/output \
  --namespace myproject

# Skip slow steps if you just want the graph structure
prism-rag ingest-code --repo /path/to/repo --skip-embed --skip-cluster
```

**What ingest produces** (in data-dir):
- `graph.json` — the knowledge graph (nodes + edges + communities)
- `bm25.json` — BM25 inverted index
- `embeddings.lance/` — LanceDB vector store
- `embed_cache.json` — embedding cache for incremental updates

### Ingest an Obsidian Vault

```bash
prism-rag ingest --vault /path/to/vault --output /path/to/data
```

### Multi-Namespace Federation

PrismRag can serve multiple graphs simultaneously. Configure via `PRISM_GRAPHS` env var:

```bash
export PRISM_GRAPHS='[
  {"namespace":"code", "vault_path":"/path/to/repo", "data_dir":"/path/to/data/code", "writable":false},
  {"namespace":"docs", "vault_path":"/path/to/vault", "data_dir":"/path/to/data/docs"}
]'
prism-rag serve
```

Each namespace is independently queryable via the `scope` parameter.

---

## Query Strategy Guide

### Best Practices

1. **Start with `search_knowledge`** — it handles most queries. Use `explain_node` only when you already know the exact symbol.

2. **Use `scope` to narrow search** — querying all namespaces is slower and may return cross-domain noise.

3. **BFS vs DFS**:
   - `bfs` (default): broad context, good for "what is X" or "how does X work"
   - `dfs`: follows one path deeply, good for "trace the call chain from X to Y"

4. **Budget tuning**: default 4000 tokens. Use 2000 for quick lookups, 8000 for deep analysis.

5. **Chinese queries work** — the embedding layer handles cross-lingual semantics. "混合检索" will find `hybrid_search`.

### Query Examples

| Goal | Query |
|------|-------|
| Understand a function | `"How does hybrid_search work?"` |
| Find implementation of a concept | `"Where is Leiden community detection implemented?"` |
| Trace dependencies | `"What does BM25Index depend on?"` (+ DFS mode) |
| Cross-language lookup | `"知识图谱的构建流程"` |
| Explore structure | `list_communities` → pick a community → `explore_community` |

---

## Node ID Format

Understanding node IDs helps with `explain_node` and `trace_path`:

- **Code nodes**: `code::relative/path/to/file.py::SymbolName`
  - Example: `code::prism_rag/retrieve/hybrid.py::hybrid_search`
- **Vault nodes**: `relative/path/to/note` (no extension)
  - Example: `projects/architecture-overview`
- **KNOW nodes** (atomized knowledge): `KNOW-{6digit}` — **Atomize 功能当前已暂停，不可用**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  Ingest Pipeline                 │
│                                                  │
│  Source Code ──→ Tree-sitter AST ──→ Nodes/Edges │
│  Vault Notes ──→ Wikilink Parse ──→ Nodes/Edges  │
│         │                                │       │
│         ▼                                ▼       │
│   Similarity Linker ──→ INFERRED edges           │
│         │                                        │
│         ▼                                        │
│   Leiden Clustering ──→ Community assignments     │
│         │                                        │
│         ▼                                        │
│   Embedder (Ollama) ──→ LanceDB vectors          │
│         │                                        │
│         ▼                                        │
│   graph.json + bm25.json + embeddings.lance/     │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│                 Query Pipeline                   │
│                                                  │
│  Query ──→ ① Exact Match (substring on labels)   │
│        ──→ ② BM25 (keyword relevance)            │
│        ──→ ③ Embedding ANN (semantic similarity)  │
│                    │                              │
│                    ▼                              │
│         Reciprocal Rank Fusion (RRF)             │
│                    │                              │
│                    ▼                              │
│            Top-K Seed Nodes                      │
│                    │                              │
│                    ▼                              │
│         BFS/DFS Graph Traversal                  │
│         (token-budget bounded)                   │
│                    │                              │
│                    ▼                              │
│         Formatted Context → Agent                │
└─────────────────────────────────────────────────┘
```

---

## Requirements

- Python 3.11+
- Ollama running locally with an embedding model (default: `qwen3-embedding:8b`)
  - Set via `PRISM_OLLAMA_MODEL` env var
  - Embedding is optional — BM25 + exact match still work without it
- Tree-sitter (auto-installed per language)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRISM_VAULT_PATH` | — | Path to Obsidian vault |
| `PRISM_EMBED_BACKEND` | `ollama` | Embedding backend (`ollama` or `gemini`) |
| `PRISM_OLLAMA_MODEL` | `bge-m3` | Ollama embedding model name |
| `PRISM_GRAPHS` | — | JSON array of namespace configs (for federation) |

---

