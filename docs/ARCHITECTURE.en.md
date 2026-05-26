# PrismRag v4.0 Architecture

> This document is a concise in-repo architecture summary. The complete design (including ADRs, table schemas, and roadmap) is available at:
>
> 👉 [NimbusVault/knowledge/PrismRag-v4.0-设计文档.md](https://github.com/Zengyn42/NimbusVault/blob/master/knowledge/PrismRag-v4.0-设计文档.md)

---

## Paradigm: Graph-First RAG

The core proposition of PrismRag v4.0 is captured in one sentence from the graphify school of thought:

> **Storage is a graph, clustering is topology, retrieval is traversal; Embeddings are only used at index-time to build similarity edges and multimodal bridges.**

Key differences from traditional RAG:

| Phase | Traditional RAG | PrismRag v4.0 |
|---|---|---|
| **Index time** | Split chunks → embed → write to vector store | Parse AST → extract edges → embed (similarity edges only) → Leiden clustering → generate graph and report |
| **Query time** | Vector search top-K → re-rank → return chunks | Match entry node → graph traversal (BFS/DFS) → token budget pruning → return nodes |

## Five-Layer Pipeline

### Pass 1: AST Extraction (Deterministic, Zero LLM)

**Input**: `.md` files in NimbusVault

**Processing**:
- `python-frontmatter` parses YAML frontmatter → metadata
- `markdown-it-py` parses markdown into AST
- Extract structured signals:
  - `[[wikilink]]` → `links_to` edge
  - `[[note#heading]]` → `links_to_section` edge
  - `[[note^block-id]]` → `links_to_block` edge
  - `#tag` → `tagged_as` edge (tags as independent nodes)
  - `![[embedded.png]]` → `embeds` edge (media nodes)
  - frontmatter `aliases:` → `aliased_as` edge
  - frontmatter `category:` → `categorized_as` edge
  - Callout `> [!NOTE]` → intra-node structure markers

**Output**: All **EXTRACTED** edges (`confidence_score = 1.0`), zero cost.

**Why this pass matters most**: Obsidian wikilinks are **explicitly** declared semantic connections by the user — more precise than code imports. graphify uses tree-sitter for AST extraction on codebases; here we use markdown-it on Obsidian, but the signal quality is even higher.

### Pass 2: Media Extraction (Optional, Enable on Demand)

**Input**: Attachment files (images, PDFs, audio)

**Processing**:
- **Images** → Gemini Vision description → generates `image:filename` text node with Vision description as content
- **PDF** → `pypdf` text extraction → split into markdown-style nodes (or a single large node)
- **Audio** → `faster-whisper` local transcription → text node
  - Advanced: use god nodes from Pass 4 clustering as Whisper domain prompts, improving technical term accuracy (reference: graphify)

**Output**: All media converted to text nodes, attached as targets of the source document's `embeds` edge.

### Pass 3: Embedding + Similarity Edge Generation (THE ONLY PLACE embeddings are used at index-time)

**Input**: All text nodes (Pass 1 markdown + Pass 2 media-to-text)

**Processing**:
1. Compute Gemini Embedding 2 vectors for each node
2. Store in LanceDB (pure cache, not used at query-time)
3. Global top-K nearest neighbor search
4. For each node's top-K neighbors, generate `semantically_similar_to` edges
   - `confidence = INFERRED`
   - `confidence_score = cosine similarity (0.4–0.95)`
   - Below threshold (e.g. 0.6) edges are not generated, to avoid noise
5. Deduplication: if a `semantically_similar_to` edge is dominated by a stronger EXTRACTED edge (e.g. `links_to`), optionally merge / upgrade weight

**Why not use embeddings at query-time**: Graph traversal already reaches "semantically related but not wikilinked" nodes via `semantically_similar_to` edges. The value of embeddings is already "baked into" the graph.

**Multimodal bridge**: Gemini Embedding 2 natively supports text + image + audio + PDF in the same vector space. A text query for "cat" can hit an image node's vector even if the Vision description never used the word "cat".

### Pass 4: Leiden Community Detection

**Input**: The merged complete graph

**Processing**:
- `python-igraph` converts NetworkX to igraph representation
- `leidenalg.find_partition(..., leidenalg.ModularityVertexPartition)` runs community detection
- Edge weight = `confidence_score × weight` (EXTRACTED edges highest, INFERRED next)
- Identify god nodes per community: top-N nodes by degree within the community
- Name communities: using LLM (or simple heuristics) summarizing from god node labels

**Output**:
- `community_id` attribute on each node
- Per-community `{id, label, god_nodes, member_count}`

**Tag priority**: Obsidian `#tag` values can serve as initial partitions for Leiden, allowing community detection to converge faster toward structures that match user mental models.

### Pass 5: Report and Persistence

**Output**:
- `graph.json` — full graph serialization (nodes + edges + communities + metadata)
- `GRAPH_REPORT.md` — human-readable report including:
  - Name, god nodes, and member count per community
  - Top 10 god nodes (the most central nodes of the entire graph)
  - Surprising connections (high-confidence cross-community edges)
  - Open questions (optionally LLM-generated, inferred from "INFERRED edges without EXTRACTED counterparts")
- `graph.html` — optional, interactive visualization generated by `pyvis`

---

## Query Time (Zero Embedding)

The query flow is **pure graph traversal**:

```
User query ──► entry point resolution
                     │
                     ├─ exact match by label
                     ├─ match by alias
                     └─ find nearest node by embedding (fallback)
                     │
                     ▼
              entry_node
                     │
                     ▼
           ┌─────────────────┐
           │ traversal       │
           │  BFS (default)  │──► broad context
           │  DFS (--dfs)    │──► single chain
           │  path (a → b)   │──► shortest path
           └─────────────────┘
                     │
                     ▼
           token budget pruning
           (each node carries a token count, accumulated up to the budget limit)
                     │
                     ▼
           return: [nodes], [edges], community_info
```

**Key properties**:
- **No vector search** (except entry point fallback, which is a single top-1 lookup)
- **No re-ranking**
- **Hard budget cap**: `--budget 4000` means at most 4000 tokens of node content are returned
- **Deterministic**: same query yields same results

## MCP Tools

Exposed MCP tools (see `prism_rag.mcp_server.server`):

| Tool | Purpose | Parameters |
|---|---|---|
| `search_knowledge` | Primary query entry point: query → entry → BFS → return relevant nodes | `query`, `budget`, `mode=bfs\|dfs` |
| `explain_node` | Return all info + neighbor summary for a specific node | `node_id_or_label` |
| `trace_path` | Return the shortest path between two nodes | `from`, `to` |
| `list_communities` | List all communities + god nodes | — |
| `explore_community` | Drill into a community to examine internal structure | `community_id_or_label` |

## Incremental Updates

- Each file's SHA256 is computed and stored in `data/cache/file_hashes.json`
- On the next `ingest`, only files with changed hashes are reprocessed
- Impact of a single file change:
  - Passes 1–3 re-run for that file
  - Pass 4 Leiden incremental update (not a full recompute)
  - Pass 5 report regenerated

## Privacy Tiers

| Data | Local Processing | Sent to Gemini API |
|---|---|---|
| Markdown content, wikilinks, frontmatter | ✅ Pass 1 | ❌ never leaves local |
| PDF text | ✅ pypdf locally | ❌ |
| Audio | ✅ faster-whisper locally | ❌ |
| Image pixels | ❌ | ✅ Gemini Vision |
| Text embeddings | ❌ | ✅ Gemini Embedding 2 |

**Default paid tier**: Gemini free-tier data is used for training. PrismRag requires a paid-tier API key by default; users must explicitly set `PRIVACY_TIER=free` to use the free tier, and a warning is shown at startup.

## Differences from v3.2 Design (Summary)

| Dimension | v3.2 (Traditional RAG) | v4.0 (Graph-First) |
|---|---|---|
| Primary storage | LanceDB vector store | NetworkX + JSON |
| Primary retrieval path | Hybrid Search + RRF + cross-encoder | Graph traversal BFS/DFS |
| Embedding role | Core at query time | Index-time only (similarity edges + multimodal bridge) |
| Phase 1/2 division | Phase 1: basic RAG; Phase 2: GraphRAG | Phase 1 is already the complete graph |
| Code complexity | High (multi-pipeline) | Low (single pipeline) |
| LLM dependency | Query-time re-rank | Index-time only in Pass 2 image description |

**Complete diff, ADRs, and migration notes** are in the v4.0 design document in NimbusVault.
