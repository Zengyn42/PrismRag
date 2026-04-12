# PrismRag

> 无垠智穹 · 图优先 RAG 系统 · Zengyn42
> 从 [NimbusVault](https://github.com/Zengyn42/NimbusVault)（Obsidian 知识库）构建知识图谱，通过 MCP Server 对外提供基于图遍历的语义检索。

**状态**：🚧 v4.0 设计中（graphify 流派），骨架已建，实现待开发

---

## 核心命题

> **Clustering is graph-topology-based. Retrieval is graph traversal. Embedding only builds edges.**

PrismRag v4.0 借鉴 [safishamsi/graphify](https://github.com/safishamsi/graphify) 的图优先架构，同时保留 Gemini Embedding 2 作为**索引时的相似性边生成器**和**多模态桥**。

**和传统 RAG 的区别**：

| 维度 | 传统 RAG | PrismRag v4.0 |
|---|---|---|
| 主存储 | 向量数据库 | NetworkX 图 (+ JSON 序列化) |
| 主检索 | 向量相似度搜索 | 图遍历 (BFS / DFS) |
| Embedding 角色 | Query-time 核心路径 | **Index-time only**（生成相似边 + 多模态桥） |
| 聚类 | 可选 / 不做 | Leiden 社区发现（拓扑驱动） |
| 可解释性 | 向量距离数字 | EXTRACTED / INFERRED / AMBIGUOUS 置信度标签 |
| 查询开销 | O(vector search) | O(graph traversal) — 便宜 |
| 增量 | 全量重建 embeddings | SHA256 文件 cache，只重算变化文件 |

## 管线概览

```
NimbusVault (.md + 附件)
       │
       ├─ Pass 1: AST 抽取（确定性，零 LLM）
       │    └─ 解析 wikilinks [[...]] / tags #... / frontmatter / callouts
       │       → EXTRACTED 边（confidence_score = 1.0）
       │
       ├─ Pass 2: 媒体抽取
       │    ├─ 图片 → Gemini Vision 描述 → 文本节点
       │    ├─ PDF  → pypdf 抽文本 → 文本节点
       │    └─ 音频 → faster-whisper 本地转录 → 文本节点
       │
       ├─ Pass 3: Embedding + 相似边生成（index-time only）
       │    ├─ Gemini Embedding 2 算每个节点的向量
       │    ├─ 全局 top-K 近邻检索
       │    └─ 生成 semantically_similar_to 边（INFERRED, score = 余弦相似度）
       │
       ├─ Pass 4: Leiden 社区发现
       │    └─ 纯拓扑聚类，识别社区 + god nodes（最高度数概念）
       │
       └─ Pass 5: 报告生成
            ├─ graph.json          # 持久化知识图
            ├─ GRAPH_REPORT.md     # 社区概览 + god nodes + 惊奇连接
            └─ graph.html          # 可选，pyvis 交互式可视化
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  MCP Server (query-time) │
                    │                          │
                    │  search_knowledge(query) │
                    │  explain_node(name)      │
                    │  trace_path(from, to)    │
                    │  list_communities()      │
                    │  explore_community(name) │
                    └──────────────────────────┘
                                   │
                                   ▼
              ZenithLoom agents (Hani / Asa / apex_coder / ...)
```

## 关键技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| **图存储** | NetworkX + JSON | 零部署、可序列化、工具生态成熟 |
| **社区发现** | `leidenalg` + `python-igraph` | Leiden 算法的规范实现（比 Louvain 更稳定） |
| **Markdown AST** | `markdown-it-py` + `python-frontmatter` | 支持 wikilinks / callouts / frontmatter |
| **Embedding** | Gemini Embedding 2 | 原生多模态，统一向量空间，index-time 一次性调用 |
| **Embedding 缓存** | LanceDB（降级用途） | 只做 embedding cache，不做 query-time 检索 |
| **MCP Server** | `mcp` 官方 SDK | 标准化工具协议，Zengyn42 其他 agent 可调用 |
| **本地音频** | `faster-whisper` (optional) | 隐私：音频不出本地 |
| **PDF / 图片** | `pypdf` / `pillow` + Gemini Vision | 按需启用 |

## 设计原则

1. **图优先** — Query-time 绝不做向量搜索，只走图遍历
2. **Embedding 降级** — 只在 index-time 用一次，生成相似边 + 多模态桥
3. **透明置信度** — 每条边标 EXTRACTED / INFERRED / AMBIGUOUS
4. **Token 预算** — 所有查询接 `--budget N` 硬上限
5. **增量友好** — SHA256 文件 cache，只重算变化的文件
6. **隐私优先** — 本地优先（md AST / pdf / whisper）；只有 embedding 和图片描述走 Gemini API
7. **零运维** — `pip install -e .` 即可启动，没有 Neo4j / Qdrant / Docker

## Graph Schema

**Node:**
```python
{
    "id": "filestem_entityname",          # 稳定 ID
    "label": "Human Readable Name",
    "kind": "note|concept|tag|image|pdf|audio",
    "source_file": "relative/path.md",
    "content_hash": "sha256:...",         # 增量检测
    "frontmatter": {...},
    "tokens": 1234,                       # 节点内容 token 数（budget 管理用）
}
```

**Edge:**
```python
{
    "source": "node_id_a",
    "target": "node_id_b",
    "relation": "links_to|tagged_as|embeds|semantically_similar_to|mentions|illustrates",
    "confidence": "EXTRACTED|INFERRED|AMBIGUOUS",
    "confidence_score": 1.0,              # EXTRACTED=1.0, INFERRED=0.4-0.9, AMBIGUOUS=0.1-0.3
    "weight": 1.0,                        # 用于 Leiden 聚类
    "source_pass": "ast|media|embedding|llm",
}
```

## 目录结构

```
PrismRag/
├── prism_rag/
│   ├── __init__.py
│   ├── config.py               # Settings (paths, API keys, privacy tier)
│   ├── cli.py                  # Typer CLI entrypoint
│   │
│   ├── ingest/                 # 5-pass pipeline
│   │   ├── vault_loader.py     # Pass 1: md + frontmatter parsing
│   │   ├── ast_extractor.py    # Pass 1: wikilinks / tags / callouts → EXTRACTED edges
│   │   ├── media_extractor.py  # Pass 2: image / pdf / audio → text nodes
│   │   ├── embedder.py         # Pass 3: Gemini Embedding 2 client
│   │   └── similarity_linker.py # Pass 3: global top-K → semantically_similar_to edges
│   │
│   ├── store/
│   │   ├── graph.py            # NetworkX graph + JSON (de)serialization
│   │   └── embedding_cache.py  # LanceDB cache (index-time only)
│   │
│   ├── cluster/
│   │   └── leiden.py           # Pass 4: Leiden community detection + god node detection
│   │
│   ├── report/
│   │   └── graph_report.py     # Pass 5: GRAPH_REPORT.md generation
│   │
│   ├── retrieve/               # Query-time (no vector search)
│   │   ├── bfs.py              # BFS traversal
│   │   ├── dfs.py              # DFS traversal
│   │   ├── budget.py           # Token-budget-aware pruning
│   │   └── entry.py            # Entry point resolution (node match)
│   │
│   └── mcp_server/
│       └── server.py           # MCP tools: search/explain/trace/communities
│
├── tests/
├── docs/
│   └── ARCHITECTURE.md         # 架构详解（本 repo 内）
├── data/                       # Runtime: graph.json + embedding cache (gitignored)
├── pyproject.toml
├── .gitignore
└── README.md
```

## Quick Start

```bash
# 安装（开发模式）
git clone git@github.com:Zengyn42/PrismRag.git
cd PrismRag
pip install -e ".[dev,media,viz]"

# 配置（settings via env vars or .env）
export GEMINI_API_KEY="..."
export PRISM_VAULT_PATH="$HOME/Foundation/Vault"

# 初次索引 NimbusVault（未实现）
# prism-rag ingest

# 启动 MCP Server（未实现）
# prism-rag serve

# 查询（未实现）
# prism-rag query "辩论子图的 session 机制"
# prism-rag path "subgraph_topic" "fresh_per_call"
# prism-rag explain "SubgraphMapperNode"
```

## 相关仓库

| Repo | 定位 |
|---|---|
| [Zengyn42/ZenithLoom](https://github.com/Zengyn42/ZenithLoom) | Agent 编排引擎（LangGraph 核心） |
| [Zengyn42/NimbusVault](https://github.com/Zengyn42/NimbusVault) | Obsidian 知识库（PrismRag 的数据源） |
| **Zengyn42/PrismRag** | **本仓库** |

## 设计文档

- 完整架构设计（含 ADR、表结构、Phase 路线）：[NimbusVault/knowledge/PrismRag-v4.0-设计文档.md](https://github.com/Zengyn42/NimbusVault/blob/master/knowledge/PrismRag-v4.0-设计文档.md)（待写）
- 历史 v3.2 设计（向量检索流派）：[NimbusVault/knowledge/Obsidian 多模态 RAG 系统架构设计.md](https://github.com/Zengyn42/NimbusVault/blob/master/knowledge/Obsidian多模态RAG系统架构设计.md) — 作为历史记录保留

## 灵感来源

- [safishamsi/graphify](https://github.com/safishamsi/graphify) — 图优先 RAG、Leiden 拓扑聚类、EXTRACTED/INFERRED/AMBIGUOUS 置信度标签
- Gemini Embedding 2（多模态向量空间）
- Obsidian wikilink graph（天然的 EXTRACTED 边源）

## License

Proprietary — 内部使用。

---

*— Zengyn42 · 无垠智穹*
