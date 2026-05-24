# PrismRag

> 无垠智穹 · 图优先 RAG 系统 · Zengyn42
> 从 Markdown vault 和代码库构建知识图谱，通过 MCP Server 对外提供基于图遍历的语义检索。

**状态**：✅ v5.7 完成，准备进入 v6.0（联邦元图）

---

## 核心命题

> **Clustering is graph-topology-based. Retrieval is graph traversal. Embedding only builds edges.**

和传统 RAG 的区别：

| 维度 | 传统 RAG | PrismRag |
|---|---|---|
| 主存储 | 向量数据库 | NetworkX 图 + JSON |
| 主检索 | 向量相似度搜索 | 图遍历 (BFS / DFS) |
| Embedding 角色 | Query-time 核心路径 | **Index-time only**（生成相似边） |
| 聚类 | 可选 / 不做 | Leiden 社区发现（拓扑驱动） |
| 可解释性 | 向量距离数字 | EXTRACTED / INFERRED / AMBIGUOUS 置信度标签 |
| 增量 | 全量重建 | SHA256 文件 cache，只重算变化文件 |

---

## 版本历程

| 版本 | 状态 | 主要内容 |
|------|------|---------|
| v4.0 | ✅ | 图优先架构基础：NetworkX + Leiden + BFS/DFS + MCP Server |
| v5.0 | ✅ | 多 namespace 支持、联合图、增量 ingest |
| v5.1 | ✅ | BM25 混合检索、hybrid search |
| v5.2 | ✅ | Edge classifier（EXTRACTED/INFERRED/AMBIGUOUS 置信度） |
| v5.3 | ✅ | `atomize_document`：LLM 将长文档拆成原子知识片段 + inbox 审核流程 |
| v5.4 | ✅ | KNOW-ID 路由 + label resolver（稳定节点 ID） |
| v5.5 | ✅ | Atomize 语义去重（复用已有节点，减少冗余）|
| v5.6 | ✅ | 图可视化升级：Obsidian URI 深链、portal 跨 namespace 节点 |
| **v5.7** | ✅ | **force-graph WebGL 渲染器**（替换 pyvis）、统一 code+docs 图、ego-graph 焦点、多选 legend、聚类语义命名 |
| v6.0 | 🔜 | 联邦元图（多 namespace 全局视图）|

---

## v5.7 可视化功能

graph.html 基于 [force-graph](https://github.com/vasturiano/force-graph)（WebGL Canvas），支持：

- **节点焦点**：单击节点 → 只显示该节点 + 直接邻居，其他节点变暗
- **Legend 多选**：点击左侧色块 → 显示对应聚类所有节点；可叠加多个
- **3-click 循环**：×1 焦点节点 → ×2 选中该节点所属聚类 → ×3 取消
- **语义聚类命名**：legend 显示聚类中心节点的标签（如 `#LangGraph (36)`），而非 `doc group`
- **LOD 标签**：缩放到一定级别才显示节点名称，避免拥挤
- **键盘控制**：WASD 平移，`+`/`-` 缩放，`Escape` 重置
- **on-demand 边**：`mentions_symbol`（doc→code 引用）默认隐藏，点击节点时按需显示
- **右键打开 Obsidian**：doc 节点右键 → `obsidian://` 直接跳转原始笔记（需 `--vault` 参数）

---

## 管线概览

```
Vault (.md) + Repo (.py)
       │
       ├─ Pass 1a: Markdown AST 抽取
       │    └─ wikilinks / tags / frontmatter → EXTRACTED 边 (confidence=1.0)
       │
       ├─ Pass 1b: Python 代码 AST (Tree-sitter)
       │    └─ module / class / function / import → code:: 节点 + 调用边
       │
       ├─ Pass 2: Leiden 社区发现
       │    └─ 纯拓扑聚类，识别社区 + 中心节点
       │
       ├─ Pass 3a: Embedding (Ollama bge-m3 / Gemini)
       │    └─ 向量化每个节点，写入 Lance 索引
       │
       ├─ Pass 3b: 相似边生成
       │    └─ doc↔doc / code↔code / doc↔code 语义相似度边
       │
       ├─ Pass 3c: Symbol links
       │    └─ doc 文本中提到的代码符号 → mentions_symbol 边
       │
       └─ Pass 4: 持久化
            ├─ graph.json          # 知识图谱
            ├─ GRAPH_REPORT.md     # 社区概览
            └─ graph.html          # force-graph 交互式可视化
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
# 安装（开发模式）
git clone git@github.com:Zengyn42/PrismRag.git
cd PrismRag
pip install -e ".[dev]"

# 配置（.env 或环境变量）
cp .prism_env .env
# 编辑 .env: PRISM_VAULT_PATH, GEMINI_API_KEY（可选）
```

### ingest 文档 vault

```bash
prism-rag ingest --vault ~/Projects/MyVault --namespace my_project
# → 输出到 ~/Projects/MyVault/.prismrag/my_project/
```

### ingest 代码 + 文档统一图

```bash
prism-rag ingest-project --repo ~/Projects/Pulsify
# → 输出到 ~/Projects/Pulsify/.prismrag/pulsify/
```

### 查看可视化

```bash
prism-rag visualize
# → 打开 .prismrag/<ns>/graph.html
```

### 启动 MCP Server

```bash
prism-rag serve          # stdio 模式（Claude Code / Cursor）
prism-rag serve --transport sse  # SSE 模式（网络访问）
```

### atomize（原子化拆分长文档）

```bash
prism-rag atomize propose --node "my-note-id"
prism-rag atomize inbox             # 审核提案
prism-rag atomize promote <id>      # 批准写入图
```

---

## 目录结构

```
PrismRag/
├── prism_rag/
│   ├── cli.py                  # CLI 入口 (prism-rag 命令集)
│   ├── cli_atomize.py          # atomize 子命令
│   ├── config.py               # PrismRagSettings
│   ├── ingest/                 # 摄入管线
│   │   ├── vault_loader.py     # Markdown 文件扫描
│   │   ├── obsidian_parser.py  # Obsidian wikilinks/frontmatter
│   │   ├── ast_extractor.py    # 图节点/边提取
│   │   ├── code_parser.py      # Tree-sitter Python 解析
│   │   ├── atomize.py          # LLM 原子化拆分
│   │   ├── embedder.py         # Ollama / Gemini 向量化
│   │   ├── similarity_linker.py# 语义相似度建边
│   │   ├── symbol_linker.py    # doc→code 符号引用边
│   │   ├── edge_classifier.py  # 边置信度分类
│   │   ├── label_resolver.py   # KNOW-ID 标签解析
│   │   ├── dedup_log.py        # 去重日志
│   │   └── incremental.py      # 增量更新
│   ├── store/                  # 图数据库层
│   │   ├── graph.py            # KnowledgeGraph 核心
│   │   ├── networkx_backend.py # NetworkX 实现
│   │   ├── embedding_store.py  # Lance 向量索引
│   │   ├── bm25_index.py       # BM25 关键词索引
│   │   ├── federated.py        # 多 namespace 联合图
│   │   └── registry.py         # Namespace 注册表
│   ├── cluster/
│   │   └── leiden.py           # Leiden 社区检测
│   ├── retrieve/               # 检索引擎
│   │   ├── entry.py            # 检索入口
│   │   ├── hybrid.py           # 混合检索
│   │   ├── bfs.py / dfs.py     # 图遍历
│   │   └── impact.py           # 影响分析
│   ├── inbox/                  # Atomize 审核收件箱
│   ├── report/
│   │   ├── visualize.py        # force-graph WebGL HTML
│   │   └── graph_report.py     # 文字统计报告
│   ├── vault_ops/              # Vault 文件读写操作
│   └── mcp_server/             # MCP Server (18 tools)
├── docs/                       # 架构文档 + 设计方案
├── pyproject.toml
└── README.md
```

---

## 技术栈

| 层 | 选型 |
|---|---|
| 图存储 | NetworkX + JSON |
| 社区发现 | `leidenalg` + `python-igraph` |
| 代码解析 | `tree-sitter` |
| Markdown AST | `markdown-it-py` + `python-frontmatter` |
| Embedding | Ollama `bge-m3`（默认）/ Gemini Embedding |
| 向量索引 | LanceDB（index-time only）|
| 可视化 | [force-graph](https://github.com/vasturiano/force-graph)（WebGL Canvas） |
| MCP Server | `mcp` 官方 SDK (FastMCP) |

---

## 相关仓库

| Repo | 定位 |
|---|---|
| [Zengyn42/ZenithLoom](https://github.com/Zengyn42/ZenithLoom) | Agent 编排引擎（LangGraph） |
| [Zengyn42/NimbusVault](https://github.com/Zengyn42/NimbusVault) | Obsidian 知识库（PrismRag 数据源之一） |
| **Zengyn42/PrismRag** | **本仓库** |

---

## License

Proprietary — 内部使用。

---

*— Zengyn42 · 无垠智穹*
