# PrismRag v5.7 架构

> 历史版本：[v4.0 架构说明](archive/ARCHITECTURE-v4.0.md)

---

## 核心命题

> **Clustering is graph-topology-based. Retrieval is graph traversal. Embedding only builds edges.**

PrismRag 与传统 RAG 的本质区别：

| 维度 | 传统 RAG | PrismRag v5.7 |
|---|---|---|
| 主存储 | 向量数据库 | NetworkX 图 + JSON |
| 主检索 | 向量相似度搜索（query time） | 图遍历 BFS / DFS（query time） |
| Embedding 角色 | Query-time 核心路径 | **Index-time only**（生成相似边） |
| 聚类 | 可选 / 不做 | Leiden 社区发现（纯拓扑，无 LLM） |
| 可解释性 | 向量距离数字 | EXTRACTED / INFERRED / AMBIGUOUS 置信度标签 |
| 增量更新 | 全量重建 | SHA256 文件 cache，只重处理变化文件 |
| 数据来源 | 单一文档集 | Markdown vault + Python 代码库（统一图） |

---

## 输入类型

| 类型 | 命令 | 说明 |
|---|---|---|
| Markdown vault（Obsidian） | `prism-rag ingest` | 纯文档图 |
| Python 代码库 | `prism-rag ingest-code` | 纯代码图 |
| 代码 + 文档统一图 | `prism-rag ingest-project` | code + docs，symbol link 跨接 |

输出目录约定：`<target>/.prismrag/<namespace>/`

---

## Pipeline 总览（七步）

```
Vault (.md) + Repo (.py)
       │
       ├─ Pass 1a: Markdown AST 抽取
       │
       ├─ Pass 1b: Python 代码 AST（Tree-sitter）
       │
       ├─ Pass 2: Leiden 社区发现
       │
       ├─ Pass 3a: Embedding（Ollama / Gemini）
       │
       ├─ Pass 3b: 相似边生成
       │
       ├─ Pass 3c: Symbol links（doc→code）
       │
       └─ Pass 4: 持久化 + 可视化
```

---

## Pass 1a：Markdown AST 抽取（确定性，零 LLM）

**输入**：vault 下的 `.md` 文件

**处理**：
- `python-frontmatter` 解析 YAML frontmatter → metadata
- `markdown-it-py` 把 md 解析成 AST
- 提取结构化信号：

| 信号 | 生成边类型 | 置信度 |
|---|---|---|
| `[[wikilink]]` | `links_to` | EXTRACTED (1.0) |
| `[[note#heading]]` | `links_to_section` | EXTRACTED (1.0) |
| `#tag` | `tagged_as` | EXTRACTED (1.0) |
| frontmatter `aliases:` | `aliased_as` | EXTRACTED (1.0) |
| frontmatter `category:` | `categorized_as` | EXTRACTED (1.0) |

**输出**：全部 **EXTRACTED** 边（`confidence_score = 1.0`），零 LLM 成本。

**KNOW-ID 路由（v5.4+）**：frontmatter 中声明 `knowledge_id` 的节点使用稳定 ID；label 按三层规则解析：`title` → `clean_slug` → stem。

---

## Pass 1b：Python 代码 AST（Tree-sitter）

**输入**：repo 下的 `.py` 文件

**处理**：
- `tree-sitter` 解析 Python AST
- 提取以下代码结构：

| 结构 | 节点前缀 | 示例 |
|---|---|---|
| 模块 | `code::module::` | `code::module::prism_rag.cli` |
| 类 | `code::class::` | `code::class::KnowledgeGraph` |
| 函数 | `code::func::` | `code::func::search_knowledge` |
| import | — | 生成 `imports` 边 |

- 生成 `defines` 边（module → class/func）
- 生成 `calls` 边（caller → callee，static call graph）
- 生成 `imports` 边（module → imported symbol）

**输出**：code:: 节点 + 调用图边，全部 EXTRACTED。

---

## Pass 2：Leiden 社区发现

**输入**：Pass 1 生成的完整图（含 doc 节点 + code 节点）

**处理**：
- `python-igraph` 将 NetworkX 转为 igraph 表示
- `leidenalg.find_partition(ModularityVertexPartition)` 跑社区发现
- 边权重 = `confidence_score × weight`（EXTRACTED 边权重最高）
- 每个社区识别 **hub node**（社区内 degree 最高的节点）
- Hub node label 作为社区名（可视化图例用，如 `#LangGraph (36)`）

**输出**：每个节点标记 `community_id`；社区 metadata（hub node、成员数）

---

## Pass 3a：Embedding（INDEX-TIME ONLY）

**输入**：所有文本节点

**模型选项**：

| 模型 | 后端 | 向量维度 | 适用场景 |
|---|---|---|---|
| `bge-m3` | Ollama（本地） | 1024 | 默认，支持中英文 |
| `qwen3-embedding:8b` | Ollama（本地） | 1024 | 中文为主的内容 |
| `text-embedding-004` | Gemini API | 768 | 云端备用 |

**处理**：
1. 给每个节点计算向量
2. 写入 LanceDB（`embed_cache.lance`），仅作 cache
3. SHA256 内容 hash 防止重复计算

**关键约束**：向量**不在 query time 使用**（仅 Pass 3b 建边时用）

---

## Pass 3b：相似边生成

**输入**：LanceDB 向量 cache

**处理**：
- 全局 top-K ANN 搜索（默认 K=10）
- 对每对节点生成 `semantically_similar_to` 边
- 置信度规则：
  - `confidence = INFERRED`
  - `confidence_score = 余弦相似度`
  - 低于阈值（默认 0.5）的不生成

**跨类型边**：
- doc ↔ doc
- code ↔ code
- doc ↔ code（需 `--cross-modal`）

---

## Pass 3c：Symbol Links（doc → code）

**输入**：doc 节点文本 + code 符号集合

**处理**：
- 扫描 doc 节点内容，匹配 code 符号名（精确字符串匹配）
- 生成 `mentions_symbol` 边（doc → code）
- 置信度 = EXTRACTED

**可视化行为**：`mentions_symbol` 边默认隐藏，点击 doc 节点时按需显示（避免图面过于拥挤）

---

## Pass 4：持久化 + 可视化

**输出文件**（存入 `.prismrag/<namespace>/`）：

| 文件 | 描述 |
|---|---|
| `graph.json` | 完整知识图谱（nodes + edges + communities + metadata） |
| `GRAPH_REPORT.md` | 文字统计报告（社区概览、hub nodes、边统计） |
| `graph.html` | force-graph WebGL 交互式可视化 |
| `embed_cache.lance/` | LanceDB 向量 cache |
| `bm25_index/` | BM25 关键词索引 |

### graph.html 可视化特性（v5.7）

基于 [force-graph](https://github.com/vasturiano/force-graph)（WebGL Canvas）：

| 功能 | 描述 |
|---|---|
| 节点焦点 | 单击节点 → 只显示该节点 + 直接邻居 |
| Legend 多选 | 点击左侧色块 → 显示对应聚类；可叠加 |
| 3-click 循环 | ×1 焦点节点 → ×2 选中聚类 → ×3 取消 |
| 语义聚类命名 | legend 显示 hub node label（如 `#LangGraph (36)`）|
| LOD 标签 | 缩放到一定级别才显示节点名，避免拥挤 |
| 键盘控制 | WASD 平移，`+`/`-` 缩放，Escape 重置 |
| On-demand 边 | `mentions_symbol` 默认隐藏，点击节点按需显示 |
| 右键 Obsidian | doc 节点右键 → `obsidian://` 跳转原始笔记 |

---

## 增量更新

- 每个文件算 SHA256，存到 `file_hashes.json`
- 再次 ingest 时只重处理 hash 变化的文件
- Embedding cache 按 `(node_id, content_hash)` 键控，未变化节点复用向量
- Leiden 在图变化后全量重跑（增量 Leiden 仍在研究中）

---

## Query Time（零 Embedding）

```
User query
    │
    ▼
入口节点解析
    ├─ label / alias 精确匹配
    ├─ BM25 关键词搜索
    └─ ANN 向量搜索（fallback，top-1）
    │
    ▼
图遍历
    ├─ BFS（默认，广度优先，宽上下文）
    ├─ DFS（深度优先，单链追踪）
    └─ path(a→b)（最短路径）
    │
    ▼
Token 预算裁剪（--budget N）
    │
    ▼
返回：[nodes], [edges], community_info
```

**核心性质**：
- **无 vector search**（除 entry point fallback，且只 top-1）
- **无 re-ranking**
- **确定性输出**：同 query 同结果
- **硬预算上限**：`--budget 4000` 最多返回 4000 tokens

---

## Atomize 流程（v5.3+）

针对"一个 md 文件包含多个独立知识点"的情况：

```
prism-rag atomize propose --node <id>
    │
    ▼ LLM（Gemini / Claude）拆分
    │
    ▼ 生成 atomic proposals（inbox）
    │
prism-rag atomize inbox          # 人工审核
    │
prism-rag atomize promote <id>   # 批准 → 写入图
```

- 原子节点使用 `knowledge_id` frontmatter 作为稳定 ID
- 语义去重（v5.5）：生成前检查是否已有等价节点，避免冗余

---

## MCP Server（18 tools）

`prism-rag serve` 启动，支持 stdio 和 SSE 两种 transport。

### 检索工具

| 工具 | 用途 |
|---|---|
| `search_knowledge` | 主检索（hybrid: BM25 + entry + BFS） |
| `explain_node` | 返回节点详情 + 邻居概览 |
| `trace_path` | 两节点间最短路径 |
| `list_communities` | 列出所有社区 + hub nodes |
| `explore_community` | 深入某个社区 |

### Atomize 工具

| 工具 | 用途 |
|---|---|
| `atomize_document` | LLM 拆分节点为原子片段 |
| `inbox_review` | 列出待审核提案 |
| `inbox_promote` | 批准提案写入图 |

### Vault CRUD 工具（11 个）

读取、创建、更新、删除 Obsidian vault 中的笔记；支持 frontmatter 操作、alias 管理等。

---

## 数据存储布局

```
<target>/.prismrag/<namespace>/
├── graph.json            # 完整知识图谱
├── GRAPH_REPORT.md       # 统计报告
├── graph.html            # 交互式可视化
├── file_hashes.json      # SHA256 增量 cache
├── embed_cache.lance/    # LanceDB 向量 cache
└── bm25_index/           # BM25 索引
```

每个 target（vault 或 repo）有自己独立的 `.prismrag/` 目录，互不干扰。

---

## 技术栈

| 层 | 技术选型 |
|---|---|
| 图存储 | NetworkX + JSON |
| 社区发现 | `leidenalg` + `python-igraph` |
| 代码解析 | `tree-sitter` |
| Markdown AST | `markdown-it-py` + `python-frontmatter` |
| Embedding | Ollama `bge-m3` / `qwen3-embedding:8b`（默认）；Gemini API（备用） |
| 向量 cache | LanceDB |
| 关键词索引 | BM25（`rank_bm25`） |
| 可视化 | [force-graph](https://github.com/vasturiano/force-graph)（WebGL Canvas） |
| MCP Server | `mcp` 官方 SDK（FastMCP） |
| 配置 | `pydantic-settings`（.env / 环境变量） |

---

## v7.0 展望：联邦元图

> 详见 [v7.0-design.md](v7.0-design.md) 和 [v7.0-implementation-plan.md](v7.0-implementation-plan.md)

核心目标：跨多个 namespace（不同 vault / repo）的统一全局视图。

- **manifest.json**：每个 namespace 的元数据注册表
- **FederatedGraph**：运行时联合多个 KnowledgeGraph 实例
- **跨 namespace 桥接边**：共享 tag + import（确定性）→ hub node ANN（语义）
- **联邦可视化**：namespace 超节点元图 + 下钻到单 namespace 图

---

*— Zengyn42 · 无垠智穹*
