# PrismRag v4.0 架构

> 本文档是本 repo 内的架构简要版。完整设计（含 ADR、表结构、路线图）见：
>
> 👉 [NimbusVault/knowledge/PrismRag-v4.0-设计文档.md](https://github.com/Zengyn42/NimbusVault/blob/master/knowledge/PrismRag-v4.0-设计文档.md)

---

## 范式：图优先 RAG

PrismRag v4.0 的核心命题是 graphify 流派的一句话：

> **存储是图，聚类是拓扑，检索是遍历；Embedding 只在 index-time 用于构建相似性边和多模态桥。**

这和传统 RAG 的核心区别：

| 时间 | 传统 RAG | PrismRag v4.0 |
|---|---|---|
| **Index time** | 切 chunk → embed → 写向量库 | 解析 AST → 抽取边 → embed（只算相似边）→ Leiden 聚类 → 生成图和报告 |
| **Query time** | 向量搜索 top-K → re-rank → 返回 chunks | 匹配入口节点 → 图遍历（BFS/DFS）→ token 预算裁剪 → 返回节点 |

## 五层 Pipeline

### Pass 1: AST 抽取（确定性，零 LLM）

**输入**：NimbusVault 里的 `.md` 文件

**处理**：
- `python-frontmatter` 解析 YAML frontmatter → metadata
- `markdown-it-py` 把 md 解析成 AST
- 提取结构化信号：
  - `[[wikilink]]` → `links_to` 边
  - `[[note#heading]]` → `links_to_section` 边
  - `[[note^block-id]]` → `links_to_block` 边
  - `#tag` → `tagged_as` 边（tag 作为独立节点）
  - `![[embedded.png]]` → `embeds` 边（媒体节点）
  - frontmatter `aliases:` → `aliased_as` 边
  - frontmatter `category:` → `categorized_as` 边
  - Callout `> [!NOTE]` → intra-node 结构标记

**输出**：全部是 **EXTRACTED** 边（`confidence_score = 1.0`），零成本。

**为什么这一步最重要**：Obsidian 的 wikilink 是用户**显式**声明的语义链接，比代码 import 还精准。graphify 在代码库上靠 tree-sitter 抽 AST；我们在 Obsidian 上靠 markdown-it，但信号质量更高。

### Pass 2: 媒体抽取（可选，按需启用）

**输入**：附件文件（图片、PDF、音频）

**处理**：
- **图片** → Gemini Vision 描述 → 生成 `image:filename` 文本节点，内容是 Vision 的描述
- **PDF** → `pypdf` 提取文本 → 切成 md 风格的节点（或单个大节点）
- **音频** → `faster-whisper` 本地转录 → 文本节点
  - 高级：用 Pass 4 聚类之后的 god nodes 作为 Whisper 的 domain prompt，提升技术术语准确度（参考 graphify）

**输出**：所有媒体都转成文本节点，挂到原文档的 `embeds` 边的目标上。

### Pass 3: Embedding + 相似边生成（INDEX-TIME 唯一用到 embedding 的地方）

**输入**：所有文本节点（Pass 1 的 md + Pass 2 的媒体转文本）

**处理**：
1. 给每个节点算 Gemini Embedding 2 向量
2. 存到 LanceDB（纯 cache，不做 query-time 用）
3. 全局 top-K 近邻搜索
4. 对每个节点的 top-K 邻居，生成 `semantically_similar_to` 边
   - `confidence = INFERRED`
   - `confidence_score = 余弦相似度 (0.4-0.95)`
   - 低于阈值（比如 0.6）的不生成，避免噪声
5. 去重：如果 `semantically_similar_to` 边已经有更强的 EXTRACTED 边（比如 `links_to`），可选择合并 / 升级权重

**为什么不在 query-time 用**：图遍历已经能通过 `semantically_similar_to` 边到达"语义相关但没 wikilink"的节点。Embedding 的价值已经"固化"到图里了。

**多模态桥**：Gemini Embedding 2 原生支持 text + image + audio + PDF 在同一个向量空间。文字 query "猫" 能命中图片节点的向量，即使 Vision 没把"猫"这个词写进描述里。

### Pass 4: Leiden 社区发现

**输入**：合并后的完整图

**处理**：
- `python-igraph` 把 NetworkX 转成 igraph 表示
- `leidenalg.find_partition(..., leidenalg.ModularityVertexPartition)` 跑社区发现
- 边权重 = `confidence_score × weight`（EXTRACTED 边权重最高，INFERRED 次之）
- 每个社区识别 god nodes：社区内 degree 最高的 N 个节点
- 给社区起名：用 LLM（或简单启发式）从 god nodes 的 label 概括

**输出**：
- 每个节点的 `community_id` 属性
- 每个社区的 `{id, label, god_nodes, member_count}`

**Tag 优先级**：Obsidian 的 `#tag` 可以作为 Leiden 的 initial partition（初始标签），让社区发现更快收敛到"符合用户认知"的结构。

### Pass 5: 报告与持久化

**输出**：
- `graph.json` — 图的完整序列化（nodes + edges + communities + metadata）
- `GRAPH_REPORT.md` — 人类可读的报告，包含：
  - 每个社区的名称、god nodes、成员数
  - Top 10 god nodes（整个图的中心节点）
  - Surprising connections（跨社区的高置信度边）
  - Open questions（LLM 可选生成，从"有 INFERRED 边但没 EXTRACTED 边"推导）
- `graph.html` — 可选，`pyvis` 生成的交互式可视化

---

## Query Time（零 embedding）

查询流程是**纯图遍历**：

```
User query ──► entry point resolution
                     │
                     ├─ 按 label 精确匹配
                     ├─ 按 alias 匹配
                     └─ 用 embedding 找最近节点（fallback）
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
           (每个节点带 tokens 计数，累加到 budget 上限)
                     │
                     ▼
           return: [nodes], [edges], community_info
```

**关键性质**：
- **没有 vector search**（除了 entry point 的 fallback，且是 top-1 单次）
- **没有 re-ranking**
- **预算硬上限**：`--budget 4000` 意思是最多返回 4000 tokens 的节点内容
- **确定性**：同样的 query 得到同样的结果

## MCP Tools

对外暴露的 MCP 工具（详见 `prism_rag.mcp_server.server`）：

| 工具 | 用途 | 参数 |
|---|---|---|
| `search_knowledge` | 主要查询入口，query → entry → BFS → 返回相关节点 | `query`, `budget`, `mode=bfs\|dfs` |
| `explain_node` | 返回一个具体节点的所有信息 + 邻居概览 | `node_id_or_label` |
| `trace_path` | 返回两个节点间的最短路径 | `from`, `to` |
| `list_communities` | 列出所有社区 + god nodes | — |
| `explore_community` | 钻进某个社区看内部结构 | `community_id_or_label` |

## 增量更新

- 每个文件算 SHA256，存到 `data/cache/file_hashes.json`
- 下次 `ingest` 时只重处理 hash 变化的文件
- 单个文件变化触发的影响：
  - Pass 1-3 重跑该文件
  - Pass 4 Leiden 增量更新（不全量重算）
  - Pass 5 报告重新生成

## 隐私层级

| 数据 | 本地处理 | 走 Gemini API |
|---|---|---|
| md 内容、wikilinks、frontmatter | ✅ Pass 1 | ❌ 不出本地 |
| PDF 文本 | ✅ pypdf 本地 | ❌ |
| 音频 | ✅ faster-whisper 本地 | ❌ |
| 图片像素 | ❌ | ✅ Gemini Vision |
| 文本 embedding | ❌ | ✅ Gemini Embedding 2 |

**默认付费层**：Gemini 免费层数据用于训练。PrismRag 默认要求付费层 API key，用户显式设置 `PRIVACY_TIER=free` 才能使用免费层，启动时会警告。

## 和 v3.2 设计的差异（简要）

| 维度 | v3.2（传统 RAG） | v4.0（图优先） |
|---|---|---|
| 主存储 | LanceDB 向量库 | NetworkX + JSON |
| 主检索路径 | Hybrid Search + RRF + cross-encoder | 图遍历 BFS/DFS |
| Embedding 角色 | 查询时核心 | index-time 只用一次（相似边 + 多模态桥） |
| Phase 1/2 划分 | Phase 1 基础 RAG，Phase 2 GraphRAG | Phase 1 就是完整图 |
| 代码复杂度 | 高（多管线） | 低（单管线） |
| LLM 依赖 | query-time re-rank | index-time 只在 Pass 2 图片描述 |

**完整差异对比、ADR、迁移说明**见 NimbusVault 里的 v4.0 设计文档。
