# PrismRag 架构概览

> 本文档是架构简要版本。完整设计（v3.2，含 ADR #1-52、表结构、GraphRAG Phase 2）
> 见 NimbusVault：
>
> 👉 [knowledge/Obsidian 多模态 RAG 系统架构设计.md](https://github.com/Zengyn42/NimbusVault/blob/master/knowledge/Obsidian多模态RAG系统架构设计.md)

---

## 数据流

```
NimbusVault (Obsidian Markdown + 附件)
        │
        ▼
┌─────────────────┐
│ prism_rag.ingest │
│                  │
│ vault_loader ──► │  读 .md + frontmatter + 附件
│ chunker ────────►│  切分（优先级：段落 > 句子，800 tokens 阈值）
│ embedder ───────►│  Gemini Embedding 2（多模态统一向量空间）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ prism_rag.store  │
│                  │
│ LanceDB          │  统一存储 chunks / atoms / relations
│  ├─ chunks table │
│  ├─ atoms table  │  (Phase 2: K-atom 知识原子)
│  └─ relations    │  (Phase 2: GraphRAG 关系图)
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ prism_rag.retrieve   │
│                      │
│ hybrid (Phase 1)     │  dense + sparse + RRF + re-rank
│ graph (Phase 2)      │  Leiden 社区发现 + beam search
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ prism_rag.mcp_server │  MCP 协议
│                      │
│ tools:               │
│  - search_knowledge  │
│  - get_note          │
│  - list_sources      │
│  - (diagnostic)      │
└──────────────────────┘
           │
           ▼
   ZenithLoom agents (Hani / Asa / apex_coder / ...)
```

## Phase 路线图

### Phase 1 — MVP（待开发）
- Obsidian Vault loader（md + frontmatter + attachment 抽取）
- 基于段落优先级的 chunker（`tiktoken` 计数）
- Gemini Embedding 2 客户端（带隐私层检查：默认付费层）
- LanceDB 存储（chunks 表 + Hybrid Search）
- MCP Server（tools: `search_knowledge`, `get_note`）
- CLI：`prism-rag ingest` / `prism-rag serve`

### Phase 2 — GraphRAG（设计完成，待开发）
- Atoms（K-atom 知识原子提取）
- Relations（双向链接 + 语义关系）
- Leiden 社区发现（`python-louvain`）
- Ensemble Retrieval + RRF 融合
- Cross-encoder Re-ranking（`sentence-transformers`）
- 渐进式激活机制

## 核心决策（从 NimbusVault ADR 摘录）

| ADR | 决策 | 动机 |
|---|---|---|
| #1 | LanceDB 取代 Qdrant + SQLite | 零部署、单存储、内置 Hybrid Search |
| #5 | 默认付费层 API | 免费层数据用于训练，隐私风险 |
| #12 | Checkpoint 多行设计 | 可中断可恢复的幂等同步 |
| #31-45 | 3 轮蜂群审查修正 | SQL 注入防护、Schema 迁移、dry_run |
| #46-52 | GraphRAG Phase 2 定稿 | Leiden + Ensemble + Re-ranking |

完整 ADR 列表见 NimbusVault 设计文档。

## 配置约定

- **隐私层**：默认 `paid`。切换到 `free` 需显式写入配置文件 `privacy_tier = "free"`
- **知识源路径**：默认读 `~/Foundation/Vault/`（本地 Obsidian vault 目录）
- **LanceDB 存储路径**：默认 `./data/lance/`（.gitignore 已排除）
- **API key**：`GEMINI_API_KEY` 环境变量
