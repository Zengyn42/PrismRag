# [v3.2 · 历史记录] Obsidian 多模态 RAG 系统 — 架构设计

> ⚠️ **历史文档** — 2026-04-11 起被 v4.0（图优先 RAG）取代。
> 新版设计见 [PrismRag-v4.0-设计文档.md](./PrismRag-v4.0-设计文档.md)。
> 本文件作为设计思考的历史记录保留，其中多数 ADR（隐私层、checkpoint 机制、SQL 注入防护等）在 v4.0 中仍然适用；被替换的是"向量检索 + Hybrid Search + RRF + Re-rank"的主检索路径，v4.0 改走"图遍历 BFS/DFS + index-time embedding 辅助"。
>
> 版本：v3.2 | 日期：2026-04-07 | 作者：Hani · 无垠智穹
> 状态：**已被 v4.0 取代**（向量检索流派 → 图优先流派）
> v3.0 变更：LanceDB 统一替换 Qdrant + SQLite（Claude-Gemini 四轮辩论共识），完整表结构定义，Checkpoint 同步机制，统一 SearchQuery 接口，新增诊断 MCP Tools
> v3.1 变更：3 轮蜂群审查 + Claude-Gemini 终审辩论修正（ADR #31-45），修复 list_ LIKE 查询、双向冗余、SQL 注入防护、Checkpoint 多行设计、Schema 迁移策略、dry_run 模式、技术债盘点机制
> v3.2 变更：Phase 2 GraphRAG 架构定稿（Claude-Gemini 三轮辩论，ADR #46-52）— Leiden 社区发现 + 软归属层、Ensemble Retrieval + RRF 融合、Cross-encoder Re-ranking、渐进式激活机制、Deep GraphRAG 论文借鉴

---

## 1. 项目概述

### 1.1 目标

构建一个基于 Obsidian Vault 的多模态 RAG（Retrieval-Augmented Generation）系统，支持文本、图片、PDF、音频的统一语义检索，以 MCP Server 形态部署，供无垠智穹全体 Agent 调用。

### 1.2 核心技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| 知识源 | Obsidian Vault（Markdown + 附件） | 双向链接 = 天然知识图谱 |
| Embedding | Gemini Embedding 2（2026.3.10 发布） | 原生多模态，统一向量空间，免费额度 + 低成本付费 |
| 存储层 | LanceDB（嵌入式） | 零部署、单存储统一 chunks/atoms/relations、内置 Hybrid Search |
| 部署形态 | MCP Server | 标准化工具协议，所有 Agent 可调用 |

### 1.3 设计原则

1. **务实精简** — Phase 1 只做必要功能，拒绝过度工程
2. **隐私优先** — 默认要求付费层 API，免费层需强制确认风险
3. **幂等可恢复** — 同步过程可中断可恢复，Checkpoint + 精确清理保证一致性
4. **多模态原生** — 充分利用 Gemini Embedding 2 的跨模态能力
5. **六空间罗盘** — 以王延章六空间理论（K/S/D/F/I/O）为设计语言，指导知识建模方向，但不作为系统规格强制实现
6. **零部署** — 用户无需安装 Docker 或外部服务，`pip install` 即可启动

---

## 2. Gemini Embedding 2 技术参数

### 2.1 模型信息

- **模型 ID**: `gemini-embedding-2-preview`
- **发布日期**: 2026-03-10
- **支持模态**: 文本、图片、音频、视频、PDF → 统一向量空间
- **维度**: 128~3072 可调（Matryoshka），推荐 768
- **语言**: 100+ 语言

### 2.2 输入限制

| 模态 | 格式 | 单次请求限制 |
|------|------|-------------|
| 文本 | 纯文本 | 8,192 tokens |
| 图片 | PNG, JPEG | 最多 6 张/请求 |
| PDF | PDF | 最多 6 页/请求 |
| 音频 | MP3, WAV | 最长 80 秒/请求 |
| 视频 | MP4, MOV | 最长 128 秒/请求 |

### 2.3 关键特性

- **Interleaved Input**: 一次请求传入 文本+图片 → 返回一个融合语义的向量
- **Task Type**: `RETRIEVAL_DOCUMENT`（索引端）/ `RETRIEVAL_QUERY`（查询端）
- **Matryoshka**: 可截断至任意维度（768 维 MTEB 67.99，几乎无损）

### 2.4 定价

| 层级 | 文本 | 图片 | 音频 |
|------|------|------|------|
| 免费层 | 免费（有速率限制） | 免费 | 免费 |
| 付费层 | $0.20/M tokens | $0.45/M tokens | $6.50/M tokens |

### 2.5 隐私政策（P0 关键）

| 维度 | 免费层 | 付费层 |
|------|--------|--------|
| 数据用于训练 | **是** | 否 |
| 人工可能审阅 | **是** | 仅违规检测 |
| 数据保留 | 不限期 | 55 天 |
| ZDR（零数据留存） | 不可用 | 可申请 |

**架构决策**: 默认要求付费层。免费层用户必须手动编辑配置文件写入 `privacy_tier = "free"` 才能启动。

### 2.6 成本估算

```
场景A：1000 篇笔记，少量附件
  全量索引: ~$0.24（一次性）
  每月增量: ~$0.01-0.03

场景B：5000 篇笔记，中等附件（200图片+50PDF）
  全量索引: ~$1.20（一次性）
  每月增量: ~$0.05-0.15
```

---

## 3. 系统架构

### 3.1 整体数据流

```
Obsidian Vault
     │
     ▼
[Vault 解析器] ─── 读取 .obsidian/app.json 获取附件配置
     │
     ├── 文本 → Heading→段落→句子递进切分 (800 token 阈值)
     ├── 图片 → 整图 embedding + interleaved(图+上下文文字)
     ├── PDF  → 5 页滑动窗口 (1 页重叠)
     ├── 音频 → VAD 切割至 ~70 秒
     └── 附件去重 → SHA-256 content hash
     │
     ├──────────────────────────────────────────┐
     ▼                                          ▼
[Gemini Embedding 2]                    [Layer 1 K-atom 提取]（Phase 1c）
  付费层，768 维                          wikilinks + tags + frontmatter + aliases
     │                                          │
     ▼                                          ▼
[LanceDB: chunks 表]                   [LanceDB: atoms 表 + relations 表]
  chunk 向量 + 元数据列                    K-atom 向量 + 属性 + 关系
     │                                          │
     └──────── relations 表关联 ────────────────┘
               同一 DB 内关联，事务一致
     │
     ▼
[MCP Server]
     ├── rag_search(query, reasoning_mode?)
     ├── rag_sync(mode=incremental|full, dry_run?)
     ├── rag_status()
     ├── extract_atoms(mode=refresh|enhance)     ← Phase 1.5
     ├── atom_query(name, relation_depth?)       ← Phase 1.5
     ├── atom_relate(source, target, type)       ← Phase 1.5
     ├── inspect_storage_stats()                 ← 诊断
     ├── query_metadata(table, where)            ← 诊断
     └── optimize_storage()                      ← 诊断
```

### 3.2 MCP Server 接口

#### `rag_search`

```json
{
  "tool": "rag_search",
  "params": {
    "query": "Docker 怎么部署",
    "top_k": 5,
    "modality_filter": ["text", "image"],
    "note_filter": ["projects/*"],
    "reasoning_mode": false
  }
}
```

返回：命中的 chunk 列表 + 动态拉取的相邻 chunk 上下文。

**`reasoning_mode` 增强搜索（Phase 1.5+）：**

算法路径：
- **路径 A（chunk 驱动，Phase 1.5）**: query → 向量搜索 chunks → 查 chunks 关联的 atoms → 沿关系遍历 → 拉取更多 chunks → 合并返回
- **路径 B（概念驱动，Phase 2）**: query → 向量搜索 atoms → 沿关系遍历 → 找关联 chunks → 与路径 A 合并 + RRF 融合

门控行为：
- Feature Flag `k_atom_search_boost` **OFF** 时：`reasoning_mode` 参数被静默忽略
- Flag **ON** + `reasoning_mode=true`：启用路径 A 增强

#### `rag_sync`

```json
{
  "tool": "rag_sync",
  "params": {
    "mode": "incremental",
    "dry_run": false
  }
}
```

- `dry_run=true`：只扫描变更，返回待处理文件列表 + 预估 API 成本，**不执行任何写入**
- `dry_run=false`（默认）：执行完整同步

返回：同步结果 + 进度 + Checkpoint 状态（支持断点续传）。

#### `rag_status`

返回：索引统计、健康状态、最近同步信息、成本统计、错误摘要、`fragment_count`（碎片数）、`db_size_mb`（数据库大小）、`last_sync_at`（最近同步时间）。

#### `extract_atoms`（Phase 1.5）

```json
{
  "tool": "extract_atoms",
  "params": {
    "note_path": "projects/my-project.md",
    "mode": "enhance"
  }
}
```

- `refresh`：重跑 Layer 1 确定性提取（aliases/frontmatter 变更后修正）
- `enhance`：跑 Layer 2 LLM 提取，补充隐含实体和关系

返回示例：
```json
{
  "atoms_created": 12,
  "atoms_updated": 3,
  "relations_created": 18,
  "atoms": [
    {"id": "uuid-1", "name": "Qdrant", "ontology_type": "tool", "confidence": 1.0},
    {"id": "uuid-2", "name": "向量搜索", "ontology_type": "concept", "confidence": 0.87}
  ]
}
```

#### `atom_query`（Phase 1.5）

```json
{
  "tool": "atom_query",
  "params": {
    "name": "Qdrant",
    "ontology_type": "tool",
    "relation_depth": 2
  }
}
```

内部自动归一化（大小写不敏感）。

返回示例：
```json
{
  "atom": {"id": "uuid-1", "name": "Qdrant", "ontology_type": "tool", "attributes": {"type": "向量数据库"}},
  "relations": [
    {"target": {"id": "uuid-3", "name": "Docker"}, "type": "related_to", "depth": 1},
    {"target": {"id": "uuid-4", "name": "RAG系统"}, "type": "part_of", "depth": 1}
  ],
  "source_notes": ["tools/qdrant.md", "projects/obsidian-rag.md"]
}
```

#### `atom_relate`（Phase 1.5）

```json
{
  "tool": "atom_relate",
  "params": {
    "source_atom_id": "uuid-1",
    "target_atom_id": "uuid-2",
    "relation_type": "is_a",
    "weight": 0.9
  }
}
```

#### `query_metadata`（诊断）

```json
{
  "tool": "query_metadata",
  "params": {
    "table": "chunks",
    "where": "vault_path LIKE 'projects/%'",
    "columns": ["vault_path", "chunk_type", "char_count"],
    "limit": 20
  }
}
```

**安全约束（防注入）：**
- `table` 参数白名单校验：只允许 `chunks` | `atoms` | `relations` | `sync_state`
- `where` 参数列名白名单：只允许各表已定义的列名出现（正则提取标识符后校验）
- `where` 禁止关键字：`DROP` / `DELETE` / `UPDATE` / `INSERT` / `ALTER` / `;`（大小写不敏感）
- `columns` 参数列名白名单校验，永远排除 `vector` 列
- 输入不合规 → 返回明确错误信息，不执行查询

```python
ALLOWED_TABLES = {"chunks", "atoms", "relations", "sync_state"}
FORBIDDEN_KEYWORDS = re.compile(r'\b(DROP|DELETE|UPDATE|INSERT|ALTER)\b|;', re.IGNORECASE)

def validate_query_metadata(table: str, where: str | None, columns: list[str] | None):
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Table '{table}' not allowed. Allowed: {ALLOWED_TABLES}")
    if where and FORBIDDEN_KEYWORDS.search(where):
        raise ValueError("where clause contains forbidden keywords")
    if columns and "vector" in columns:
        raise ValueError("vector column is excluded from query_metadata")
    # 列名白名单校验：提取 where 中的标识符，校验是否在表 schema 内
```

永远排除 vector 列，避免输出爆炸。

---

## 4. Chunking 策略

### 4.1 文本 Chunking

**递进式切分规则（Token 计数，非字数）：**

| 优先级 | 规则 | 说明 |
|--------|------|------|
| 1 | Heading (H1-H6) 边界 | 每个 Heading 下内容作为初步单元 |
| 2 | 段落（空行分隔） | 单个 Heading 下超过 800 tokens 时 |
| 3 | 句子 | 单个段落超过 800 tokens 时，100-200 tokens overlap |
| 4 | 特殊结构保持完整 | 代码块、表格、Callout 作为原子 chunk |

**Token 计数**: 使用 `tiktoken` 或等效 tokenizer，避免中英文混合失准。

### 4.2 图片 Chunking

两层索引策略：

1. **整图独立 embedding** — 图片本身作为一个 chunk
2. **Interleaved embedding** — 图片 + 笔记中的上下文文字 → 一个融合向量

去重：同一张图被多篇笔记引用时，整图 embedding 只做一次（SHA-256 hash 去重），但每个引用上下文的 interleaved embedding 各自独立。

### 4.3 PDF Chunking

- 5 页为一个 batch，1 页重叠（第 1-5 页、第 5-10 页...）
- 超过 30 页的 PDF：仅索引前 30 页，`rag_status` 中报告 warning

### 4.4 音频 Chunking

- VAD（Voice Activity Detection）在自然停顿处切割
- 目标段长 60-70 秒（留余量给 80 秒 API 限制）
- Phase 1 不做转录双轨索引（Phase 2 考虑）

---

## 5. LanceDB 存储层

> v3.0 核心变更：LanceDB 嵌入式数据库统一替换 Qdrant Docker + SQLite，消除三存储一致性问题。

### 5.1 为什么选 LanceDB

| 维度 | Qdrant + SQLite（v2.3） | LanceDB（v3.0） |
|------|------------------------|-----------------|
| 部署 | Docker 容器 + SQLite 文件 | `pip install lancedb`，零部署 |
| 存储系统 | 3 个（2 Qdrant collections + 1 SQLite DB） | 1 个（同一 LanceDB 实例） |
| 一致性 | 最终一致（三存储写入协议） | 事务一致（单 DB） |
| Hybrid Search | Phase 2 才有，需配置 sparse vector | 内置 BM25 + 向量混合 |
| 版本控制 | 无 | Lance 格式自带 MVCC |
| 备份 | Qdrant snapshot + SQLite backup | 复制文件夹 |
| 关系查询 | 跨系统（Qdrant 无 JOIN） | 同 DB 内 SQL-like where |
| 用户门槛 | 需要 Docker | 纯 Python，pip install 即可 |

### 5.2 存储路径

```
默认路径（XDG 规范）：
  ~/.local/share/obsidian-rag-mcp/lance_db/

环境变量覆盖：
  OBSIDIAN_RAG_DB_PATH=/custom/path

目录结构：
  lance_db/
  ├── chunks.lance/          # S 空间：文档 chunk 向量 + 元数据
  ├── atoms.lance/           # K 空间：K-atom 向量 + 属性
  ├── relations.lance/       # K 空间：实体间关系
  ├── sync_state.lance/      # 同步状态（文件级）
  ├── sync_checkpoint.lance/ # 同步 Checkpoint（任务级）
  ├── _meta.lance/           # Schema 版本管理
  ├── atom_community.lance/  # Phase 2：社区归属
  ├── community_summary.lance/ # Phase 2：社区摘要
  └── .mcp_server.lock       # 进程锁
```

> **决策**：不放在 `.obsidian/` 内，避免与 Obsidian 同步冲突（iCloud/Dropbox）。不放在 Vault 目录内，避免被 Obsidian 索引。

### 5.3 表结构定义（PyArrow Schema）

#### Chunks 表（S 空间）

```python
CHUNKS_SCHEMA = pa.schema([
    # 身份
    pa.field("id",                pa.utf8()),          # chunk 唯一 ID
    pa.field("vault_path",        pa.utf8()),          # 源文件在 Vault 中的路径
    pa.field("chunk_index",       pa.int32()),         # 在源文件中的序号

    # 内容
    pa.field("content",           pa.utf8()),          # 原始文本
    pa.field("content_tokenized", pa.utf8()),          # jieba 预分词（Phase 2，初始为空）
    pa.field("char_count",        pa.int32()),         # 字符数
    pa.field("content_hash",      pa.utf8()),          # SHA-256（变更检测 + 去重）

    # 上下文
    pa.field("heading_path",      pa.utf8()),          # Markdown 标题路径 "H1 > H2 > H3"
    pa.field("chunk_type",        pa.utf8()),          # text | code | table | list | image | interleaved

    # 元数据
    pa.field("frontmatter_tags",  pa.list_(pa.utf8())),# YAML frontmatter 标签
    pa.field("tag_hierarchy",     pa.list_(pa.utf8())),# 嵌套 tag 展开 ["dev", "dev/python"]
    pa.field("folder_path",       pa.utf8()),          # 笔记所在文件夹
    pa.field("is_moc",            pa.bool_()),         # 是否为 MOC 笔记
    pa.field("outgoing_links",    pa.list_(pa.utf8())),# [[]] 链接目标笔记列表
    # knowledge_element_ids 已移除 — chunk↔atom 关系统一走 relations 表（ADR #37）
    pa.field("created_at",        pa.utf8()),          # ISO 8601
    pa.field("updated_at",        pa.utf8()),

    # 向量
    pa.field("vector",            pa.list_(pa.float32(), 768)),
])
```

#### Atoms 表（K 空间）

```python
ATOMS_SCHEMA = pa.schema([
    # 身份
    pa.field("id",                pa.utf8()),          # atom 唯一 ID
    pa.field("name",              pa.utf8()),          # 原始名称
    pa.field("name_normalized",   pa.utf8()),          # name.strip().lower()
    pa.field("ontology_type",     pa.utf8()),          # unclassified|concept|entity|process|tool|project|tag

    # 内容
    pa.field("content",           pa.utf8()),          # 知识元描述（嵌入用）
    pa.field("content_tokenized", pa.utf8()),          # Phase 2

    # 属性（六空间 Am）
    pa.field("attributes_json",   pa.utf8()),          # JSON 字符串，如 {"type": "向量数据库"}
                                                        # Phase 1 可为空字符串 "{}"，Phase 1.5 LLM 提取填充

    # 来源 — source_chunk_ids 和 vault_paths 已移除，统一通过 relations 表反查（ADR #37）
    # 查询 atom 来源：relations.where("target_id = ? AND target_type = 'atom' AND source_type = 'chunk'")

    # 质量
    pa.field("confidence",        pa.float32()),       # 0.0~1.0
    pa.field("extraction_method", pa.utf8()),          # deterministic | llm

    # 元数据
    pa.field("tags",              pa.list_(pa.utf8())),
    pa.field("created_at",        pa.utf8()),
    pa.field("updated_at",        pa.utf8()),

    # 向量
    pa.field("vector",            pa.list_(pa.float32(), 768)),
])
```

**去重策略**: `(name_normalized, ontology_type)` 全局唯一。Phase 1 精确匹配 `.strip().lower()`；Phase 1.5 fuzzy 合并（向量相似度 >0.95 提示用户确认）。

**K-atom 嵌入内容**: `embed(f"{atom.name}: {atom.ontology_type}. {atom.attributes_json}")`。

**Aliases 处理**: Layer 1 提取 `[[Docker Engine]]` 时，查找目标笔记的 frontmatter `aliases`，合并到同一 K-atom。

#### Relations 表

```python
RELATIONS_SCHEMA = pa.schema([
    pa.field("id",              pa.utf8()),
    pa.field("source_id",       pa.utf8()),          # 源实体 ID
    pa.field("target_id",       pa.utf8()),          # 目标实体 ID
    pa.field("source_type",     pa.utf8()),          # "chunk" | "atom"
    pa.field("target_type",     pa.utf8()),          # "chunk" | "atom"
    pa.field("relation_type",   pa.utf8()),          # links_to|is_a|part_of|related_to|co_occurs|supports|contradicts|elaborates
    pa.field("weight",          pa.float32()),       # 0.0~1.0
    pa.field("created_at",      pa.utf8()),
])
```

> **决策**：`source_type`/`target_type` 支持 chunk↔atom、atom↔atom 等异构关系。

#### Sync State 表

```python
SYNC_STATE_SCHEMA = pa.schema([
    pa.field("file_path",       pa.utf8()),          # Vault 内文件路径（主键）
    pa.field("content_hash",    pa.utf8()),          # 文件内容哈希
    pa.field("last_synced_at",  pa.utf8()),
    pa.field("chunk_count",     pa.int32()),
    pa.field("atom_count",      pa.int32()),
    pa.field("status",          pa.utf8()),          # synced | pending | error
    pa.field("error_message",   pa.utf8()),
])
```

#### Sync Checkpoint 表

```python
SYNC_CHECKPOINT_SCHEMA = pa.schema([
    pa.field("sync_id",               pa.utf8()),          # 同步任务 ID（多行共享）
    pa.field("batch_index",           pa.int32()),         # 当前 batch 序号（每 batch 一行）
    pa.field("total_batches",         pa.int32()),
    pa.field("status",                pa.utf8()),          # in_progress | committed | failed
    pa.field("started_at",            pa.utf8()),
    pa.field("committed_at",          pa.utf8()),
    pa.field("processed_file_paths",  pa.list_(pa.utf8())),# 仅本 batch 处理的文件
    pa.field("written_chunk_ids",     pa.list_(pa.utf8())),# 仅本 batch 写入的 chunk IDs
    pa.field("written_atom_ids",      pa.list_(pa.utf8())),# 仅本 batch 写入的 atom IDs
    pa.field("written_relation_ids",  pa.list_(pa.utf8())),# 仅本 batch 写入的 relation IDs
])
# 设计：每个 batch 完成后插入一行（而非更新同一行追加列表）
# 崩溃恢复：查询 sync_id 下所有 status != 'committed' 的行，按各行 IDs 精确清理
# 优势：避免 5000 文件全量同步时单行 processed_file_paths 膨胀至 250KB+
```

#### Meta 表

```python
META_SCHEMA = pa.schema([
    pa.field("key",             pa.utf8()),          # "schema_version" | "created_at" | ...
    pa.field("value",           pa.utf8()),
])
```

#### Atom Community 表（Phase 2）

```python
ATOM_COMMUNITY_SCHEMA = pa.schema([
    pa.field("atom_id",              pa.utf8()),
    pa.field("community_id",         pa.utf8()),
    pa.field("level",                pa.int32()),         # 社区层级（0=最细粒度）
    pa.field("is_primary",           pa.bool_()),         # Leiden 硬划分 = true，软归属 = false
    pa.field("affiliation_strength", pa.float32()),       # 软归属强度（0.0~1.0），硬划分为 1.0
    pa.field("updated_at",           pa.utf8()),
])
```

#### Community Summary 表（Phase 2）

```python
COMMUNITY_SUMMARY_SCHEMA = pa.schema([
    pa.field("community_id",        pa.utf8()),
    pa.field("level",               pa.int32()),
    pa.field("summary",             pa.utf8()),           # LLM 生成的社区主题摘要
    pa.field("member_count",        pa.int32()),
    pa.field("vector",              pa.list_(pa.float32(), 768)),  # 摘要 embedding
    pa.field("updated_at",          pa.utf8()),
])
```

### 5.4 `is_moc` 检测逻辑

（优先级递减）：
1. frontmatter 含 `type: moc` 或 `moc: true` → 确定
2. 文件名含 "MOC" 或 "Map of Content"（不区分大小写）→ 确定
3. outgoing_links 数量 ≥ 10 且正文文字 < 500 字 → 启发式

### 5.5 链接关系

- **只存 `outgoing_links`**，不存 `incoming_links`
- 查询 "谁链接了笔记 B"：反查 relations 表 `where("target_id = ? AND relation_type = 'links_to'")` + JOIN chunks
  - **不用** `outgoing_links LIKE '%B%'` — LanceDB 对 `pa.list_(pa.utf8())` 列不支持 LIKE 操作，运行时会报错
  - 备选方案：如果 LanceDB 未来支持 `array_has_any(outgoing_links, ['B'])`，可切回 chunks 表直接查询
- 伪 GraphRAG：命中 chunk → 查 relations 表 `source_type='chunk' AND relation_type='links_to'` → 拉取关联笔记的 top chunk → 合并上下文

### 5.6 LanceStorageManager

```python
class LanceStorageManager:
    """统一存储管理器 — 管理所有表的生命周期"""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or self._default_path()
        # XDG 默认: ~/.local/share/obsidian-rag-mcp/lance_db/
        # 环境变量: OBSIDIAN_RAG_DB_PATH

    async def initialize(self):
        """MCP Server 启动时调用"""
        # 1. 进程锁（portalocker，跨平台）
        # 2. 打开/创建 LanceDB
        # 3. 确保 8 张表存在
        # 4. Schema 版本校验（_meta 表）
        #    - 版本匹配 → 继续
        #    - 版本不匹配 → 清表 + 重建空表（不触发全量 sync，下次 rag_sync 自然重建）
        #    - 日志输出 WARNING："Schema 版本变更 v{old}→v{new}，数据已清除，请运行 rag_sync"
        # 5. 崩溃恢复（检查未完成的 Checkpoint）
        # 6. Checkpoint 清理（清除已 committed 的历史 Checkpoint 行）

    async def shutdown(self):
        """MCP Server 关闭时调用"""
        # 1. Compaction（碎片整理）
        # 2. 释放进程锁

    async def health_check(self) -> dict: ...
    async def storage_stats(self) -> dict: ...
```

### 5.7 并发安全

- **进程锁**：`portalocker.lock(f, portalocker.LOCK_EX | portalocker.LOCK_NB)`（跨平台文件锁库，Unix 用 fcntl、Windows 用 msvcrt，无需 platform 判断）
- **单写多读**：LanceDB 支持并发读，写入串行化
- **Rayon 线程限制**：`RAYON_NUM_THREADS = max(cpu_count // 2, 1)`，避免与 MCP Server 争抢 CPU

---

## 6. K-atom 知识元架构

> 源自王延章六空间理论中 K-空间（知识元空间）的工程化实现。
> 定位：六空间理论是**设计语言**，不是系统规格；是**决策辅助**，不是决策本身；是**方向感**，不是路线图。

### 6.1 理论基础与工程边界

**六空间模型（K/S/D/F/I/O）辩论结论：**

| 空间 | 理论定义 | 工程采纳 | 理由 |
|------|----------|----------|------|
| K（知识元空间） | 知识元三元组 (Nm, Am, Rm) | ✅ 采纳 | 与知识图谱天然同构，映射为 LanceDB atoms 表 |
| S（元数据空间） | 知识元的描述性元数据 | ⚠️ 已有 | LanceDB chunks 表的列即 S 空间 |
| D（数据空间） | 原始数据存储 | ⚠️ 已有 | Obsidian Vault 即 D 空间 |
| F（形式模型空间） | 数学形式化表达 | ❌ 延期 | 过度工程，等有真实需求再考虑 |
| I（实体模型空间） | 可执行的实例化模型 | ❌ 延期 | 同上 |
| O（操作算子空间） | 操作算子集合 | ⚠️ 已有 | MCP tools 已是天然的 O 空间 |

**三个不可谈判条件：**

1. **罗盘≠KPI** — 六空间作为设计思维工具，不作为功能验收标准
2. **数据驱动诊断** — 所有 K-atom 功能必须有可量化的效果指标
3. **免疫系统前置** — Feature Flags 控制所有六空间扩展功能，默认 OFF

### 6.2 ontology_type 枚举

关系是边不是节点，不设 "relation" 类型：

| 类型 | 含义 | 推断来源 |
|------|------|----------|
| unclassified | 未分类（Layer 1 默认值） | `[[wikilink]]` 无 frontmatter |
| concept | 抽象概念 | Layer 2 LLM 推断 |
| entity | 具体实体（工具/人/组织） | Layer 2 LLM 或 frontmatter |
| process | 流程/方法/操作 | Layer 2 LLM 推断 |
| tool | 软件工具/技术 | frontmatter `type: tool` |
| project | 项目 | frontmatter `type: project` |
| tag | 标签 | Layer 1 确定性（`#tag`） |

### 6.3 K-atom 提取管线

**两层提取策略：**

| 层级 | 方法 | 输入 | 输出 | 准确率 |
|------|------|------|------|--------|
| Layer 1 | 确定性提取 | `[[wikilinks]]`、`#tags`、YAML frontmatter | K-atom (高置信度) | ~99% |
| Layer 2 | LLM 辅助提取 | Chunk 文本 → Claude Haiku | K-atom (需验证) | ~85% |

- Layer 1 在 Phase 1c 同步流程中附带执行，零额外 API 成本
- **Aliases 处理**：查找目标笔记的 frontmatter `aliases`，合并到同一 K-atom
- Layer 2 在 Phase 1.5 启用，通过 `extract_atoms(mode="enhance")` 按需触发
- 所有 LLM 提取结果标记 `extraction_method="llm"` + `confidence` 分数

### 6.4 Repository Pattern

```python
@dataclass(frozen=True)
class SearchQuery:
    """统一搜索请求 — 协议层唯一的搜索入参"""
    vector: list[float] | None = None       # 向量搜索
    text: str | None = None                 # 全文搜索（Phase 2）
    top_k: int = 10
    where: str | None = None                # SQL-like 过滤
    distance_threshold: float | None = None
    vector_weight: float = 0.7              # Hybrid 时向量权重

    def __post_init__(self):
        if self.vector is None and self.text is None:
            raise ValueError("至少提供 vector 或 text 之一")


class ChunkRepository(Protocol):
    async def search(self, query: SearchQuery) -> list[Chunk]: ...
    async def upsert(self, chunk: Chunk) -> None: ...
    async def upsert_batch(self, chunks: list[Chunk]) -> None: ...
    async def delete_by_vault_path(self, vault_path: str) -> None: ...
    async def count(self) -> int: ...


class AtomRepository(Protocol):
    async def search(self, query: SearchQuery) -> list[KnowledgeAtom]: ...
    async def get_by_id(self, atom_id: str) -> KnowledgeAtom | None: ...
    async def get_by_ids(self, atom_ids: list[str]) -> list[KnowledgeAtom]: ...
    async def upsert(self, atom: KnowledgeAtom) -> None: ...
    async def upsert_batch(self, atoms: list[KnowledgeAtom]) -> None: ...
    async def delete(self, atom_id: str) -> None: ...
    async def count(self) -> int: ...


# 实现：
# - InMemoryChunkRepository / InMemoryAtomRepository — 测试用
# - LanceChunkRepository / LanceAtomRepository       — 生产用
```

> **决策**：`SearchQuery` 用 frozen dataclass 而非 Pydantic。这是内部协议对象，不需要序列化，零依赖更快。

搜索分发逻辑：
- `vector` only → `_vector_search()`
- `text` only → `_fts_search()`（Phase 2）
- 两者都有 → `_hybrid_search()` → RRF（Reciprocal Rank Fusion）融合

### 6.5 Feature Flags

```toml
# config.toml
[features]
k_atom_extraction = false      # Layer 1 确定性提取
k_atom_llm_extraction = false  # Layer 2 LLM 提取
k_atom_search_boost = false    # 搜索时 K-atom 关系增强
ontology_type_filter = false   # 按本体类型过滤搜索结果
```

所有六空间扩展功能默认关闭。用户按需启用。

---

## 7. 同步机制

### 7.1 变更检测

```
文件系统快照对比：
  1. 遍历 vault/*.md + 附件目录
  2. 对比 mtime → 只有变化的才计算 content hash
  3. 与 LanceDB sync_state 表对比 → 识别 新增/修改/删除
```

### 7.2 同步粒度

- **文本笔记**: 笔记级重索引（删除旧 chunks → 重新 chunking → upsert）
- **附件**: content hash 变化才重新 embedding
- **引用关系变化**: 只更新引用方笔记的 outgoing_links

### 7.3 Checkpoint 机制

```python
class SyncPipeline:
    BATCH_SIZE = 20  # 每批处理文件数

    async def sync(self, changed_files: list[Path], dry_run: bool = False):
        if dry_run:
            return {"files_to_process": len(changed_files), "estimated_cost": ...}

        sync_id = uuid4()
        batches = chunked(changed_files, self.BATCH_SIZE)

        for i, batch in enumerate(batches):
            # 每 batch 插入一行 Checkpoint（status=in_progress）
            chunk_ids, atom_ids, relation_ids, errors = await self._process_batch(batch)
            # _process_batch 内部：单文件 embedding 失败 → 跳过该文件，标记 sync_state status=error
            # processed_file_paths 记录所有尝试的文件（含失败的）
            # 更新本行 Checkpoint（written_*_ids + status=committed）

        # 全部完成 → 清理历史 committed Checkpoint 行（ADR #41）

    async def recover_if_needed(self):
        """启动时调用 — 处理崩溃残留"""
        # 查找最近一次未完成的 sync_id
        incomplete = get_incomplete_sync()
        if incomplete:
            # 查询该 sync_id 下所有 batch 行
            batches = get_batches_by_sync_id(incomplete.sync_id)
            for batch in batches:
                # 精确清理：每行记录的 IDs 独立删除（三表全清）
                delete_by_ids("chunks", batch.written_chunk_ids)
                delete_by_ids("atoms", batch.written_atom_ids)
                delete_by_ids("relations", batch.written_relation_ids)
```

> **决策**：精确清理（按 ID 删除）而非全表版本回滚。`table.restore(version)` 会回滚整张表，可能丢失用户在同步期间的其他操作。

### 7.4 触发时机

- **定时轮询**: 每 10-15 分钟
- **手动触发**: `rag_sync` MCP tool

---

## 8. Obsidian 解析器

### 8.1 设计定位

> **RAG 语义提取，非 Obsidian 渲染复刻。**

提取的：文档结构、语义内容、元数据（frontmatter/tags/links）、附件引用。
不关心的：CSS 渲染、Dataview 动态结果、Canvas 布局、Templater 宏。

### 8.2 核心解析能力

| 语法 | 处理方式 |
|------|----------|
| YAML Frontmatter | 解析为 metadata，不进入 chunk 正文 |
| `[[wikilink]]` | 保留原文，目标 ID 存入 `outgoing_links` |
| `![[embed]]` | Phase 1 不展开，标记为 `embedded_links` |
| `![[image.png]]` | 解析附件路径，送入图片 embedding 流程 |
| `![[file.pdf]]` | 解析附件路径，送入 PDF embedding 流程 |
| `![[audio.mp3]]` | 解析附件路径，送入音频 embedding 流程 |
| `> [!callout]` | 剥离语法标记，保留文本，标记 chunk_type |
| ````code```` | 作为原子 chunk，标记 chunk_type = code |
| `$$LaTeX$$` | 保留原始文本 |
| `#tag` | 解析到 metadata.tags，正文中保留 |

### 8.3 附件路径解析

需读取 `.obsidian/app.json` 中的 `attachmentFolderPath` 和 `newLinkFormat`。

**Shortest path 模式**（Obsidian 默认）：
1. 在 vault 中全局搜索文件名
2. 文件名唯一 → 直接匹配
3. 文件名重复 → 用路径前缀消歧义

### 8.4 工具选型

- 优先使用 `obsidiantools`（支持 media_file_index）
- 附件反向引用（哪些笔记引用了某附件）需自实现
- Callout/代码块边界识别用正则

---

## 9. 隐私与安全

### 9.1 隐私层级

```python
class PrivacyTier(Enum):
    STANDARD = "standard"       # 付费层，55天日志，不用于训练
    MAXIMUM = "maximum"         # 付费层 + ZDR，零数据留存
    FREE_ACKNOWLEDGED = "free"  # 免费层，用户已知悉风险并确认
```

**默认 STANDARD**。免费层需手动配置确认。

### 9.2 首次启动

检测到免费层 API Key → 输出隐私警告 → Server 拒绝启动 → 用户必须在 `config.toml` 中写入 `privacy_tier = "free"` 才能继续。

### 9.3 API Key 安全

- 存储位置: `~/.obsidian-rag/config.toml`，权限 `600`
- 不得提交到版本控制
- 泄露后到 https://aistudio.google.com/apikey 重新生成

---

## 10. 项目结构

```
obsidian-rag-mcp/
├── src/obsidian_rag_mcp/
│   ├── __main__.py              # 入口，设置 RAYON_NUM_THREADS
│   ├── server.py                # MCP Server 定义
│   ├── config.py                # 配置管理、隐私层级、Feature Flags
│   │
│   ├── storage/                 # LanceDB 存储层
│   │   ├── schemas.py           # PyArrow Schema 定义（8张表）
│   │   ├── manager.py           # LanceStorageManager（生命周期+进程锁）
│   │   ├── compaction.py        # CompactionPolicy
│   │   └── index_manager.py     # 自适应索引管理
│   │
│   ├── repository/              # Repository 协议 + 实现
│   │   ├── protocols.py         # ChunkRepository / AtomRepository Protocol
│   │   ├── models.py            # KnowledgeAtom / Chunk / SearchQuery (dataclass)
│   │   ├── lance_chunk_repo.py  # LanceChunkRepository
│   │   ├── lance_atom_repo.py   # LanceAtomRepository
│   │   ├── lance_relation_repo.py
│   │   └── lance_sync_repo.py   # LanceSyncStateRepository
│   │
│   ├── vault/                   # Obsidian 解析器
│   │   ├── parser.py
│   │   ├── sections.py          # Heading/Callout/Code 切分
│   │   └── attachments.py       # 附件路径解析、去重
│   │
│   ├── chunking/                # 分块器
│   │   ├── text.py
│   │   ├── image.py
│   │   ├── pdf.py
│   │   └── audio.py
│   │
│   ├── embedding/               # Embedding 客户端
│   │   ├── client.py            # EmbeddingClient Protocol + Gemini 实现
│   │   └── batch.py             # 批量 embedding + 速率控制
│   │
│   ├── atoms/                   # K-atom 提取
│   │   ├── extractor.py         # Layer 1 确定性提取
│   │   └── llm_extractor.py     # Layer 2 LLM 辅助提取
│   │
│   ├── sync/                    # 同步管线
│   │   ├── pipeline.py          # SyncPipeline（含 Checkpoint）
│   │   └── text_processor.py    # jieba 预分词（延迟加载）
│   │
│   └── tools/                   # MCP Tool 定义
│       ├── search_tools.py      # rag_search
│       ├── sync_tools.py        # rag_sync / rag_status
│       ├── atom_tools.py        # extract_atoms / atom_query / atom_relate
│       └── diagnostic_tools.py  # inspect_storage_stats / query_metadata / optimize_storage
│
├── tests/
│   ├── test_storage_lifecycle.py
│   ├── test_schema_model_consistency.py
│   ├── test_chunk_repo.py
│   ├── test_atom_repo.py
│   ├── test_sync_pipeline.py
│   ├── benchmarks/
│   │   ├── generate_large_vault.py
│   │   └── bench_sync.py
│   └── ...
├── pyproject.toml
└── README.md
```

> **决策**：无 Dockerfile。LanceDB 嵌入式，无外部服务依赖。

---

## 11. Phase 规划

### Phase 1a — 真·MVP：文本搜索跑通（2 周）

| 功能 | 工作量 |
|------|--------|
| 隐私层级配置 + 强制确认 | 0.5 天 |
| LanceStorageManager + 8 张表 Schema | 1 天 |
| Obsidian 解析器（文本 + 附件路径识别） | 2 天 |
| 文本 Chunking（递进式切分） | 1.5 天 |
| Gemini Embedding 2 集成 | 1 天 |
| LanceChunkRepository | 0.5 天 |
| SyncPipeline + Checkpoint 机制 | 1.5 天 |
| MCP Server（rag_search / rag_sync / rag_status） | 2 天 |

> **里程碑**：文本笔记可搜索、增量同步可用、Checkpoint 崩溃恢复就绪。
> **第一个可验证测试**：`test_storage_lifecycle` — 初始化 → 健康检查 → 关闭 → 锁释放。

### Phase 1b — 多模态 + 健壮性（1.5 周）

| 功能 | 工作量 |
|------|--------|
| 图片 embedding + interleaved | 1.5 天 |
| PDF chunking（滑动窗口） | 1 天 |
| 音频 chunking（VAD） | 1 天 |
| 性能 benchmark | 1 天 |
| 诊断 MCP Tools（inspect_storage_stats / query_metadata / optimize_storage） | 1 天 |
| README + 文档 | 1 天 |

> **里程碑**：所有模态可搜索、诊断工具可用。

### Phase 1c — K-atom 基础（1.5 周）

| 功能 | 工作量 |
|------|--------|
| KnowledgeAtom 数据模型 + LanceAtomRepository | 1 天 |
| LanceRelationRepository | 0.5 天 |
| Layer 1 确定性提取（wikilinks / tags / frontmatter / aliases） | 1.5 天 |
| Feature Flags 配置体系 | 0.5 天 |
| K-atom 写入集成到 SyncPipeline | 2 天 |
| models↔schemas 一致性测试 | 0.5 天 |

> ⚠️ **K-atom 集成 SyncPipeline 的隐藏复杂度**：不是简单追加，需要改 Checkpoint 追踪逻辑（每 batch 追加 atom/relation IDs）、insert 前查询 `name_normalized` 去重、以及 atom 提取失败不阻塞 chunk 写入的错误隔离。原估 1 天调至 2 天。
> **里程碑**：K-atom 提取和存储就绪。同一 LanceDB 内事务一致，无需双写协议。
> **Phase 1c 完成时必做**：技术债盘点（debt inventory）— 审查所有延期决策（where:str 耦合、F/I 空间、结构化 Filter 等），确认每笔债务仍可控，无意外耦合（ADR #45）。

### Phase 1.5 — K-atom MCP Tools + 增强搜索（1-2 周）

| 功能 | 优先级 | 工作量 |
|------|--------|--------|
| `extract_atoms` MCP tool（refresh + enhance） | P0 | 1 天 |
| `atom_query` MCP tool | P0 | 1 天 |
| `atom_relate` MCP tool | P1 | 0.5 天 |
| `rag_search` reasoning_mode 路径 A | P1 | 1 天 |
| K-atom fuzzy 去重 | P1 | 1.5 天 |
| Layer 2 LLM 提取（Claude Haiku，含 prompt 工程） | P2 | 3 天 |

> reasoning_mode 路径 A 在 LanceDB 内完成，同 DB 查询，无跨系统延迟。

### Phase 2 — GraphRAG + 高级检索（Claude-Gemini 三轮辩论定稿）

#### 2a. 社区发现：Leiden + 软归属层

**核心方案**：对 atoms + relations 图运行 Leiden 算法（Louvain 的改进版，保证连通性），生成确定性的硬划分社区，再计算跨社区软归属。

**软归属计算**：atom 的邻居中属于其他社区的比例超过阈值 `T_affiliation` 时，创建 `is_primary=false` 的记录。解决知识元跨领域归属问题（如 "Docker" 同属 DevOps 和 ML Infra 社区）。

**渐进式激活**（避免冷启动噪音）：
- 系统监控图规模：`atom_count >= 200 AND average_degree >= 2.0`
- 不满足 → 仅运行全局向量检索（Phase 1 路径）
- 满足 → 自动触发首次 Leiden 划分，启用社区检索路径
- 社区重算触发条件：增量 sync 后图结构变化超过 `T_drift`（需实验标定，临时值 0.15）

**层级**：个人 Vault 规模（5000-20000 atoms）下 **2 级足够**（L0 细粒度社区 + L1 聚合社区）。

#### 2b. Ensemble Retrieval：并行通道 + RRF 融合

```
query
  ├── 通道 1: 全局向量搜索 chunks → top-30
  ├── 通道 2: BM25 全文搜索 chunks（jieba 中文分词）→ top-30
  └── 通道 3: 社区 Beam Search（社区摘要 → 社区内 atoms → 关联 chunks）→ top-20
      │
      ▼
  RRF 融合（Reciprocal Rank Fusion，基于排名，无需调参，k=60）
      │
      ▼
  Cross-encoder Re-ranking（预热加载，~150MB 内存）→ top-10
      │
      ▼
  返回最终结果
```

**Cross-encoder 部署要求**：MCP Server 启动时预热，额外 ~150MB 内存，并行通道总召回 50-60 → top-30 → 返回 top-10。

#### 2c. 其他 Phase 2 功能

- **HyDE 模式**：`rag_search(query_mode="hyde")` — LLM 先生成假设回答，用回答 embedding 检索
- **Query → Structured Filter**：LLM 将自然语言转成 metadata filter
- **Ontology 增强搜索**：reasoning_mode 路径 B（概念驱动）
- 本地 Embedding Fallback（nomic-embed-text）
- 音频转录双轨索引
- Late Chunking / ColBERT 实验
- Canvas 文件支持

> **参考文献**：
> - [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch) — 18 个 RAG 技术渐进式教程
> - [Deep GraphRAG](https://arxiv.org/abs/2601.11144) — 三层级社区图检索 + Beam Search + 多阶段 Re-ranking

### Phase 3（远期）

- 多 Vault 支持
- 多用户/多租户
- Web UI
- 实时文件监控（替代轮询）
- F/I 空间评估（仅在有真实用户场景驱动时考虑）

---

## 12. 开发顺序（DAG 依赖图）

```
models.py ◄════ 协同约束 ════► schemas.py
    │        (字段必须 1:1 对应，          │
    │         改一个必须改另一个，          │
    │         CI: test_schema_model_consistency.py)
    │                                     │
protocols.py ─────────────────────┐       │
    │                             │       │
manager.py ──────────┐            │       │
    │                │            │       │
    ▼                ▼            ▼       ▼
lance_chunk_repo   lance_atom_repo  lance_sync_repo  lance_relation_repo
    │                │               │
    └────────────────┼───────────────┘
                     │
                     ▼
              pipeline.py（SyncPipeline + Checkpoint）
                     │
                     ▼
              tools/*.py（MCP Tool 接入）
                     │
                     ▼
              __main__.py（启动流程）
```

---

## 13. 关键架构决策记录（ADR）

| # | 决策 | 理由 |
|---|------|------|
| 1 | 付费层 API 为默认 | 个人知识库隐私敏感，免费层数据可能被 Google 用于训练 |
| 2 | 768 维而非 3072 | MTEB 几乎无损（67.99 vs 68.17），存储节省 75% |
| 3 | LanceDB 替换 Qdrant + SQLite | 零部署、单存储事务一致、内置 Hybrid Search、消除三存储一致性问题 |
| 4 | 不存 incoming_links | 动态反查 outgoing_links 消除级联更新 |
| 5 | 不存父 Chunk | 检索时动态拉相邻 chunk，省存储省同步 |
| 6 | `![[embed]]` 不展开 | 避免循环引用，Phase 1 保留标记 |
| 7 | 笔记级重索引 | 避免 chunk 级 diff 的边界问题 |
| 8 | MCP Server 部署 | 共享基础设施，所有 Agent 可调用 |
| 9 | 解析器定位：语义提取非渲染复刻 | 不追求 100% Obsidian 兼容，聚焦 RAG 需要的信息 |
| 10 | Checkpoint + 精确清理 | 按 ID 删除崩溃残留，不做全表版本回滚（避免丢失用户其他操作） |
| 11 | 六空间理论仅采纳 K-空间 | F/I 空间过度工程，S/D/O 已被现有架构覆盖 |
| 12 | F/I 空间延期至有真实需求 | 数学形式化和可执行模型在个人知识管理场景无即时价值 |
| 13 | Feature Flags 控制所有扩展功能 | 默认 OFF，免疫系统前置 |
| 14 | Repository Protocol + Lance 单后端 | InMemory（测试）+ Lance（生产），不再需要双后端封装 |
| 15 | K-atom 两层提取策略 | Layer 1 确定性（零成本）+ Layer 2 LLM（按需），渐进式 |
| 16 | 六空间定位为设计语言 | 理论是罗盘不是 KPI，是方向感不是路线图 |
| 17 | Relations 表支持异构关系 | source_type/target_type 支持 chunk↔atom、atom↔atom |
| 18 | K-atom 按 (name_normalized, ontology_type) 全局去重 | 避免 N 篇笔记产生 N 个冗余实体 |
| 19 | ontology_type 去掉 "relation" | 关系是边不是节点 |
| 20 | Payload 增强：folder_path + tag_hierarchy + is_moc | 零成本，立即提升检索过滤能力 |
| 21 | Obsidian aliases 合并到同一 K-atom | 避免别名产生冗余实体 |
| 22 | 统一 SearchQuery dataclass | frozen=True 值对象，支持 vector/text/hybrid 三模式自动分发 |
| 23 | 存储路径遵循 XDG 规范 | 不放 .obsidian/ 内避免同步冲突，不放 Vault 内避免被 Obsidian 索引 |
| 24 | 时间字段用 UTF8 存 ISO 8601 | LanceDB timestamp 跨时区行为不一致，字符串更可控 |
| 25 | content_tokenized 从 Phase 1a 预留 | 避免 Phase 2 做 Schema 迁移，初始值为空字符串 |
| 26 | 不存 embedding_model 在数据行上 | 模型切换应触发全量重建，不应混存不同模型的向量 |
| 27 | 进程锁 portalocker | 跨平台防多实例并发写入，单写多读 |
| 28 | Phase 1 拆为 1a/1b/1c | 降低耦合风险，先验证核心链路 |
| 29 | 诊断 MCP Tools（3个） | inspect_storage_stats / query_metadata / optimize_storage，生产可观测性 |
| 30 | query_metadata 永远排除 vector 列 | 768 维浮点数组序列化会让响应爆炸 |
| 31 | outgoing_links 反查走 relations 表 | LanceDB 对 `list_` 列不支持 LIKE，relations 表查询可靠且已有索引 |
| 32 | Checkpoint 清理覆盖三表 | chunks + atoms + relations 全部按 ID 精确清理，防止崩溃后 relations 残留 |
| 33 | query_metadata 输入白名单校验 | 防 SQL 注入：表名白名单、列名白名单、禁止 DROP/DELETE 等关键字 |
| 34 | 进程锁用 portalocker 替代 fcntl | fcntl 是 Unix-only，portalocker 跨平台（Unix fcntl + Windows msvcrt） |
| 35 | Checkpoint 改多行设计（每 batch 一行） | 避免 5000 文件全量同步时单行 list 列膨胀至 250KB+ |
| 36 | models.py ↔ schemas.py 协同约束 + CI 校验 | 两处维护同一字段集，CI 测试强制校验一致性，防漂移 |
| 37 | 去掉 chunk↔atom 双向冗余字段 | chunks.knowledge_element_ids + atoms.source_chunk_ids/vault_paths 全部移除，统一走 relations 表 |
| 38 | Schema 统一用 ontology_type | atom_type → ontology_type，与 MCP 接口和文档一致 |
| 39 | 新增 attributes_json 列（六空间 Am） | JSON 字符串存储结构化属性，Phase 1 可为空 "{}"，Phase 1.5 LLM 填充 |
| 40 | Schema 迁移策略：清表 + 下次 sync 重建 | 版本不匹配时 initialize() 只清表建空 Schema，不触发全量重建 |
| 41 | Checkpoint 清理双时机 | sync 成功后清理已 committed 的历史行 + 崩溃恢复后清理 failed 行 |
| 42 | Embedding 失败：batch 内尽力而为 | 单文件 embedding 失败不阻塞整 batch，error 文件下次 sync 自动重试 |
| 43 | rag_sync dry_run 模式 | dry_run=true 只扫描变更返回预估，不执行写入 |
| 44 | 结构化 Filter 延期 Phase 2 | Phase 1 保持 where: str 直通 LanceDB |
| 45 | Phase 1c 后做技术债盘点 | 多个"可控小债务"累积时交互复杂度超线性增长 |
| 46 | 社区发现用 Leiden 而非 Louvain | Leiden 保证社区连通性，Louvain 可能产生断裂社区 |
| 47 | 2 级社区层级（非 3 级） | 个人 Vault 5000-20000 atoms 规模下 2 级足够，3 级过度 |
| 48 | 软归属层解决跨社区 atom 问题 | Leiden 硬划分 + 邻居比例计算软归属，避免重叠社区算法的不稳定性 |
| 49 | 社区功能渐进式激活 | atom_count >= 200 AND avg_degree >= 2.0 时自动启用，避免冷启动噪音 |
| 50 | RRF 作为默认融合策略 | 不同通道分数不可比（余弦 vs BM25 vs beam），RRF 只用排名，鲁棒无需调参 |
| 51 | Cross-encoder 启动时预热 | 消除首次请求冷启动延迟，额外 ~150MB 内存，纳入资源需求 |
| 52 | T_drift 阈值需实验标定 | embedding 模型敏感性不同，硬编码不可靠，标定前用保守临时值 0.15 |

---

## 14. 六空间理论集成备忘

> 本节记录 Claude-Gemini 辩论共识，供未来架构决策参考。

### 核心定位

**六空间理论是设计语言，不是系统规格；是决策辅助，不是决策本身；是方向感，不是路线图。**

### 从理论中提取的工程价值

1. **知识元三元组 (Nm, Am, Rm)** → K-atom 数据模型，映射为 LanceDB atoms 表
2. **知识粒度层级** → 指导 Chunking 粒度选择（句子 < 段落 < 章节 < 文档）
3. **默会知识显性化** → Layer 2 LLM 提取的理论基础

### 未采纳但值得关注的概念

- **六空间计算循环 (D→S→K→F→I→O→K→S→D)** — 未来知识自动化处理流水线参考
- **认知科学映射（第8章）** — 用户行为分析参考
- **AI 集成（第7章）** — "辅助而非替代人类认知"的立场有参考价值

### 触发 F/I 空间重新评估的条件

仅当以下 **任意一条** 成立时，才考虑实现 F/I 空间：
1. 用户明确要求对笔记中的数据进行自动建模/推理
2. K-atom 数量超过 10,000 且关系网络复杂度需要形式化约束
3. 出现需要可执行知识模型的真实用例（如 3D 物理推理、自动报告生成）

---

*— Hani · 无垠智穹*
