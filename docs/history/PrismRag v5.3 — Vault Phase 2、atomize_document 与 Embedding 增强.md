---
title: "PrismRag v5.3 — Vault Phase 2、atomize_document 与 Embedding 增强"
tags:
  - "#PrismRag"
  - "#GraphRAG"
  - "#architecture"
  - "#knowledge-graph"
  - "#design"
status: spec
created: 2026-05-05
last_audited: 2026-05-05
review_notes: "R1 P0×3（TOCTOU/target_title/embed_cache）+ P1 mentioned_in。R2 P0×5（scan_id+section_id / scan TTL / doc_sha stale / partially_applied 终态 / claim 去重）+ batch alloc。R3 P1×2（propose 幂等 key / step3+4 write_atomic 文档自证）+ P2 embed_cache GC + P3×2（SUPERSEDES 内存修正 / scan cache 持久化）。R4 KNOW-ID :06d + int排序 + 新§4.6 AtomicFileOp asyncio 双层锁 + ingest_file(skip_embed=True) + applied_pending_embed 状态 + D22/D23/D24。"
milestone: v5.3
related:
  - "设计细节/PrismRag v5.2 — EdgeClassifier 跨边分类与 inbox 工作流.md"
  - "设计细节/PrismRag v5.1 — mentions_symbol 跨命名空间链接设计.md"
  - "设计细节/PrismRag-Jei整合路线图.md"
  - "设计细节/atomize-document skill — Jei 文档原子化能力设计.md"
  - "设计细节/PrismRag Multimodal Embedding — 设计方向与路线图.md"
---

# PrismRag v5.3 — Vault Phase 2、atomize_document 与 Embedding 增强

> v5.2 完成了 EdgeClassifier 三档分类器 + inbox 工作流（TUI / MCP / CLI）。
> 图的生命周期管理到位，但图里的 vault 知识节点粒度粗、无稳定 ID、无类型化关系。
> v5.3 解决这三个问题，同时修复 embedding 的可靠性瓶颈。

---

## 一、v5.3 定位与边界

### 四个支柱

| # | 支柱 | 范围 |
|---|---|---|
| 0 | **收尾：Step 4 + Step 5** | ZenithLoom ingest 完成、Jei 端到端验证、旧代码清理 |
| 1 | **Vault Phase 2 — 知识节点数据模型** | REGISTRY、`knowledge_id`、`relations:` schema |
| 2 | **`atomize_document` MCP 工具族** | `atomize_scan / atomize_propose / atomize_apply` |
| 3 | **Embedding 增强** | 断点续传、GPU/CPU 感知（两个 namespace 共享同一模型，不做 per-namespace 切换）|

**范围边界**：v5.3 只做 PrismRag 仓库的改动。Jei/ZenithLoom 侧的 ROLE.md、PROTOCOL.md
更新不在本 sprint，通过 MCP tool 接口变化自然驱动。

---

## 二、前置条件 — Step 4 + Step 5（v5.3 正式任务开始前必须完成）

### Step 4 — 真实环境验证

执行顺序（GPU 空闲后）：

```bash
# 1. ZenithLoom ingest（GPU 必须空闲，qwen3-embedding:8b 才能加载）
cd /home/kingy/Foundation/PrismRag
set -a && source .env && set +a
nohup python3 -m prism_rag.cli ingest-code \
  --repo /home/kingy/Foundation/ZenithLoom \
  --data-dir /home/kingy/Foundation/ZenithLoom/.graph/code \
  --namespace code \
  > /tmp/prismrag-ingest-code.log 2>&1 &

# 2. 确认 ingest 完成
tail -f /tmp/prismrag-ingest-code.log

# 3. 运行 classify-edges
python3 -m prism_rag.cli classify-edges

# 4. 重启 MCP server（kill 旧进程后重启 Jei）
systemctl --user restart jei

# 5. 端到端测试（T1–T4）
```

**T1–T4 测试矩阵（来自整合路线图）：**

| 测试 | 内容 | 通过判据 |
|---|---|---|
| T1 连通性 | Jei 调用 `list_namespaces` | 返回 nimbus + code 两个命名空间 |
| T2 图查询 | Jei 回答一个跨越 vault + code 的问题 | 引用 `explain_node` 或 `trace_path` 结果 |
| T3 inbox 审核 | Jei 用 `list_pending_edges` 列出待审边 | 返回 classify-edges 后的 Tier 2 列表 |
| T4 写入路径 | 让 Jei 创建一个 test note | vault 里出现文件、audit JSONL 有记录、图有新节点 |

### Step 5 — 清理

| 任务 | 文件 / 位置 |
|---|---|
| 删除 Obsidian MCP 目录 | `ZenithLoom/mcp_servers/obsidian/`（整个目录） |
| 清理 dead field | `ZenithLoom/framework/schema/base.py` → 移除 `knowledge_result: str` |
| 更新相关测试 | `test_e2e_mcp.py` — 移除 knowledge_shelf fixture |

> Step 4 + Step 5 **不属于 v5.3 的 sprint 任务**，但必须在 v5.3 第一个 sprint 任务开始前完成。
> 单独 commit，不进 v5.3 tag。

---

## 三、支柱 1 — Vault Phase 2：知识节点数据模型

### 3.1 问题

当前 vault node 的粒度是**文件级**：一个 800 行的设计文档 = 一个 node。
这导致：
- 搜索结果粗糙（整篇文档出现，不是具体结论）
- 无法引用具体结论（引用路径会随文件改名断掉）
- 过时的结论无法被标记为 superseded（整个文件要么全动要么不动）
- EdgeClassifier 的 `mentions_symbol` 边无法指向文件内的具体 claim

### 3.2 目标形态

引入 **knowledge node** 概念——从长文档抽出的原子 claim，有：
- 稳定 ID：`KNOW-000001`（格式 `KNOW-NNNNNN`，6 位补零）
- 类型：`fact | concept | decision | rule | procedure | relation`
- 关系（typed edges）：`depends_on | refines | contradicts | references | supersedes`
- 反向索引：`mentioned_in` — 原始文档的路径

原始文档**不删除**，只在被抽取的段落处加 `> 详见 [[KNOW-XXXXXX]]` 引用。

### 3.3 数据模型变更

#### 3.3.1 新 `NodeKind` 值

```python
# prism_rag/store/graph.py
class NodeKind(str, Enum):
    ...
    KNOWLEDGE = "knowledge"     # 新增：原子 knowledge node
```

#### 3.3.2 新 Node 属性

```python
# 在 KnowledgeGraph._set_node_attrs 或 NodeAttr 里新增
knowledge_id: str | None = None     # e.g. "KNOW-000042"
knowledge_type: str | None = None   # fact | concept | decision | rule | procedure | relation
mentioned_in: list[str] = []        # 反向索引：原始文档路径列表
status: str = "active"              # active | superseded | draft
```

#### 3.3.3 新边类型 `RelationType`

```python
# prism_rag/store/graph.py
class RelationType(str, Enum):
    # 现有类型保留不变
    CONTAINS = "contains"
    MENTIONS_SYMBOL = "mentions_symbol"
    EMBEDDING_SIMILAR = "embedding_similar"
    # v5.3 新增
    DEPENDS_ON = "depends_on"       # A 以 B 为前提
    REFINES = "refines"             # A 是 B 的更精确版本
    CONTRADICTS = "contradicts"     # A 否定 B
    REFERENCES = "references"       # A 提及 B（最弱）
    SUPERSEDES = "supersedes"       # A 取代 B；B.status → superseded 由调用方（atomize_apply）显式写入 B.md frontmatter，vault_loader ingest 侧只读边不写状态
```

#### 3.3.4 REGISTRY

```
data_dir/
└── registry.json         # KNOW-ID 分配器
```

```json
{
  "schema": "v1",
  "namespace": "nimbus",
  "next_id": 43,
  "last_assigned": "KNOW-000042",
  "updated_at": "2026-05-05T10:00:00Z"
}
```

**分配规则**：
- `next_id` 单调递增，不复用（哪怕 node 被删）
- 格式化：`f"KNOW-{next_id:06d}"`；排序按整数值（`int(re.search(r'KNOW-(\d{6,})', kid).group(1))`），regex `KNOW-(\d{6,})` 兼容未来位数扩展
- 写入时用 `atomic_write`，读写时持文件锁（见 §4.6 AtomicFileOp 规约）防竞态

#### 3.3.5 vault frontmatter schema（约定，非强制校验）

```yaml
---
knowledge_id: KNOW-000042
title: "fresh_per_call 是每次调用重建 client 的决策"
type: decision           # fact | concept | decision | rule | procedure | relation
status: active           # active | superseded | draft
scope: PrismRag
created: 2026-05-05
relations:
  - type: depends_on
    target: KNOW-000039    # 决策依赖的前提
  - type: refines
    target: KNOW-000031
mentioned_in:
  - "设计细节/PrismRag v5.0 — 详细设计与任务分解.md"
---
```

### 3.4 REGISTRY MCP 工具

```python
@mcp.tool()
def alloc_knowledge_id(namespace: str = "") -> str:
    """分配下一个 KNOW-ID，原子写入 REGISTRY。
    返回：{"knowledge_id": "KNOW-000043", "registry_path": "..."}
    """
```

```python
@mcp.tool()
def list_knowledge_nodes(
    namespace: str = "",
    ktype: str = "",      # 过滤 type
    status: str = "active",
) -> str:
    """列出所有 KNOW-* 节点，支持按 type / status 过滤。"""
```

### 3.5 ingest 侧变更

`vault_loader.py` 在解析 frontmatter 时：
- 发现 `knowledge_id` 字段 → 设置 `node.kind = NodeKind.KNOWLEDGE`，记录 `knowledge_id` 属性
- 发现 `relations:` 列表 → 在图中写对应类型的边（`RelationType.DEPENDS_ON` 等）
- 发现 `mentioned_in:` → 写 `REFERENCES` 边，方向 **knowledge_node → source_document**（正向，knowledge 引用它所出自的文档）

**SUPERSEDES 内存修正**（图构建阶段完成后执行）：
- 遍历图中所有 SUPERSEDES 边：若 `A --SUPERSEDES--> B` 且 `A.status == active`，则内存中将 B.status 覆写为 `superseded`
- 必须校验 A.status == active：若 A 本身已废弃，不传递降级效果（防止死节点链式传播）
- 此修正仅作用于内存图，不写回 vault 文件（文件由 atomize_apply 负责显式更新）
- live_sha_set 副产物：vault_loader 在遍历所有文档时顺带输出 `{(node_id, content_sha)}` 集合，供 embed_cache GC 使用（零额外磁盘 I/O）

**embed_cache GC**（startup 时触发，在所有 namespace loader 完成后）：
- **Namespace 分离 sweep**：GC 按 node_id 前缀路由，不同 namespace 使用各自 loader 的 live_sha_set：
  - `nimbus::` 前缀 → vault_loader 输出的 live_sha_set
  - `code::` 前缀 → code_loader（ingest-code）输出的 live_sha_set；若 code_loader 未运行，**跳过 code namespace 的 GC**（保守策略，宁可保留过期条目，不可误删有效向量）
  - 其他未知前缀 → 跳过 GC
- 过期条目过滤后重写 embed_cache.jsonl（完整重写一次，后续 append 照常）
- 加载时若同一 node_id 出现多次，取 sha 最新的一条（last-wins 语义，对应最近一次内容变更）

现有无 `knowledge_id` 的 vault 文档继续以 `NOTE` kind 处理，**不做 migration**。

---

## 四、支柱 2 — `atomize_document` MCP 工具族

### 4.1 设计原则

**LLM 做判断，PrismRag 做搬运。**

`atomize_document` 的语义决策（哪些是 claim、什么类型、什么关系）由调用方的 LLM（Jei）完成。
PrismRag 只做机械步骤：读结构、分配 ID、原子写文件、更新图。

这对应三个独立 MCP 工具，形成一个三阶段协议：

```
[Jei LLM]                          [PrismRag]
   │                                    │
   ├──atomize_scan(path)──────────────► │ 读文档结构（sections, existing IDs, graph node）
   │ ◄── document_structure ───────────┤
   │                                    │
   │  (Jei 做语义判断：哪些段落值得原子化，  │
   │   什么类型，和哪些现有 node 有关系)    │
   │                                    │
   ├──atomize_propose(path, scan_id, claims)► │ 分配 KNOW-IDs, 写 proposal 文件
   │ ◄── {proposal_id, preview} ───────┤
   │                                    │
   │  (可选：人工审核 proposal 文件)       │
   │                                    │
   ├──atomize_apply(proposal_id)──────► │ 创建 knowledge 文件, patch 原文, 更新图
   │ ◄── {created, patched, graph} ────┤
```

### 4.2 `atomize_scan` — 读文档结构

```python
@mcp.tool()
def atomize_scan(path: str, namespace: str = "") -> str:
    """
    读取文档的结构，供 LLM 判断哪些段落值得原子化。

    返回：
    {
      "scan_id": "scan-2026-05-05T10-20-a3f9c2b1-rag-architecture",  # 唯一扫描 ID（含 UUID 短码防分钟级碰撞），propose 时必须携带
      "doc_sha": "sha256:abc123...",                          # 扫描时全文 SHA
      "path": "设计细节/rag-architecture-design.md",
      "node_id": "nimbus::设计细节/rag-architecture-design.md",
      "already_atomized": ["KNOW-000031", "KNOW-000042"],   # frontmatter 里已有的引用
      "sections": [
        {
          "section_id": "s_0",                # 稳定索引 ID，LLM 用此字段引用 section（见 §4.3）
          "heading": "三、核心架构",
          "level": 2,
          "line_start": 45,
          "line_end": 120,
          "char_count": 2340,
          "has_knowledge_refs": false,    # 是否已含 [[KNOW-XXXXXX]] 引用
          "first_200_chars": "..."       # LLM 用此做语义判断；content_snapshot 不在此返回（见实现要点）
        },
        ...
      ],
      "related_nodes": [                  # graph 中引用过这个文档的节点
        {"node_id": "...", "relation": "mentions_symbol", ...}
      ]
    }
    """
```

**实现要点**：
- 用现有 `ObsidianParser` / `markdown_ops.py` 解析 section 结构
- 扫描 frontmatter 里已有的 `knowledge_id` 引用，避免重复原子化
- 从 federated graph 查 `mentioned_in` 反向边
- 不做任何写操作，纯读
- scan 结果（含每个 section 的 `content_snapshot` 全文）持久化到 `data_dir/scan_cache/{scan_id}.json`；`scan_id` 格式：`scan-{timestamp_min}-{uuid4()[:8]}-{slug}`，UUID 短码防止同一文档在同一分钟内两次扫描导致碰撞
- MCP server 启动时加载 scan_cache/ 下所有文件，检查 TTL（24h），过期文件删除；重启后已有 scan 记录不丢失，LLM session 无需重新 scan
- **`content_snapshot` 不出现在 MCP 返回值中**：LLM 用 `first_200_chars` + `heading` 做语义判断已足够；snapshot 只由 `propose_impl` 通过 `scan_id + section_id` 做 O(1) 查找后注入 proposal，避免向 LLM 传输长文档 20 个 section × 50 行 ≈ 10,000 token 的无效数据
- `atomize_propose` 查不到对应 `scan_id`（不存在或已过期）时返回 `410 Gone`，要求重新 scan

### 4.3 `atomize_propose` — 写入 Proposal

```python
@mcp.tool()
def atomize_propose(
    path: str,
    namespace: str,
    scan_id: str,          # 必填：来自 atomize_scan 返回的 scan_id
    claims: list[dict],    # LLM 提供的候选清单（见下方 schema）
) -> str:
    """
    接收 LLM 的语义判断结果，分配 KNOW-IDs，写入 proposal 文件。

    scan_id 必填。atomize_propose_impl 用它从 server 端 scan cache 取出对应的
    section 元数据，按 section_id 注入 content_snapshot 到 proposal claim 中。
    LLM 无需自行转发 content_snapshot。若 scan_id 不存在或已过期（>24h），
    返回 410 Gone，要求重新调用 atomize_scan。

    claims 的每个元素（LLM 提供，source_section 改为 section_id）：
    {
      "title": "fresh_per_call 决策",           # < 30 字
      "type": "decision",                        # fact|concept|decision|rule|procedure|relation
      "section_id": "s_2",                       # 来自 atomize_scan 返回的 section_id（稳定索引）
      "body": "每次调用重建 client 是为了...",    # 提炼后的原子内容
      "relations": [
        {"type": "depends_on", "target_id": "KNOW-000039"},          # target_id → atomize_apply 写 frontmatter 时转为 target
        {"type": "refines", "target_title": "Gemini client 配置"}  # 没有 ID 时用 title，由 propose 解析后写为 target_id
      ],
      "confidence": "high"                       # high|medium（low 的不提交）
    }

    target_title 两级解析（atomize_propose_impl 内部执行）：
    1. 批内精确匹配：同一 proposal 内的 claim 先分配 ID，若 target_title 命中批内某 claim.title → 直接用分配到的 KNOW-ID
    2. 图内查找：
       a. 标准化精确：graph.find_knowledge_nodes(title.strip().lower()) → 唯一命中则用其 knowledge_id
       b. 语义 fallback：对图内所有 knowledge node 算 title embedding 相似度，
          score ≥ 0.90 且 top-1 唯一性（第二名 score < 0.85）→ 用 top-1 的 knowledge_id
    3. 仍未命中 → 写入 proposal claim 的 unresolved_relations 列表，不报错；
       apply 时跳过该边，audit 中记录；LLM 之后可补充正确 ID 重新 propose

    Claim 内容哈希去重（atomize_propose_impl 内部执行）：
    - 按 source_doc + section_id 查询当前文档已有 knowledge nodes
    - 对每个 claim 计算 sha256(claim.body + claim.section_id)
    - 命中已有 node 的哈希 → 该 claim 标记 status="duplicate"，跳过 ID 分配
    - 未命中 → 正常处理
    - 目的：部分 apply 失败后重新 scan→propose 时，已成功的知识点不会在图谱中重复出现

    幂等性：
    - propose_key = sha256(doc_sha + sorted([sha256(c.body + c.section_id) for c in claims]))
    - propose_key 写入 data_dir/propose_dedup_index.json（key → {proposal_id, status} 映射）
    - 若 propose_key 已存在，按 dedup_index 中的 status 分两路处理：
        a. status == "pending"（上次 propose 成功但尚未 apply）→ 直接返回已有 proposal_id，
           不分配新 ID，不写新文件（网络超时后 LLM 重试的正常路径）
        b. status == "applied_pending_embed" | "partially_applied"（已 apply）→
           返回 {"error": "already_applied", "proposal_id": "XXX"}，LLM 可直接跳过
    - dedup_index 永不主动清理（doc_sha 已绑定文档版本，文档内容变化时 key 自然失效；
      千级 Vault 下 index 文件 < 100KB，无膨胀风险）

    ID 分配：
    - 一次 fcntl.flock（通过 AtomicFileOp）→ 读 REGISTRY next_id → 分配 N 个连续 ID → 写回 → unlock
    - 若 propose 中途失败，已分配但未写入文件的 ID 成为空洞（D7 by-design），不做回滚；
      REGISTRY 的单调递增特性保证后续分配不会复用这些 ID

    返回：
    {
      "proposal_id": "atomize-2026-05-05T10-23-rag-architecture",
      "proposal_path": "data/atomize-proposals/pending/atomize-2026-05-05T10-23-rag-architecture.json",
      "assignments": [
        {"title": "fresh_per_call 决策", "knowledge_id": "KNOW-000043", "status": "assigned"},
        {"title": "已存在的知识点",       "knowledge_id": null,        "status": "duplicate"},
        ...
      ],
      "total_claims": 5,
      "skipped_duplicates": 1
    }
    """
```

**Proposal 文件格式**（`data/atomize-proposals/pending/<id>.json`）：

```json
{
  "id": "atomize-2026-05-05T10-23-rag-architecture",
  "source_path": "设计细节/rag-architecture-design.md",
  "namespace": "nimbus",
  "scan_id": "scan-2026-05-05T10-20-a3f9c2b1-rag-architecture",
  "created_at": "2026-05-05T10:23:00Z",
  "status": "pending",
  "doc_sha": "sha256:abc123...",
  "claims": [
    {
      "knowledge_id": "KNOW-000043",
      "title": "fresh_per_call 决策",
      "type": "decision",
      "section_id": "s_2",
      "source_section": "三、核心架构",   # 供人阅读，heading 文本
      "source_lines": [45, 78],
      "content_snapshot": "...",           # 由 propose_impl 从 scan cache 注入，apply 时 TOCTOU 慢路径使用
      "body": "...",
      "relations": [...],
      "unresolved_relations": [],          # target_title 解析失败的 relation
      "claim_status": "pending",           # pending | applied | conflict | invalid_source | duplicate
      "confidence": "high"
    }
  ]
}
```

**实现要点**：
- 批量分配 KNOW-IDs：`propose_impl` 调用私有函数 `_batch_alloc(n: int) -> list[int]`，一次 `fcntl.flock` 分配 N 个连续 ID，减少锁竞争。公开 MCP 工具 `alloc_knowledge_id` 保持单次分配不变（外部调用者用）
- `propose_impl` 按 `section_id` 从 scan cache 注入 `content_snapshot`
- 写 proposal JSON 到 `data_dir/atomize-proposals/pending/`
- 不写任何 vault 文件

### 4.4 `atomize_apply` — 提交变更

```python
@mcp.tool()
def atomize_apply(proposal_id: str, namespace: str = "") -> str:
    """
    读取已有 proposal，执行所有写操作，更新图。

    Proposal 状态机（单向无环）：
    pending → applied | partially_applied | failed | stale

    - applied：所有 claim 均成功（claim_status = applied）
    - partially_applied：至少一个 claim conflict/invalid_source，其余成功
    - failed：apply 过程崩溃或整体错误，proposal 保持 pending 可重试
    - stale：doc_sha 不一致，终态，不可 apply

    partially_applied 是终态，不支持局部重入。
    如需处理 conflict claim，必须发起全新 scan → propose（propose 层 claim 去重会自动跳过已成功的 claim）。

    执行顺序（CAS 安全点 + 文档自证幂等，failed 时可重试）：
    1. 读 proposal 文件，验证 status == "pending"
    2. for each claim（跳过 claim_status != "pending" 的）：
       a. 用 write_note 创建 {vault_root}/knowledge/<KNOW-XXXXXX>-<slug>.md
          （vault_root = PrismRagSettings.vault_path，即 nimbus namespace 对应的 vault 根目录；
           knowledge/ 是其下的固定子目录，与 code namespace 无关；slug 由 claim.title 小写化+连字符生成）
       b. 写入标准 frontmatter（knowledge_id, type, status, relations, mentioned_in）
          claim.relations 中的 target_id 字段写入 frontmatter 时转为 target（与 §3.3.5 schema 对齐）
          若 claim.relations 含 SUPERSEDES 边 → 同步将被取代节点 B.md frontmatter 的 status 改为 superseded
       c. 更新该 claim 的 claim_status = "applied"，写回 proposal JSON（CAS 安全点 1）
       d. 写 audit 事件
    3. patch 原文档 + 追加 frontmatter（合并为单次原子写入）：
       重跑幂等判断（文档自证）：
         读取源文档 frontmatter 的 atomized_nodes 列表，
         若已包含本 proposal 所有 applied claim 的 KNOW-ID → 跳过 step 3，直接进入 step 4
         （文档本身是 Source of Truth，无需依赖外部 doc_patched 标志）
       否则执行 patch：
       a. 全文 SHA == proposal 记录的 doc_sha → 用 source_lines 行号直接替换（快路径）
       b. SHA 不一致 → 对每个 claim_status == "applied" 的 claim 用 str.find(content_snapshot) 重新定位：
          - 找到唯一匹配 → 替换对应行为 "> 详见 [[KNOW-XXXXXX]]"
          - 找到多处匹配 → 取第一处（section heading 后第一个出现）
          - 找不到 → 该 claim 的 claim_status 更新为 "conflict"（知识文件已创建，仅原文 patch 失败），
            写回 proposal JSON，跳过该 claim patch，继续其余 claim
          注：claim_status "conflict" 在 step 3 中表示"patch 阶段失败"，区别于 step 2 中可能因 CAS hash 不同产生的 conflict
       c. 将 content patch + atomized_nodes frontmatter 追加合并为单次 write_atomic（tmpfile + os.rename），
          确保两者不会出现"patch 成功但 frontmatter 未更新"的半状态
    4. 触发 incremental ingest（同步）：
       对每个新建的 knowledge 文件调用 `ingest_file(path, skip_embed=True)`，复用现有
       `_sync_graph()` 路径；图节点立即可用，embedding 不在此步触发。
    5. 根据所有 claim_status 计算最终 proposal status，移到 applied/ 目录
    6. 同步更新 propose_dedup_index.json 中对应 key 的 status
       （pending → applied_pending_embed | partially_applied）；
       `applied_pending_embed` 表示图已更新但向量尚未生成，可由后台 `prism-rag embed` 批量补齐。
       用 atomic_write + AtomicFileOp（见 §4.6）写回

    返回：
    {
      "proposal_status": "applied | partially_applied",
      "created_nodes": ["KNOW-000043", "KNOW-000044", ...],
      "failed_claims": [],                 # claim_status=conflict 或 invalid_source 的 claim title 列表
      "patched_document": "设计细节/rag-architecture-design.md",
      "graph_update": {"added_nodes": 5, "added_edges": 7},
      "graph_available": true,             # 图节点已可查询
      "embedding_status": "pending",       # 向量待下次 embed 批量生成
      "audit_path": "data/audit.jsonl"
    }
    """
```

**容错与幂等性**：
- 每个 claim 写入前检查 `KNOW-XXXXXX.md` 是否已存在（CAS hash 不同则 claim_status → conflict）
- 每个 claim 完成后立即写回 `claim_status = "applied"`（CAS 安全点 1）
- step 3 以文档 frontmatter 的 atomized_nodes 作为幂等判断依据（文档自证，Source of Truth）；patch + frontmatter 追加合并为单次 `write_atomic`（tmpfile + os.rename），消除半状态风险
- proposal 移到 `applied/` 是最后一步

### 4.5 目录结构

```
data_dir/
├── registry.json                       # KNOW-ID 分配器
├── propose_dedup_index.json            # propose 幂等性索引（见下方 schema）
├── scan_cache/                         # scan 持久化缓存（TTL 24h，重启后加载）
│   └── scan-2026-05-05T10-20-a3f9c2b1-rag-architecture.json
└── atomize-proposals/
    ├── pending/                        # 待 apply
    │   └── atomize-2026-05-05T10-23-rag-architecture.json
    └── applied/                        # 已完成（含 applied + partially_applied）
```

**`propose_dedup_index.json` schema**：

```json
{
  "schema": "v1",
  "entries": {
    "sha256:e3b0c44298fc...": {
      "proposal_id": "atomize-2026-05-05T10-23-rag-architecture",
      "created_at": "2026-05-05T10:23:00Z",
      "status": "applied_pending_embed"
    }
  }
}
```

- key：`sha256(doc_sha + sorted([sha256(c.body + c.section_id) for c in claims]))`（用 LLM 提交的全部 claims，包含 duplicate；保证同一 payload 重试时 key 不变）
- `status` 字段随 proposal apply 结果同步更新（`pending → applied_pending_embed | partially_applied`）；stale（doc_sha 不一致）时 apply 提前返回，dedup_index 不更新，仍为 `pending`；`applied_pending_embed` 是终态，表示图已可用、向量待下次 `prism-rag embed` 批量补齐；dedup_index 的作用（阻止重复提交）至此已满足，不存在向 `applied` 升级的机制
- 读写时通过 AtomicFileOp（见 §4.6）包裹 `fcntl.flock`，用 `atomic_write` 落盘

### 4.6 AtomicFileOp — asyncio 环境并发安全规约

MCP server 运行在 asyncio 事件循环中。`fcntl.flock` 是**阻塞系统调用**，直接在协程体内调用会挂起整个事件循环，导致 MCP server 对其他请求无响应。

**规约**：任何涉及 `fcntl.flock` 或同步文件 I/O 的临界区，**禁止在协程体内直接调用**，必须通过 `AtomicFileOp` 包装：

```python
# prism_rag/store/atomic_file_op.py
class AtomicFileOp:
    """asyncio 安全的文件临界区包装。
    
    每个受保护资源持一把 asyncio.Lock（协程层互斥），
    实际 flock + I/O 通过 run_in_executor 派发到线程池，
    不阻塞事件循环。
    """
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, resource_key: str) -> asyncio.Lock:
        if resource_key not in self._locks:
            self._locks[resource_key] = asyncio.Lock()
        return self._locks[resource_key]

    async def run(self, resource_key: str, fn: Callable) -> Any:
        """在 asyncio.Lock 保护下，将 fn（含 flock）派发到 executor。"""
        async with self._get_lock(resource_key):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, fn)
```

**受保护资源**（三处，使用不同 resource_key）：

| 资源 | resource_key | 使用位置 |
|---|---|---|
| `registry.json` | `"registry"` | `_batch_alloc(n)` / `alloc_knowledge_id` |
| `scan_cache/` | `"scan_cache"` | scan cache 读写 |
| `propose_dedup_index.json` | `"dedup_index"` | propose 幂等写 / apply status 更新 |

`AtomicFileOp` 实例在 MCP server 启动时创建，作为 singleton 注入各 impl 函数。`embed_cache.jsonl` 的 append 已用 flock（§5.3），ingest 不在 asyncio 上下文中运行，直接 flock 即可，无需 AtomicFileOp 包装。

### 4.7 CLI 入口

```bash
# 查看所有 pending proposals
prism-rag atomize list

# apply 一个 proposal（apply 完成后会自动移到 applied/）
prism-rag atomize apply <proposal_id>

# 查看 proposal 详情
prism-rag atomize show <proposal_id>
```

### 4.8 测试矩阵

| 测试文件 | 内容 |
|---|---|
| `tests/test_registry.py` | `alloc_knowledge_id` 批量分配原子性、锁、格式化、回滚 |
| `tests/test_atomize_scan.py` | 解析 sections + section_id 赋值、scan_id 生成、scan cache TTL 24h、发现已有 KNOW-IDs |
| `tests/test_atomize_propose.py` | scan_id 查找（正常 / 过期 410）、content_snapshot 注入、claim 去重（duplicate 标记）、batch ID 分配、propose_key 幂等（网络超时重试命中同一 proposal）|
| `tests/test_atomize_apply.py` | 文件创建、SUPERSEDES 副作用写 B.md、原文 patch+frontmatter write_atomic（快路径 / 慢路径 / conflict）、proposal 状态机（applied / partially_applied）、claim_status CAS 写回、文档自证幂等（atomized_nodes 已含 KNOW-ID 则跳过 patch）|
| `tests/test_atomize_resume.py` | `atomize_apply` 中途模拟崩溃（crash after step 2 / crash mid-patch）；重跑验证：claim_status CAS 安全点 1 跳过已完成 claim；文档自证幂等跳过已 patch 文档；最终状态 partially_applied |
| `tests/test_atomic_file_op.py` | AtomicFileOp 并发安全：多协程并发调用同一 resource_key 不产生竞态；不同 resource_key 并发无阻塞；`run_in_executor` 不阻塞事件循环（timeout 断言） |

---

## 五、支柱 3 — Embedding 增强

### 5.1 问题

v5.2 ingest 暴露了一个关键问题：**无断点续传**。

ingest 被 kill 后（例如 GPU 被其他工作占用导致 timeout），所有已算完的 embedding 进度全部丢失，
下次必须从头算。1907 个 code 节点哪怕已算了 234 个，重启后全部重来。

两个 namespace（nimbus + code）使用同一个 embedding 模型（`qwen3-embedding:8b`），这是正确的——
两者向量空间必须相同，CrossNamespaceProbe 的 embedding_similar 才有意义。**不做 per-namespace 模型切换。**

### 5.2 GPU/CPU 感知

ingest 开始时自动检测当前 model 是否加载在 GPU 上：

```python
# prism_rag/ingest/embedder.py

def _detect_model_device(model: str, host: str) -> str:
    """
    调用 Ollama /api/ps，查找 model 当前使用的设备。
    返回 "gpu" | "cpu" | "unknown"。
    """
    try:
        resp = httpx.get(f"{host}/api/ps", timeout=5)
        for entry in resp.json().get("models", []):
            if model in entry.get("name", ""):
                return "gpu" if entry.get("size_vram", 0) > 0 else "cpu"
    except Exception:
        pass
    return "unknown"
```

**行为**：
- 若检测到 model 在 CPU 上运行，打印警告并建议等 GPU 空闲后重试
- 不自动切换，不自动终止（用户决策），但警告信息足够明显

### 5.3 Embedding 断点续传

#### 原理

ingest 过程中，embed 完的 node 立即写入 `embed_cache.jsonl` sidecar 文件。
下次 ingest 先读 cache，已有 embedding 的 node 跳过重算。

```
data_dir/
└── embed_cache.jsonl       # 每行：{"node_id": "...", "sha": "...", "vec": [...]}
```

- `sha` = `sha256(node_content)`。content 变了 → sha 变 → 重新 embed
- cache 用 `node_id` + `sha` 做 key，不是行号（图节点改名需要重算）
- ingest 完成后 cache **不删**，作为增量 ingest 的加速器
- **并发安全**：`_append_cache_entry` 写入前加 `fcntl.flock(fd, LOCK_EX)`，写完后 `LOCK_UN`。单次 append 为一行 NDJSON，原子性由 `PIPE_BUF`（4096 字节）保证——但 4096 维 float32 向量单行约 65 KB，超出 PIPE_BUF，**必须加锁**，不能依赖原子性假设。

#### API 变更

```python
# prism_rag/ingest/embedder.py

def compute_embeddings_ollama(
    graph: KnowledgeGraph,
    settings: PrismRagSettings,
    *,
    cache_path: Path | None = None,    # v5.3 新增
) -> None:
    """
    若 cache_path 非 None，先读 cache，跳过 sha 一致的 node，
    每 embed 完一个 node 立即 append 到 cache。
    """
```

#### CLI 入口

```bash
# 查看各 namespace 的 embed 进度
prism-rag embed-status

# 输出示例：
# namespace=nimbus  nodes=452  embedded=452  pending=0    model=qwen3-embedding:8b  device=gpu
# namespace=code    nodes=1907 embedded=234  pending=1673 model=qwen3-embedding:8b  device=cpu ⚠️ GPU not active
```

### 5.4 Timeout 配置

全局默认从 300s → **60s**。`qwen3-embedding:8b` on GPU 单次远低于 60s；在 CPU 上运行会在 60s 超时，
快速暴露问题而不是静默等待 5 分钟。

### 5.5 测试矩阵

| 测试文件 | 内容 |
|---|---|
| `tests/test_embed_cache.py` | cache 命中跳过、sha 变化重算、并发写入安全（使用 4096 维向量，真实触发 PIPE_BUF 超限场景；1 维向量不能覆盖此路径）|
| `tests/test_embed_status.py` | CLI `embed-status` 输出格式 |
| `tests/test_gpu_detection.py` | Ollama `/api/ps` mock 返回 cpu/gpu/unknown |

---

## 六、文件变更一览

### 新建文件

| 文件 | 说明 |
|---|---|
| `prism_rag/store/registry.py` | `Registry` 类：`alloc_id()`, `load()`, `save()` + 文件锁 |
| `prism_rag/store/atomic_file_op.py` | `AtomicFileOp` 类：asyncio.Lock + run_in_executor 双层锁（见 §4.6）|
| `prism_rag/ingest/atomize.py` | `atomize_scan_impl`, `atomize_propose_impl`, `atomize_apply_impl` |
| `prism_rag/cli_atomize.py` | `atomize` typer group：`list / show / apply` |
| `tests/test_registry.py` | REGISTRY 单元测试 |
| `tests/test_atomize_scan.py` | atomize_scan 单元测试 |
| `tests/test_atomize_propose.py` | atomize_propose 单元测试 |
| `tests/test_atomize_apply.py` | atomize_apply 单元测试 |
| `tests/test_atomize_resume.py` | `atomize_apply` 崩溃恢复测试（claim_status CAS + 文档自证幂等）|
| `tests/test_atomic_file_op.py` | AtomicFileOp 并发安全测试 |
| `tests/test_embed_cache.py` | embed cache 测试 |
| `tests/test_embed_status.py` | embed-status CLI 测试 |
| `tests/test_gpu_detection.py` | GPU 检测 mock 测试 |

### 修改文件

| 文件 | 变更 |
|---|---|
| `prism_rag/store/graph.py` | 新增 `NodeKind.KNOWLEDGE`、`RelationType` 新值、KNOW-ID 属性 |
| `prism_rag/ingest/vault_loader.py` | 解析 `knowledge_id` / `relations:` / `mentioned_in` frontmatter；图构建完成后执行 SUPERSEDES 内存修正（A.status==active 断言）；`load()` 返回时附带输出 `live_sha_set: set[tuple[node_id, sha]]` 供 embed_cache GC 使用 |
| `prism_rag/ingest/embedder.py` | 断点续传 cache、GPU 检测、timeout=60s 默认 |
| `prism_rag/mcp_server/server.py` | 注册 `atomize_scan / atomize_propose / atomize_apply / alloc_knowledge_id / list_knowledge_nodes` |
| `prism_rag/cli.py` | 注册 `embed-status`、引入 `cli_atomize.atomize` group |

---

## 七、实施顺序建议

```
前置条件（不进 sprint）
  └── Step 4 ZenithLoom ingest 完成 + Jei 验证
  └── Step 5 清理

Sprint A — Embedding 增强（独立，可先做）
  └── embed cache（断点续传）
  └── GPU 检测 + 警告
  └── timeout 缩短（300s → 60s）
  └── embed-status CLI

Sprint B — Vault Phase 2 数据模型
  └── NodeKind.KNOWLEDGE + RelationType 新值
  └── Registry
  └── vault_loader 解析 knowledge_id / relations
  └── alloc_knowledge_id + list_knowledge_nodes MCP 工具

Sprint C — atomize_document 工具族（依赖 Sprint B）
  └── atomize_scan
  └── atomize_propose（使用 Registry）
  └── atomize_apply（使用 vault CRUD + incremental ingest）
  └── CLI atomize group
```

**为什么 Embedding 增强先做**：它独立于 Phase 2 数据模型，且直接解除 ingest 被 kill 后重头算的痛点。

---

## 八、已知约束与非目标

### 非目标（v5.3 明确不做）

- **多模态 embedding**（图片/音频）— 参见 `PrismRag Multimodal Embedding — 设计方向与路线图.md`，状态：deferred
- **自动语义去重**（"KNOW-000043 和 KNOW-000031 其实在说同一件事"）— 留给后续 dedupe sprint
- **`atomize_document` 内部 LLM 调用** — PrismRag 不内嵌 LLM，语义判断由调用方（Jei）完成
- **vault watch mode**（文件变更自动触发 ingest）— 后续 sprint
- **跨 vault KNOW-ID 统一命名空间** — 目前每个 namespace 各自 REGISTRY，无全局统一

### 开放问题

| 问题 | 默认答案（暂定） |
|---|---|
| embed_cache 与图 node 的 TTL | cache 永久有效，只按 content SHA 失效，不按时间 |
| atomize_apply 失败时 KNOW-ID 是否回收 | 不回收（ID 已写入 REGISTRY，永久保留，file 不存在就是 "reserved but missing"） |
| proposal 的 `relations` 指向不存在的 KNOW-ID | 接受并写边，将来 node 存在后边会生效（宽松策略） |

---

## 九、决策日志

- **D1：`atomize_document` 是 PrismRag MCP 工具，不是 Jei 内嵌 skill** — PrismRag 是工具层，Jei 调用它，职责清晰。Jei 做语义判断，PrismRag 做机械写入。
- **D2：三工具协议（scan / propose / apply）** — 单一 `atomize_document(mode=...)` 调用会把语义决策时间和工具执行时间混在一起，LLM 难以中断审核。三工具协议让每个阶段有明确的输入输出，Jei 可以在 propose 后人工检查再 apply。
- **D3：proposal 文件存在 `data_dir`，不在 vault** — vault 是用户内容，proposal 是系统中间态，混在一起会污染 ingest。
- **D4：两个 namespace 使用同一 embedding 模型** — CrossNamespaceProbe 的 embedding_similar 需要两侧向量在同一空间，per-namespace 模型切换会让跨命名空间相似度分数失去意义。
- **D5：embed_cache 用 content SHA 做失效键** — 比时间戳更精确：内容没变就不重算，哪怕 file mtime 变了。
- **D6：默认 timeout 从 300s → 60s** — qwen3-embedding:8b 在 GPU 上单次远低于 60s；在 CPU 上超时应该快速暴露而不是静默等 5 分钟。
- **D7：REGISTRY 空洞 by-design** — 崩溃时事务边界无法跨 REGISTRY + 文件系统两个存储，空洞 ID（分配但 file 未写成功）不可避免。标记为 "reserved but missing"，对用户透明，不影响正确性。batch 分配（D17）仅为降低锁竞争，不解决空洞问题。
- **D8：mentioned_in 边方向为 knowledge_node → source_document（P1）** — 新边类型 EXTRACTED_FROM 增加 schema 复杂度，当前 REFERENCES 已覆盖"A 提及 B"语义，无混淆风险。统一用 REFERENCES，方向从 knowledge node 指向其出处文档。
- **D9：source_lines TOCTOU 用 content_snapshot 解决（P0）** — scan 和 apply 之间文档可能被编辑，行号失效。snapshot 存全 section 文本，apply 时优先 SHA 快路径，SHA 不一致则 str.find(snapshot) 重定位，找不到则 conflict 跳过，不静默破坏文档。
- **D10：target_title 两级解析（P0）** — 批内精确匹配 → 图内标准化精确 → 语义 fallback（≥0.90 且 top-1 唯一）→ unresolved_relations，不报错。避免 LLM 提供的自然语言 title 完全无法解析时 apply 失败。
- **D11：scan 返回 section_id + scan_id（P0）** — LLM 用 `section_id`（`s_0`, `s_1`, ...）引用 section，而非 heading 文本，消除 LLM 文本变异导致的匹配失败。`scan_id` 确保并发扫描同一文档时 section_id 命名空间不混淆。
- **D12：content_snapshot 由 propose_impl 注入（P0）** — LLM 不需要转发 snapshot 数据（减少 prompt 体积和出错概率）；`propose_impl` 用 `scan_id + section_id` 做 O(1) lookup，从 server 端 scan cache 取 snapshot 写入 proposal claim。
- **D13：scan cache TTL 24h + 410 Gone** — 过期返回 410，强制重新 scan 避免 stale snapshot 被使用。TTL 24h 足够完成一次 scan→propose→apply 流程。（注：scan cache 已改为持久化到 data_dir/scan_cache/，见 D21；D13 仅保留 TTL + 410 语义，不再描述"临时内存状态"。）
- **D14：doc_sha 不一致 → stale 终态 + 409，无 force 参数** — 允许强制覆盖等于从 API 层撕裂 content_snapshot 一致性保证，与 TOCTOU 修复设计意图自相矛盾。唯一合法路径：重新 scan → propose。
- **D15：partially_applied 是终态，无局部重入** — 允许局部重入会让 proposal 状态机产生环（pending → partially_applied → pending），引入与 v5.2 inbox 设计同等级别的意外复杂度。propose 层 claim 去重是闭合业务循环的正确位置。
- **D16：propose 层 claim 内容哈希去重** — 按 `sha256(claim.body + section_id)` 与已有 knowledge nodes 比对。去重逻辑收敛在 propose 层：不污染 scan 阶段（不给 LLM 注入屏蔽列表），apply 层无需额外判断，LLM 重新提交时已成功的知识点自然跳过。
- **D17：batch alloc 为 propose_impl 私有函数 `_batch_alloc(n)`** — 公开 MCP 工具 `alloc_knowledge_id` 保持单次分配（外部调用者语义清晰）；`propose_impl` 内部用 `_batch_alloc(n)` 一次 fcntl.flock 分配 N 个连续 ID，减少锁竞争。中途失败时已分配 ID 成为空洞（D7 by-design），REGISTRY 单调递增保证不复用，不做回滚。
- **D18：propose 幂等性 key = sha256(doc_sha + sorted(claims_hash))，dedup_index 永不清理** — 时间戳生成 proposal_id 导致重试产生重复 proposal + ID 浪费。content-addressed key 让同一意图的重试命中同一 proposal。dedup_index 无需 GC：doc_sha 已绑定文档版本，文档变化时 key 自然失效；千级 Vault 下文件 < 100KB。
- **D19：step 3+4 合并为 write_atomic，以文档自证替代 doc_patched 标志** — patch + frontmatter 追加分两步写入时，crash 可产生"patch 成功但 atomized_nodes 未追加"的半状态；外部 doc_patched 标志和文件系统也可能撕裂。write_atomic（tmpfile + os.rename）消除半状态；重跑幂等判断以文档 frontmatter 中是否已有 KNOW-ID 为准（文件系统是 Source of Truth，外部标志是冗余状态）。
- **D20：vault_loader SUPERSEDES 内存修正，源节点 active 断言防死节点传播** — 图中 SUPERSEDES 边存在但 B.md status 仍为 active，图与文件不一致。vault_loader 图构建完成后在内存修正 B.status，仅校验 A.status == active（防止废弃的 A 传递降级效果），不写回文件（文件由 atomize_apply 负责）。live_sha_set 由 vault_loader 顺带输出，供 embed_cache GC 零成本复用。
- **D21：scan cache 持久化到 data_dir/scan_cache/，启动时加载 + TTL 清理** — 内存 cache 在 Jei 重启后丢失，迫使 LLM 重新 scan + 重新构造 propose payload（2000+ token 浪费 + 幻觉风险）。持久化成本极低（JSON 文件）；启动时加载并清理过期文件，TTL 语义不变。
- **D22：KNOW-ID 采用 6 位补零（:06d），regex `KNOW-(\d{6,})`，排序按整数值** — 4 位上限 9999 个 node，中等规模 vault 数年内可触顶；6 位（999999）足够整个产品生命周期。regex `\d{6,}` 兼容未来若再扩位时无需代码改动；int 排序避免字典序 "KNOW-000002 > KNOW-000010" 导致 list_knowledge_nodes 结果乱序。
- **D23：AtomicFileOp 双层锁（asyncio.Lock + run_in_executor 包裹 flock）** — asyncio MCP server 中直接调用 fcntl.flock 是阻塞系统调用，挂起事件循环会导致所有并发请求超时。asyncio.Lock 保证协程层的 happen-before；run_in_executor 将真实 flock + I/O 派发到线程池，不阻塞事件循环。embed_cache 的 flock 在 ingest（非 asyncio 环境）中调用，无此问题，无需包装。
- **D24：incremental ingest 使用 `ingest_file(skip_embed=True)`，引入 `applied_pending_embed` 状态** — atomize_apply 完成后图节点需立即可查询（Jei 下一轮 MCP call 可能就引用新节点），但 embedding 是重计算操作，不应阻塞 apply 返回。skip_embed=True 复用现有 _sync_graph 路径；applied_pending_embed 状态让 LLM 和运维脚本能区分"图已有但向量还没有"，避免混淆 embedding_similar 查询结果为空的原因。
