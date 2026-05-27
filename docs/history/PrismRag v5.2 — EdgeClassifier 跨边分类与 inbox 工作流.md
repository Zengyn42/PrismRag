---
title: "PrismRag v5.2 — EdgeClassifier 与 MCP 驱动的 inbox 审核"
tags:
  - "#PrismRag"
  - "#GraphRAG"
  - "#architecture"
  - "#cross-namespace"
  - "#design"
status: spec
created: 2026-05-04
last_audited: 2026-05-04
milestone: v5.2
related:
  - "设计细节/PrismRag v5.0 — 详细设计与任务分解.md"
  - "设计细节/PrismRag v5.0 — 通用图引擎架构设计.md"
  - "设计细节/PrismRag v5.1 — mentions_symbol 跨命名空间链接设计.md"
note: |
  本设计第五/六轮 review 后做了架构级转向：原 v1 方案把 inbox 物理放在
  vault notes 里（仅 git 历史可查，commit efb7860 之前），引入了约 10 条
  accidental 复杂度（system namespace, proposed_mention 弱边, annotation
  回写, Embedder filter-then-chunk, hash-chain death loop 等）。本 v2
  方案把 inbox 移到 PrismRag/data/inbox.jsonl，人审走三个共享 InboxStore
  的入口：TUI（主, textual 全屏键盘流, 适合批量）、Jei MCP（辅, 对话式,
  适合单条精审）、CLI（脚本/CI）。副作用表从 19 行缩到 10 行，实现量含
  TUI 与 v1 基本持平但维护负担少 ~50%，零 vault-as-state-machine 风险。
---

# PrismRag v5.2 — EdgeClassifier 与 MCP 驱动的 inbox 审核

> v5.1 实现了 `mentions_symbol` 边的精确匹配创建（wikilink + 全局唯一符号名）。
> v5.0 引入了 `CrossNamespaceProbe` 被动收集 federated graph 的桥接边（embedding_similar 软提示）。
> 当前状态：probe 已经积累了 **98 条** `embedding_similar` cross edge，但**没有任何代码消费它们**。
>
> v5.2 的任务：把被动观察升级为主动行动，给 probe 数据装一个**三档分类器**，
> 加一个**外置 inbox 队列 + Jei 对话式审核**工作流。

---

## 一、激活条件（已满足）

| 条件 | 阈值 | 当前 |
|------|------|------|
| nimbus 节点数 | > 200 | 452 ✅ |
| cross_namespace_log 条数 | > 50 | 98 ✅ |
| 人工确认有价值的条数 | ≥ 10 | v5.2 落地后产生 |

---

## 二、核心架构

```
                CrossNamespaceProbe（v5.0 已有，被动）
                            │ 持续追加 → cross_namespace_log.jsonl
                            ▼
        ┌────────────────────────────────────────────────┐
        │      EdgeClassifier（v5.2 新增）                │
        │   输入：probe.list_cross_edges() 全量            │
        │   按 (source, target) 聚合 + 评分                │
        │   分到 Tier 1 / Tier 2 / Tier 3                 │
        └─────┬────────────┬────────────┬────────────────┘
              │            │            │
       Tier 1：自动晋升  Tier 2：进 inbox  Tier 3：丢弃
              │            │            │
              ▼            ▼            ▼
        在 vault graph    append/upsert   保留在 log
        写 mentions_      到 jsonl 队列    不动作
        symbol 边         status: pending
        (ANCHORED)        ─────────┬─────────
                                   ▼
                       Jei MCP 工具 / CLI 审核
                                   │
                                   ▼
                       review_pending_edge(decision)
                          → approved → 写正式 ANCHORED 边
                          → rejected → status 改 rejected（保留审计）
```

**关键架构选择（与原 v1 对比）：**

| 维度 | v1 方案（已废弃）| v2 方案（本设计）|
|------|----------------|----------------|
| Inbox 存储 | vault `inbox/cross-edges/*.md` | `PrismRag/data/inbox.jsonl` |
| 人审 UX 主入口 | Obsidian 直接看笔记 | **TUI**（`prism-rag inbox review`，textual 全屏键盘流）|
| 人审 UX 辅助入口 | — | **Jei MCP 对话**（list / context / review 三工具）+ CLI 批量 |
| 触发审核 | vault 文件 hash 变化 → incremental hook | TUI / MCP / CLI 直接调 InboxStore |
| Inbox 可见性 | 同时 namespace 隔离 + 弱边补救 | 不在图里，只在 jsonl，三种入口显式查 |
| 死循环防护 | 多层 hook 白名单 | 不需要——没有文件 hash 链 |
| Embedder 干扰 | 需要 filter-then-chunk 剥离审计段 | 不存在——审计内容不进 vault |

**三个审核入口的分工**（详见 §六、§七）：

| 入口 | 适合场景 | 实现 |
|------|---------|------|
| **TUI**（`prism-rag inbox review`）★ | 批量过 30+ 条 pending、键盘流快速决策 | `prism_rag/inbox/tui.py`（textual app, ~300 行）|
| **Jei MCP**（3 个工具）| 对话中顺便审、单条精审、需要解释 | `prism_rag/mcp_server/server.py` 加 list/context/review 工具 |
| **CLI**（`prism-rag inbox approve <id>` 等）| 脚本化、CI、程序化批量 | `prism_rag/cli.py` |

三入口共享同一个 `InboxStore` + `apply_decision` 实现。

**Inbox 与 graph / embedder / probe 完全解耦**：
- inbox.jsonl 不被 vault loader 读取
- 不算 embedding，不进 BM25，不参与 build_bridges
- 不会触发任何 incremental hook
- 唯一写入路径：classifier 的 Trigger A、MCP 工具 review_pending_edge / CLI inbox approve|reject

---

## 三、三档分类标准

### 阈值由 model_id → profile 映射

不同 embedding 模型 confidence 分布完全不同。**阈值绑定到 model_id，配置驱动**：

```yaml
classifier_profiles:
  bge-m3:
    tier1_min_conf: 0.75
    tier1_top_k: 1
    tier1_min_consecutive: 2
    tier2_min_conf: 0.70
    tier2_margin: 0.25      # confidence ≥ top1 * (1-margin)
    tier2_hard_cap: 5
    tier2_min_consecutive: 2
  qwen3-embedding-8b:
    tier1_min_conf: 0.85
    tier1_top_k: 1
    tier1_min_consecutive: 2
    tier2_min_conf: 0.78
    tier2_margin: 0.20
    tier2_hard_cap: 5
    tier2_min_consecutive: 2
  default:
    tier1_min_conf: 0.85
    tier2_min_conf: 0.75
    tier1_top_k: 1
    tier2_margin: 0.25
    tier2_hard_cap: 5
    tier1_min_consecutive: 2
    tier2_min_consecutive: 2
```

Classifier 读取 `settings.embedding_model` → 查 profile → 用对应阈值。换模型只改 `embedding_model` 配置和（如有）profile，不动代码。

### Tier 2 候选筛选 — Margin + Hard Cap

纯 top-K 在长尾分布里太宽松。每个 source 的 cross-edge candidates 按 confidence 排序后用 margin + cap 筛：

```python
def select_tier2_candidates(ranked, margin=0.25, hard_cap=5):
    if not ranked:
        return []
    threshold = ranked[0].confidence * (1.0 - margin)
    return [e for e in ranked if e.confidence >= threshold][:hard_cap]
```

行为示例：
- top-1=0.80, margin=0.25 → 阈值 0.60，所有 ≥ 0.60 进入候选 → 取前 5
- top-1=0.74, margin=0.25 → 阈值 0.555 → 假设 50 条 ≥ 0.555，取前 5

### 三档判定（用 bge-m3 profile 举例）

| 档位 | 阈值（AND 组合） | 动作 |
|------|-----------------|------|
| **Tier 1：自动晋升** | confidence ≥ profile.tier1_min_conf **AND** 是该 source 的 top-1 跨 ns 邻居 **AND** consecutive_seen ≥ profile.tier1_min_consecutive | 在 vault graph (nimbus) 写一条 `mentions_symbol` 边，方向 vault→code，confidence_tier=`INFERRED`，**lifecycle_class=ANCHORED**，evidence 含 `auto_promoted_from_probe(model=bge-m3, score=0.XX, consecutive=N)`。同步把 inbox.jsonl 中对应 entry 的 status 改 `auto_promoted` |
| **Tier 2：进 inbox** | confidence ≥ profile.tier2_min_conf **AND** 通过 margin+hard_cap **AND** (consecutive ≥ profile.tier2_min_consecutive **OR** 是 source 的 top-1) **AND** 不在 Tier 1 | upsert 一条 entry 到 `data/inbox.jsonl`，status=pending |
| **Tier 3：丢弃** | 其他 | 不动作；如果之前在 inbox（pending）→ status 改 `discarded`（保留审计） |

### 阈值设计依据

- 0.75 不是 0.85：bge-m3 跨模态相似度天花板 ~0.75
- top-1 比绝对阈值更可靠：相对排序 > 绝对 confidence
- ≥ 2 次稳定：单次出现可能是 embedding 抖动；首次跑 v5.2 时所有条目带 MIGRATION_PENDING（§八）→ Tier 1/2 都不发，全 NOOP

### 首次跑 v5.2 的预期产出

- Tier 1：**0 条**（MIGRATION_PENDING 阻断）
- Tier 2：**0 条**（同上）
- Tier 3：所有 98 条 noop

第一轮 incremental ingest 后部分 source_file 的哨兵被覆盖：
- 估计 15-30 条进 inbox（margin+cap 比纯 top-3 严）

第二轮同 source_file ingest（consecutive_seen 增到 2）：
- 5-15 条满足 Tier 1 → 自动晋升

---

## 四、inbox.jsonl Schema

**路径**：`PrismRag/data/inbox.jsonl`

每行一条 JSON entry（NDJSON）。Append-only for 新条目；status 更新走全文 atomic rewrite（tmp + os.replace）。

**Entry 字段**：

```json
{
  "id": "code__claude.py__ClaudeSDKNode__to__nimbus__设计细节__Hani 设计文档",
  "source": "nimbus::设计细节/Hani 设计文档",
  "target": "code::framework/nodes/llm/claude.py::ClaudeSDKNode",
  "edge_kind": "mentions_symbol",
  "confidence": 0.74,
  "confidence_tier": "INFERRED",
  "model_id": "bge-m3",
  "probe_signals": [
    {
      "kind": "embedding_similar",
      "score": 0.74,
      "consecutive_seen": 2,
      "first_seen_at": "2026-05-04T07:25:20Z",
      "last_seen_at": "2026-05-04T07:30:11Z"
    }
  ],
  "top_k_rank": 1,
  "status": "pending",
  "created_at": "2026-05-04T07:25:20Z",
  "decided_at": null,
  "decided_by": null,
  "decision_note": null
}
```

**Status 状态机**：

| Status | 含义 | 终点？ |
|--------|------|--------|
| `pending` | 待审 | 否 |
| `approved` | 人工通过 → 已写正式 mentions_symbol 边 | 是（保留审计）|
| `rejected` | 人工拒绝 | 是（保留审计）|
| `auto_promoted` | classifier Tier 1 自动晋升 → 已写正式 mentions_symbol 边 | 是（保留审计）|
| `discarded` | classifier Tier 3 降级（之前在 pending，现已不满足）| 是（保留审计）|

**审计保留**：所有终点状态保留在 jsonl 里。"列出 pending" 是 `status == "pending"` 的过滤。这样能回答"什么时候 approve 了什么、之前的 confidence 多少"——任何审计追溯都能回放。

**ID 计算**：`{source.replace(":/", "_")}__to__{target.replace(":/", "_")}`（用稳定的字符替换让 ID 文件名安全且可逆推）。

**Upsert 语义**：

| 已存在 entry 状态 | classifier 重跑动作 |
|------------------|---------------------|
| `pending` | 更新 `confidence` / `probe_signals` / `top_k_rank`（保留 status=pending）|
| `approved` / `auto_promoted` | 完全跳过（终点状态不可变）|
| `rejected` / `discarded` | 完全跳过（已有人工/系统决策）|

---

## 五、Trigger A — Ingest-time evaluation

### 前置 0：LifecycleClass enum（v1 保留）

```python
class LifecycleClass(StrEnum):
    PROBABILISTIC = "probabilistic"   # probe 召回的临时记录
    DETERMINISTIC = "deterministic"   # v5.1a SymbolLinker 写的 mentions_symbol
    ANCHORED      = "anchored"        # v5.2 Tier 1 / approved 写的 mentions_symbol
```

| 操作 | PROBABILISTIC | DETERMINISTIC | ANCHORED |
|------|--------------|---------------|----------|
| probe.sweep() 清零 consecutive_seen | ✅ 可 | N/A | ❌ 跳过 |
| v5.1a SymbolLinker 全量覆写 | ❌ 不动 | ✅ 可 | ❌ 跳过 |
| 物理文件删除导致的 sweep_deleted_files() | ✅ 可 | ✅ 可 | ✅ 可 |

**核心承诺**：ANCHORED 一旦设置，任何自动化路径都不会修改它。生命周期唯一终点是物理文件删除。

`lifecycle_class` 同时存在于：
- `CrossEdgeEntry`（probe log 条目）：默认 PROBABILISTIC，Tier 1 晋升时改 ANCHORED
- `Edge`（graph 边）：v5.1a 写的 mentions_symbol = DETERMINISTIC；v5.2 晋升的 mentions_symbol = ANCHORED

**没有第四态**——所有"未确认"用 `MIGRATION_PENDING` 哨兵（§八）。SymbolLinker / Sweep 边界判定：**"不是 DETERMINISTIC 也不是 ANCHORED 的，一律视为临时边"**。

### 前置 1：v5.1a SymbolLinker 守卫

```python
# prism_rag/ingest/symbol_linker.py 改造
def _clear_existing_mentions(graph):
    for u, v, d in list(graph.edges(data=True)):
        if d.get("relation") != "mentions_symbol":
            continue
        if d.get("lifecycle_class") == LifecycleClass.ANCHORED:
            continue   # v5.2 晋升的边，SymbolLinker 不管
        graph.remove_edge(u, v)
```

历史 v5.1a 写的 mentions_symbol 边在迁移脚本中强制标 DETERMINISTIC（§八）。

### 前置 2：Probe 扩展 — per-file scan-then-sweep + model_id

```python
MIGRATION_PENDING = "MIGRATION_PENDING"

@dataclass
class CrossEdgeEntry:
    edge_id: str
    source_node: str           # "code::path/file.py::Symbol"
    target_node: str
    edge_kind: str
    confidence_tier: str
    confidence: float
    first_seen_at: str
    last_seen_at: str = ""
    last_seen_parsed_at: str = ""       # MIGRATION_PENDING for legacy entries
    source_file: str = ""               # 用于 sweep 定位
    consecutive_seen: int = 1
    model_id: str = ""
    lifecycle_class: str = LifecycleClass.PROBABILISTIC
    evidence: list[str] = field(default_factory=list)
```

**为什么不用 batch_id**：原方案"Probe 构造时一次性 UUID"假设 build_bridges = 一个批次。但 `prism-rag serve` 长跑时 build_bridges 只在 server 启动时调一次，后续都是单文件 ingest——同一个 batch_id 永远活下去，consecutive_seen 永远不增长。**per-file scan-then-sweep 把"批次"颗粒度降到一次单文件 ingest scan**。

```python
class CrossNamespaceProbe:
    def __init__(self, log_path=None, model_id=""):
        self._model_id = model_id
        # 不再有 batch_id

    def record(self, bridge: dict, scan_timestamp: str) -> None:
        edge_id = ...
        existing = self._index.get(edge_id)

        # ANCHORED 守卫：脱离概率性更新
        if existing is not None and existing.lifecycle_class == LifecycleClass.ANCHORED:
            existing.last_seen_at = now_iso()   # 仅审计可见性
            return

        if existing is None:
            entry = CrossEdgeEntry(
                ..., consecutive_seen=1,
                last_seen_parsed_at=scan_timestamp,
                source_file=bridge.get("source_file", ""),
                model_id=self._model_id,
                lifecycle_class=LifecycleClass.PROBABILISTIC,
            )
        elif existing.last_seen_parsed_at == MIGRATION_PENDING:
            # 历史迁移条目首次被实际扫描验证
            existing.last_seen_parsed_at = scan_timestamp
            existing.last_seen_at = now_iso()
            existing.confidence = float(bridge.get("weight", 0.7))
            existing.model_id = self._model_id
            # consecutive_seen 维持 1（首次真实确认）
        elif existing.model_id != self._model_id:
            # 模型变了：重置（旧 confidence 不可比）
            existing.consecutive_seen = 1
            existing.last_seen_parsed_at = scan_timestamp
            existing.model_id = self._model_id
            existing.confidence = float(bridge.get("weight", 0.7))
        elif existing.last_seen_parsed_at != scan_timestamp:
            # 同模型、新 scan：连续计数 +1
            existing.consecutive_seen += 1
            existing.last_seen_parsed_at = scan_timestamp
            existing.last_seen_at = now_iso()
        else:
            # 同模型、同 scan：no-op
            return

    def sweep(self, source_file: str, scan_timestamp: str) -> int:
        """ANCHORED / DETERMINISTIC 不归 sweep 管。"""
        swept = 0
        for entry in self._index.values():
            if entry.lifecycle_class != LifecycleClass.PROBABILISTIC:
                continue
            if entry.source_file != source_file:
                continue
            if entry.last_seen_parsed_at == scan_timestamp:
                continue
            if entry.consecutive_seen > 0:
                entry.consecutive_seen = 0
                swept += 1
        return swept
```

**调用链**：

```
incremental.ingest_file(path)
    ├── 解析文件，提取 cross-edge candidates
    ├── scan_ts = now_iso()
    ├── for each candidate: probe.record(candidate, scan_ts)
    └── probe.sweep(source_file=str(path), scan_timestamp=scan_ts)
```

### Classifier 主流程

```python
@dataclass
class ClassifyReport:
    promoted: int      # Tier 1 计数
    queued: int        # Tier 2 计数（首次入 inbox）
    upgraded: int      # Tier 2→1 跃迁
    rolled_back: int   # Tier 1/2 → 3 降级
    discarded: int     # Tier 3
    inbox_path: str    # data/inbox.jsonl

def classify_and_route(
    federated: FederatedGraph,
    probe: CrossNamespaceProbe,
    settings: PrismRagSettings,
    profile: ClassifierProfile,
) -> ClassifyReport:
    inbox = InboxStore(settings.data_dir / "inbox.jsonl")
    for entry in probe.list_cross_edges():
        # MIGRATION_PENDING 短路：迁移条目未经验证不获信用
        if entry.last_seen_parsed_at == MIGRATION_PENDING:
            continue
        # ANCHORED：已晋升过，跳过
        if entry.lifecycle_class == LifecycleClass.ANCHORED:
            continue
        tier = _classify_tier(entry, profile)
        if tier == 1:
            _promote_to_tier1(federated, probe, entry, inbox)
        elif tier == 2:
            _upsert_inbox(inbox, entry)
        else:
            _maybe_discard(inbox, entry)
    inbox.save_atomic()
    federated.save_changed_atomic()
    return ClassifyReport(...)


def _promote_to_tier1(fg, probe, entry, inbox):
    # 方向反转：probe log 是 (source=code, target=vault)，
    # mentions_symbol 语义是 vault → code
    sem_source = entry.target_node   # "nimbus::path/note"
    sem_target = entry.source_node   # "code::path::Symbol"
    vault_g = fg.get_graph("nimbus")     # ⚠️ 必须写 nimbus，不是 target_ns
    bare_src = sem_source.split("::", 1)[1]
    edge = Edge(
        source=bare_src,
        target=sem_target,
        relation="mentions_symbol",
        confidence="INFERRED",
        confidence_score=entry.confidence,
        source_pass="conv",
        lifecycle_class=LifecycleClass.ANCHORED,   # ⚠️ 关键
    )
    vault_g.add_edge(edge)

    # 同步标 probe entry ANCHORED：之后 sweep 永不清零
    probe._index[entry.edge_id].lifecycle_class = LifecycleClass.ANCHORED

    # inbox 中标 auto_promoted（如果之前在 pending 队列里）
    inbox.set_status(entry.edge_id, "auto_promoted",
                     decided_by="classifier",
                     decision_note=f"Tier 1 auto-promote (consecutive={entry.consecutive_seen})")


def _upsert_inbox(inbox, entry):
    existing = inbox.get(entry.edge_id)
    if existing and existing["status"] != "pending":
        return   # 终点状态不可变
    if existing:
        # 更新 confidence / probe_signals / top_k_rank
        inbox.update_pending(entry.edge_id, entry)
    else:
        inbox.append(entry, status="pending")


def _maybe_discard(inbox, entry):
    existing = inbox.get(entry.edge_id)
    if existing and existing["status"] == "pending":
        # 之前 Tier 2 现在 Tier 3，降级 status 但保留审计
        inbox.set_status(entry.edge_id, "discarded",
                         decided_by="classifier",
                         decision_note=f"Tier 3 rollback (conf={entry.confidence})")
```

**调用点**：
- `cli.py` 的 `ingest` 命令尾部
- `cli.py` 的 `link-symbols` 命令尾部
- 新独立 CLI：`prism-rag classify-edges`

### 图持久化的并发安全 + 500ms mtime IPC

Classifier（CLI 进程）和 MCP server（长跑进程）并发访问 graph.json + inbox.jsonl。

**写入层（Sprint 1 协议）**：所有写入走 atomic 模式：

```python
def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)   # POSIX rename：原子，不撕裂
```

**读取层（Sprint 1 协议）**：server 用 mtime 探测自动感知变化：

```python
class FederatedGraph:
    _MTIME_CHECK_INTERVAL_S: float = 0.5

    def _maybe_reload(self) -> None:
        """每 500ms 检查 graph.json 的 mtime，变化则 lazy reload。"""
        now = time.monotonic()
        if now - self._last_check_at < self._MTIME_CHECK_INTERVAL_S:
            return
        self._last_check_at = now
        for ns, src in self._sources.items():
            current_mtime = src.graph_path.stat().st_mtime
            if current_mtime > self._mtime_cache.get(ns, 0.0):
                self._reload_namespace(ns)
                self._mtime_cache[ns] = current_mtime
```

每个 MCP 工具入口调一次 `fg._maybe_reload()`。500ms 内连续查询共享一次 check 结果。

**这是 Sprint 1 的硬协议**——atomic 字节级原子 + 500ms 轮询周期，Sprint 2 业务变更期间不动。

### system graph GC（v5.2 不再需要，但 sweep_deleted_files 仍是 PrismRag 通用基础设施）

v2 方案 inbox 不在 vault，所以不存在 system graph。但 `sweep_deleted_files()` 作为 PrismRag 通用文件删除清理仍需要——它是独立于 v5.2 的基础设施改进，对其他 namespace 也有用。

```python
def sweep_deleted_files(graph: KnowledgeGraph, namespace_root: Path) -> int:
    """扫描所有 graph 节点的 source_file，磁盘上不存在 → 移除节点。
    跳过 ANCHORED 边对应的 stub 节点（不强删）。"""
```

调用时机：每次 `prism-rag ingest` 命令尾部。

---

## 六、Trigger C — 双审核入口（TUI 主 + MCP 辅）

**没有文件 hook，没有 hash 链**——审核通过两个独立入口直接调，共享 `InboxStore` 底层，原子地修改 inbox.jsonl + 写 graph 边。

| 场景 | 推荐入口 | 强项 | 弱项 |
|------|---------|------|------|
| 批量过 30 条 pending（首次跑 v5.2 后的清单）| **TUI**（B1） | 键盘流（j/k/a/r/n），3-5 秒/条 | 无 markdown 链接渲染 |
| 单条精审、需要解释 confidence 的细节 | **Jei MCP**（B2）| 自然语言追问、上下文解释 | 慢，对话往返 |
| 程序化批量（CI / 脚本）| `prism-rag inbox approve <id>` | 幂等、可脚本化 | 无交互 |

两套入口都受同一个 `USAGE CONTRACT` 约束——必须用户/操作者**显式**给出 approve/reject 指令，不允许任何"基于置信度自动批"的旁路。

---

### 入口 1（主）：TUI — `prism-rag inbox review`

**实现**：基于 `textual` 库的全屏 TUI，单 Python 进程，纯键盘驱动。

**界面草图**：

```
┌─ PrismRag Inbox Review (12 pending, 3 approved this session) ─────┐
│ [▶] 1/12   conf 0.745   top-1   consec 3   model bge-m3            │
├──── Source · vault doc ────────────────────────────────────────────┤
│ 设计细节/Hani 设计文档.md                                           │
│                                                                    │
│ Hani 是无垠智穹的技术架构师 agent，使用 Claude SDK 子进程隔离的     │
│ 方式调用 LLM。具体实现见 framework/nodes/llm/claude.py 中的         │
│ ClaudeSDKNode 类... [↓ scroll for more]                            │
├──── Target · code symbol ──────────────────────────────────────────┤
│ framework/nodes/llm/claude.py::ClaudeSDKNode  (line 42-187)        │
│                                                                    │
│ class ClaudeSDKNode(LlmNode):                                      │
│     """Claude Agent SDK subprocess node.                           │
│                                                                    │
│     Spawns claude-code as a child process via the SDK; isolates... │
│     """                                                            │
│     async def call_llm(self, prompt: str, ...) -> tuple[str, str]: │
│         ...                                                        │
├────────────────────────────────────────────────────────────────────┤
│ probe signals: embedding_similar(0.745, consec=3, last 2026-05-04) │
│ note: _                                                            │
├────────────────────────────────────────────────────────────────────┤
│ [a]pprove  [r]eject  [n]ote  [s]kip  [↑↓] navigate  [q]uit & save  │
└────────────────────────────────────────────────────────────────────┘
```

**键位**：

| 键 | 动作 |
|----|------|
| `↓` / `j` | 下一条 |
| `↑` / `k` | 上一条 |
| `a` | approve（带可选 note） |
| `r` | reject（带可选 note） |
| `n` | 编辑 note（弹多行编辑器；保存不动 status） |
| `s` | skip（status 维持 pending，不计入本次决策） |
| `Tab` | 在 source / target 两个区切 scroll 焦点 |
| `f` | 过滤（`f conf>0.75` / `f rank=1` / `f /symbolname/`）|
| `q` | quit & save（atomic 写一次 inbox.jsonl + graph.json）|

**状态语义**：所有决策**先在内存累积**，按 `q` 时一次性 atomic 写盘——避免审到一半 Ctrl-C 留下半改的 inbox。如果用户 Ctrl-C 也尝试触发紧急保存（`SIGINT` → `_emergency_save()`）。

**实现细节**：

```python
# prism_rag/inbox/tui.py
from textual.app import App
from textual.widgets import Header, Footer, Static, Input
from prism_rag.inbox.store import InboxStore
from prism_rag.inbox.approval import apply_decision

class InboxReviewApp(App):
    BINDINGS = [
        ("a", "approve", "Approve"),
        ("r", "reject", "Reject"),
        ("n", "edit_note", "Note"),
        ("s", "skip", "Skip"),
        ("j", "next_entry", "Next"),
        ("k", "prev_entry", "Prev"),
        ("q", "save_and_quit", "Quit"),
    ]

    def __init__(self, inbox_path: Path, settings: PrismRagSettings):
        super().__init__()
        self._inbox = InboxStore(inbox_path)
        self._pending = self._inbox.list_pending(top_n=999)
        self._idx = 0
        self._decisions: dict[str, tuple[str, str]] = {}   # edge_id → (decision, note)
        self._settings = settings

    def action_approve(self) -> None:
        edge_id = self._pending[self._idx]["id"]
        self._decisions[edge_id] = ("approve", self._current_note)
        self._advance()

    def action_save_and_quit(self) -> None:
        # atomic apply all decisions, then exit
        for edge_id, (decision, note) in self._decisions.items():
            apply_decision(edge_id, decision, note,
                           inbox=self._inbox, settings=self._settings,
                           decided_by="user_via_tui")
        self._inbox.save_atomic()
        # graph.json 在 apply_decision 内部 atomic save
        self.exit()

    # ... 其余方法
```

**测试策略**：textual 提供 `Pilot` 模拟键盘，`tests/test_inbox_tui.py` 用它走完整审核流程，最后 assert inbox.jsonl + graph.json 状态正确。

---

### 入口 2（辅）：MCP 工具集 — Jei 对话审核

**何时用 MCP 而不是 TUI**：

1. 用户**正在跟 Jei 对话别的话题**，顺手问"对了，刚才 classifier 跑出的那条 ClaudeSDKNode 边怎么样？我应不应该 approve？"——上下文连续，开 TUI 反而打断。
2. 用户对某条边**犹豫**，想让 Jei 把 source 代码和 target 文档**用自然语言对比**："这两个真的描述同一回事吗？"——TUI 只显示文本，Jei 能 reason。
3. 用户**远程或在手机上**，没有终端跑 TUI，只有 Jei chat。

**MCP 工具签名（3 个）**：

### MCP 工具集（3 个，给 Jei 用）

```python
@mcp.tool()
def list_pending_edges(top_n: int = 10, sort_by: str = "confidence") -> str:
    """List pending cross-namespace edges awaiting human review.

    Args:
        top_n: Maximum entries to return (default 10).
        sort_by: "confidence" (default) | "created_at" | "consecutive_seen".

    Returns:
        JSON list of pending entries, each with id, source, target, confidence,
        model_id, top_k_rank, created_at, probe_signals summary.

    USAGE: Call when user asks "list pending edges" / "what needs review".
    Do NOT auto-approve based on this output — only the user decides.
    """

@mcp.tool()
def get_pending_edge_context(edge_id: str) -> str:
    """Get full context for a single pending edge: source code symbol details
    (signature, file location, docstring) and target vault doc summary
    (frontmatter, first 500 chars).

    Args:
        edge_id: The id field from list_pending_edges.

    Returns:
        JSON with code_context (signature, file, line range, docstring excerpt)
        and vault_context (path, frontmatter, content_excerpt).

    USAGE: Call when user asks for details about a specific pending edge
    before deciding whether to approve.
    """

@mcp.tool()
def review_pending_edge(
    edge_id: str,
    decision: str,           # "approve" | "reject"
    note: str = "",
) -> str:
    """Apply a human decision to a pending edge.

    USAGE CONTRACT: This is a HUMAN DECISION. You may ONLY call this when:
      1. The user has explicitly said "approve <id>" or "reject <id>" /
         equivalent unambiguous instruction
      2. You have echoed back what you're about to do (the edge in question
         + the decision) and the user has not retracted

    DO NOT call review_pending_edge to "save the user time" or based on
    your own judgment of edge quality. The whole point of inbox is human
    curation. If the user says "list pending", you list. If they say
    "show details", you fetch. Only "approve X" / "reject X" triggers this.

    Args:
        edge_id: The id from list_pending_edges
        decision: "approve" or "reject"
        note: Optional human-supplied rationale (gets stored in inbox audit)

    Returns:
        JSON {status: ok|error, edge_id, decision, ...}

    Side effects on approve:
      - inbox.jsonl entry status → "approved"
      - vault graph: write mentions_symbol edge (lifecycle_class=ANCHORED)
      - probe entry (if exists): mark ANCHORED
      - Atomic write to both files; server picks up via 500ms mtime
    """
```

### 实现

```python
# prism_rag/inbox/store.py
class InboxStore:
    """JSONL-backed inbox queue. Atomic full-rewrite on status changes."""

    def __init__(self, path: Path):
        self._path = path
        self._entries: list[dict] = self._load()

    def list_pending(self, top_n=10, sort_by="confidence") -> list[dict]:
        pending = [e for e in self._entries if e["status"] == "pending"]
        key = {"confidence": "confidence",
               "created_at": "created_at",
               "consecutive_seen": lambda e: e["probe_signals"][0]["consecutive_seen"]}[sort_by]
        return sorted(pending, key=key, reverse=True)[:top_n]

    def get(self, edge_id: str) -> dict | None: ...

    def append(self, entry, status="pending"): ...

    def update_pending(self, edge_id, new_data): ...

    def set_status(self, edge_id, status, decided_by, decision_note=""):
        e = self.get(edge_id)
        if e["status"] != "pending":
            raise ValueError(f"Cannot transition from {e['status']} to {status}")
        e["status"] = status
        e["decided_at"] = now_iso()
        e["decided_by"] = decided_by
        e["decision_note"] = decision_note

    def save_atomic(self) -> None:
        content = "\n".join(json.dumps(e, ensure_ascii=False) for e in self._entries)
        atomic_write(self._path, content + "\n")


# prism_rag/mcp_server/server.py
@mcp.tool()
def review_pending_edge(edge_id: str, decision: str, note: str = "") -> str:
    if decision not in ("approve", "reject"):
        return json.dumps({"status": "error", "msg": "decision must be approve or reject"})

    settings = PrismRagSettings()
    inbox = InboxStore(settings.data_dir / "inbox.jsonl")
    entry = inbox.get(edge_id)
    if entry is None:
        return json.dumps({"status": "error", "msg": f"edge_id not found: {edge_id}"})
    if entry["status"] != "pending":
        return json.dumps({"status": "error", "msg": f"already {entry['status']}, cannot re-decide"})

    if decision == "approve":
        # 写正式 mentions_symbol ANCHORED 边
        fg = _ensure_federated()
        sem_src = entry["source"]
        sem_tgt = entry["target"]
        vault_g = fg.get_graph("nimbus")
        bare_src = sem_src.split("::", 1)[1]
        edge = Edge(
            source=bare_src,
            target=sem_tgt,
            relation="mentions_symbol",
            confidence="INFERRED",
            confidence_score=float(entry["confidence"]),
            source_pass="conv",
            lifecycle_class=LifecycleClass.ANCHORED,
        )
        vault_g.add_edge(edge)
        atomic_save_graph(vault_g, settings.resolved_graphs[0].graph_path)
        # 同步 probe（如果还在内存）
        if _cross_ns_probe is not None and edge_id in _cross_ns_probe._index:
            _cross_ns_probe._index[edge_id].lifecycle_class = LifecycleClass.ANCHORED

    inbox.set_status(edge_id, "approved" if decision == "approve" else "rejected",
                     decided_by="user_via_mcp",
                     decision_note=note)
    inbox.save_atomic()

    return json.dumps({
        "status": "ok",
        "edge_id": edge_id,
        "decision": decision,
        "note": note,
    })
```

**Trigger C 的副作用链极短**：MCP call → 修改 jsonl + graph.json（atomic）→ server 500ms mtime 探测自动 reload。

**没有 hash chain**——MCP 调用本身就是触发器，没有"某个 vault 文件被改 → hook → 修改 → 又触发 hook"的循环可能。

---

## 七、CLI 工具

CLI 是审核的**第三类入口**：脚本化 / CI / 程序化批量。所有命令复用 `InboxStore` + `apply_decision`（与 TUI、MCP 共享底层）。

```bash
# 分类（写入端）
prism-rag classify-edges                          # 一次性跑全 probe log → 三档分类

# ── 审核入口 ──
prism-rag inbox review                            # ★ 启动 TUI（B1 主入口，详见 §六 入口 1）
                                                  #   全屏 textual 界面，键盘 j/k/a/r/n 操作

# ── 查询 / 程序化 ──
prism-rag inbox                                   # list pending（默认）—— 纯文本输出
prism-rag inbox --status approved                 # 查看历史
prism-rag inbox --status all --top 50

prism-rag inbox show <edge-id>                    # 详情（同 MCP get_pending_edge_context 的输出）

# ── 单条决策（脚本可调，幂等）──
prism-rag inbox approve <edge-id> [--note "..."]
prism-rag inbox reject <edge-id> [--note "..."]

# ── 批量决策（高置信度自动批，仍要 --yes 确认）──
prism-rag inbox approve-all --min-conf 0.85 --yes  # 批量 approve confidence ≥ 0.85
prism-rag inbox approve-all --top-n 5 --yes        # 批量 approve top-5
```

**输出样例（list pending）**：

```
$ prism-rag inbox --top 5
[源 vault doc]                              [目标 code symbol]                            [conf] [rank] [consec]
设计细节/Hani 设计文档                       framework/nodes/llm/claude.py::ClaudeSDKNode  0.745  1      3
设计细节/Apex Coder 设计                     blueprints/.../apex_coder_state.py            0.732  1      2
...
共 12 条 pending（confidence ≥ 0.70）

提示：跑 `prism-rag inbox review` 用 TUI 批量审，或与 Jei 对话审。
```

**入口对照速查（在用户文档里印一份）**：

| 你想做的 | 用 |
|---------|----|
| 一次性过完今天 classifier 产出的 30 条 | `prism-rag inbox review` |
| 跟 Jei 聊天时顺便问"刚才那条怎么样" | Jei MCP（直接说"approve XXX"）|
| CI 脚本里把 confidence > 0.85 的全批 | `prism-rag inbox approve-all --min-conf 0.85 --yes` |
| 把某条边的详情贴给同事看 | `prism-rag inbox show <id>` |
| 看历史决策审计 | `prism-rag inbox --status all` |

---

## 八、历史数据迁移

### probe log 缺失字段（自动加载默认值）

| 字段 | 加载默认 |
|------|---------|
| `consecutive_seen` | 1 |
| `last_seen_parsed_at` | `MIGRATION_PENDING` 哨兵 |
| `last_seen_at` | 同 `first_seen_at` |
| `source_file` | 从 `source_node` 反推：`code::path/file.py::Class` → `path/file.py` |
| `model_id` | 从 `settings.embedding_model` 读 |
| `lifecycle_class` | `PROBABILISTIC` |

哨兵机制：classifier 见 `last_seen_parsed_at == MIGRATION_PENDING` → 短路返回（不参与 tier 升级）。首次 `record()` 命中此分支 → 覆盖为真实 timestamp + 重置 consecutive_seen=1（首次真实确认）。

### vault graph.json 历史 mentions_symbol 边（一次性迁移脚本，hard-fail）

```python
# prism-rag migrate-lifecycle 命令
def migrate_lifecycle_class(graph_paths: list[Path]) -> MigrationReport:
    """对每条边按来源强制对齐 lifecycle_class（hard-fail，不 fallback）。

    强制对齐规则：
      1. relation == "mentions_symbol" + source_pass in ("ast", "code")  → DETERMINISTIC
         （这是 v5.1a SymbolLinker 写的）
      2. relation == "mentions_symbol" + source_pass == "conv"            → ANCHORED
         （v5.2 approved/auto_promoted；新生成的边用此 source_pass）
      3. 其他 mentions_symbol 边                                          → 异常阻断
         不允许 fallback——必须人工分类后再跑

    blocked > 0 时整个迁移回滚（atomic 没 commit），人工修完再重跑。
    脚本幂等：第二次跑应该 0 migrated。
    """
```

**为什么 hard-fail**：fallback 猜测会把性质不明的边静默标成某一类，未来 v5.1a/sweep 按这个标记决定是否删边——错标会导致永久数据损失。

调用时机：Sprint 1 完成、Sprint 2 启动之前。

---

## 九、实现模块

### Sprint 1 — 协议层（约 280 行 + 140 行测试）

| 模块 | 路径 | 行数估计 |
|------|------|---------|
| LifecycleClass enum + Edge.lifecycle_class | `prism_rag/store/graph.py` | ~30 |
| Atomic write 协议 | `prism_rag/store/graph.py`（KnowledgeGraph.save 改造） | ~30 |
| 500ms mtime 探测 | `prism_rag/store/federated.py`（_maybe_reload） | ~50 |
| sweep_deleted_files | `prism_rag/ingest/incremental.py` | ~50 |
| 一次性迁移脚本 | `prism_rag/cli.py`（migrate-lifecycle 命令） | ~70 |
| atomic_write 工具 | `prism_rag/utils/io.py`（新文件） | ~20 |
| Settings 扩展 classifier_profiles | `prism_rag/config.py` | ~30 |
| 测试 atomic_write | `tests/test_atomic_write.py` | ~50 |
| 测试 mtime reload | `tests/test_mtime_reload.py` | ~50 |
| 测试 migrate-lifecycle | `tests/test_migrate_lifecycle.py` | ~80 |

### Sprint 2 — 业务层（约 1200 行 + 600 行测试）

| 模块 | 路径 | 行数估计 |
|------|------|---------|
| Probe 重构 | `prism_rag/store/cross_namespace_probe.py` | ~150 |
| InboxStore（jsonl 读写）| `prism_rag/inbox/store.py`（新文件）| ~120 |
| Approval 共享逻辑（TUI/MCP/CLI 三入口共用）| `prism_rag/inbox/approval.py`（新文件）| ~80 |
| EdgeClassifier 核心 | `prism_rag/ingest/edge_classifier.py` | ~250 |
| 3 个 MCP 工具 | `prism_rag/mcp_server/server.py` | ~150 |
| **TUI（textual app）— 主审核入口** | `prism_rag/inbox/tui.py`（新文件）| **~300** |
| `classify-edges` CLI | `prism_rag/cli.py` | ~30 |
| `inbox review` CLI（启动 TUI）| `prism_rag/cli.py` | ~15 |
| `inbox` CLI（list / show / approve / reject / approve-all）| `prism_rag/cli.py` | ~120 |
| v5.1a SymbolLinker 守卫 | `prism_rag/ingest/symbol_linker.py` | ~15 |
| 新依赖 `textual>=0.50` | `pyproject.toml` | ~3 |
| 测试 Probe record + sweep | `tests/test_cross_namespace_probe.py` | ~80 |
| 测试 EdgeClassifier 三档 | `tests/test_edge_classifier.py` | ~150 |
| 测试 InboxStore | `tests/test_inbox_store.py` | ~80 |
| 测试 MCP review_pending_edge | `tests/test_review_pending_edge.py` | ~100 |
| **测试 TUI（textual Pilot）** | `tests/test_inbox_tui.py` | **~120** |
| 测试 lifecycle_class 端到端 | `tests/test_lifecycle_class.py` | ~120 |
| 测试 migration_pending 哨兵 | `tests/test_migration_pending.py` | ~80 |
| 测试 inbox CLI | `tests/test_inbox_cli.py` | ~80 |

**合计：~1820 行**（Sprint 1 ~420 + Sprint 2 ~1820 = ~2240 含测试）

vs v1 方案（vault-inbox）的 ~2160 行——基本持平。多出来的 TUI ~420 行（~300 实现 + ~120 测试）换掉了 v1 的 vault-inbox 全部 accidental 复杂度（system namespace、proposed_mention 弱边、Embedder filter、annotation 回写、hash chain 防护，约 ~660 行）。

**净对比**：v1 ~2160 行带 19 行副作用；v2+TUI ~2240 行带 10 行副作用。**实现量持平，但维护负担少 ~50%**。

---

## 九-bis、副作用链与防护汇总

| 潜在副作用 | 阻断机制 |
|----------|---------|
| Classifier 跨进程写 graph.json/inbox.jsonl 时被 server 撕裂读 | atomic write（tmp + os.replace）保证字节级原子 |
| approve 后 server 缓存陈旧，Jei 下次查询拿旧数据 | 500ms mtime 探测，最多 500ms 一致性窗口 |
| 模型从 bge-m3 切换到 qwen3，旧 confidence 不可比 | model_id 不匹配时重置 consecutive_seen=1，confidence 用新值；profile 同时按新 model_id 切换 |
| 同 source_file 多次 ingest 重复累计 consecutive_seen | scan_timestamp 幂等（同一 scan 内 no-op） |
| 长 server 跑 → 同一 batch_id 永远不变 → consecutive 永不增长 | 已废弃 batch_id；scan_timestamp per-file 提供天然累积单元 |
| ANCHORED 边被 sweep 静默清零 | sweep / record 守卫：lifecycle_class != PROBABILISTIC 跳过 |
| v5.1a SymbolLinker 全量覆写删 v5.2 approved 边 | _clear_existing_mentions 加 ANCHORED 守卫；只清 DETERMINISTIC |
| 历史 98 条 log 在迁移后首次 classify-edges 时给噪声边凭空信用 | MIGRATION_PENDING 哨兵；classifier 见此值短路；首次 ingest 同 source_file 时哨兵被覆盖 |
| 迁移脚本 fallback 猜测把未知来源边错标 → 永久数据损失 | hard-fail 模式，blocked > 0 整体回滚，要人工分类后再跑 |
| Jei 误调 review_pending_edge 替用户做决定 | MCP 工具 docstring 明确 USAGE CONTRACT：必须用户明确 "approve X" 才调；prompt 工程也强约束 |
| TUI 审到一半 Ctrl-C 退出 → 部分决策落盘部分丢失 | 决策先在内存累积，按 `q` 时一次性 atomic 写盘；SIGINT 触发 _emergency_save 把已审条目落盘、未审条目维持 pending |
| TUI 与 CLI 同时审同一 entry → 后写覆盖前写 | InboxStore.set_status 检查 status != "pending" 时抛 ValueError；最差情况是后入口拿到 already-decided 错误，不会破坏数据 |
| TUI 与 MCP 同时持有 InboxStore 内存副本 → 决策互相不可见 | 两者每次写盘前先重新 load(jsonl)；TUI 启动时一次性 snapshot 但 q-save 前 reload 合并；MCP review_pending_edge 每次调用都 fresh load |

**10 行 vs v1 方案 19 行**——少的 9 行全是 vault-inbox 引入的 accidental 复杂度（hash chain death loop、weak-edge feedback、annotation 回写、embedder 污染、system namespace GC、proposed_mention 噪声等）。

---

## 十、Out of Scope（YAGNI）

- ❌ **Inbox 老化（TTL）**：终点状态（approved/rejected/discarded/auto_promoted）保留多久后清理。**TODO（v5.3 候选）：90 天 TTL**。pending 状态没有老化（人工不审就是不审，不要自动消失）。
- ❌ **撤回 approve**：approved 后又想撤回，需要手工删 vault graph 的 mentions_symbol 边。够罕见，不预留 API。
- ❌ **多源信号融合**：当前 probe 只有 `embedding_similar`。未来加 `shared_tag` 等再扩展评分函数。
- ❌ **Tier 1 → EXTRACTED 升级**：当前固定 INFERRED。等校准数据足够再考虑。
- ❌ **Web UI（HTTP 服务）**：TUI + Jei MCP + CLI 三入口已覆盖批量、对话、脚本三类场景。Web UI 是独立项目级别工作（前端框架、登录、状态同步），超出 PrismRag 范围。
- ❌ **Obsidian 直接渲染 inbox**：v1 方案的初衷，但代价是 ~10 条 accidental 复杂度。本 v2 方案用 TUI（主）+ Jei 对话（辅）+ CLI（脚本）替代——损失了 Obsidian 的 wiki 链接渲染体验，换来 ~50% 副作用链削减、~30% 实现量降低（含 TUI 后基本持平）和零 vault-as-state-machine 风险。决策已做，不再回头。
- ❌ **TUI 内调用 Jei**："在 TUI 里按某键让 Jei 解释这条边"——跨进程同步对话流，复杂度暴涨。需要时直接退出 TUI 切到 Jei chat。两个工具各自专注。
- ❌ **TUI 多人协作**：TUI 是单用户本地工具，不做并发审核协调（atomic write 保证不撕裂，但两人同时审同一条会有最后写赢覆盖问题）。当前 PrismRag 是单用户场景，YAGNI。

---

## 十一、测试

### Sprint 1

- `tests/test_atomic_write.py`：tmp + os.replace 在并发读写下不撕裂；写入失败时 tmp 文件被清理
- `tests/test_mtime_reload.py`：500ms 内多次查询共享一次 stat；mtime 变化触发 lazy reload；连续 100 次查询 stat 调用 ≤ 2 次
- `tests/test_migrate_lifecycle.py`：v5.1a 历史边正确归 DETERMINISTIC；未知 source_pass 边阻断；blocked > 0 时整体回滚不写盘；幂等

### Sprint 2

- `tests/test_cross_namespace_probe.py`（追加）：record 五分支（new / MIGRATION_PENDING 覆盖 / model_id 切换 / 新 scan / 同 scan no-op / ANCHORED 跳过）；sweep 只清 PROBABILISTIC

- `tests/test_edge_classifier.py`：
  - 三档判定（包括 margin+hard_cap 边界）
  - profile 切换（同样 confidence 在不同 model 下落到不同 tier）
  - MIGRATION_PENDING 短路（首次跑全 noop）
  - Tier 2→1 跃迁（inbox 中对应 entry status 改 auto_promoted）
  - Tier 2→3 rollback（inbox status 改 discarded，仅在 pending 时；approved/rejected 不会被改）
  - upsert 幂等（重复跑 pending entry 只更新 confidence，不重置 status）

- `tests/test_inbox_store.py`：append-only 行为；status transitions；atomic save；NDJSON 格式正确性；并发读取（mtime 触发的 reload）

- `tests/test_review_pending_edge.py`：
  - approve → 写 ANCHORED 边到 vault graph + inbox status=approved + atomic
  - reject → inbox status=rejected，无图变化
  - 重复 review 同 entry → error（status 已不是 pending）
  - 不存在的 edge_id → error
  - decision 字符串非法 → error
  - 模拟用户对话场景：list → show → approve（正确路径），以及 list 后 Jei 不该自动 approve

- `tests/test_lifecycle_class.py`：
  - ANCHORED 边在 sweep / record / SymbolLinker / classifier rollback 全部不动
  - DETERMINISTIC 边被 SymbolLinker 全量覆写正常处理
  - PROBABILISTIC entries 被 sweep 清零
  - 端到端 regression：approve → classifier 重跑 → assert ANCHORED 边和 probe entry 仍存在（即使 probe 不再召回）

- `tests/test_migration_pending.py`：历史 log 加载默认值；首次 classify-edges 全 noop；首次 record 哨兵被覆盖；同 scan_timestamp 幂等；不同 scan_timestamp +1

- `tests/test_inbox_cli.py`：list / show / approve / reject / approve-all 子命令；与 MCP 共享 InboxStore 状态

- `tests/test_inbox_tui.py`（用 textual `Pilot` 模拟键盘）：
  - 启动 TUI → 显示第一条 pending（idx=0/12）
  - 按 `j` 三次 → idx=3
  - 按 `a` → 决策记入内存 _decisions，**不写盘**
  - 按 `n` → note 编辑器打开 → 输入文字 → 保存
  - 按 `r` → reject 决策记入内存
  - 按 `q` → 触发 atomic 写盘 → assert inbox.jsonl + graph.json 状态正确
  - 模拟 SIGINT → 触发 _emergency_save → assert 已审条目落盘、未审条目仍 pending
  - 共享 InboxStore：TUI 写完，CLI `prism-rag inbox --status approved` 立刻能列出

---

## 十二、与 v5.1 / v5.0 的关系

```
v5.0  CrossNamespaceProbe（被动收集）           ← v5.2 的输入源
v5.1a mentions_symbol（精确匹配, DETERMINISTIC） ← v5.2 的输出对象（同种边）
v5.1b L2 语义层（推迟）                          ← 未来可与 v5.2 Tier 2 合并
v5.2  EdgeClassifier + jsonl inbox + TUI/MCP/CLI ← 本文
```

v5.2 不替换 v5.1a，而是**补充**：
- v5.1a 是基于精确字符串匹配的高置信度边（INFERRED tier，DETERMINISTIC lifecycle，立刻入图，全量覆写管理）
- v5.2 是基于 embedding 相似度的低置信度建议（INFERRED tier，ANCHORED lifecycle，自动 OR 人审入图，永久不被自动改写）

两者最终产出同一种 `mentions_symbol` 边，但 lifecycle_class 不同：v5.1a→DETERMINISTIC，v5.2→ANCHORED。

---

## 十三、实现里程碑（Sprint 1/2 拆分）

v5.2 拆成两个独立 Sprint，**协议层先于业务逻辑层**。Sprint 1 锁死的物理协议在 Sprint 2 中不允许变更。

### Sprint 1 — 底层协议层（数据一致性保证）

**目标**：固定 graph.json + inbox.jsonl 的物理状态转换协议。完成后系统行为不变（v5.2 业务逻辑还没接入），但底层契约成型。

| 工作项 | 验收 |
|--------|------|
| LifecycleClass enum + Edge 字段 | 现有 320 测试全绿 |
| Atomic write 协议 | 并发读写不撕裂测试通过 |
| 500ms mtime 探测 | 多次查询共享 stat 测试通过 |
| sweep_deleted_files | 物理文件删除自动清理节点 |
| 一次性迁移脚本 | 跑一次：所有 mentions_symbol 边获得 lifecycle_class，0 blocked |

Sprint 1 完成后即使 v5.2 永远不上线，PrismRag 也已经多了 atomic write + mtime reload + lifecycle_class——独立的基础设施改进，本身有价值。

### Sprint 2 — 业务逻辑层（v5.2 完整特性）

**目标**：在 Sprint 1 协议上构建分类器、inbox 队列、三个审核入口（TUI/MCP/CLI）。

| 工作项 | 依赖 | 入口归属 |
|--------|------|---------|
| Probe 重构 | Sprint 1 lifecycle_class enum | （核心）|
| InboxStore | Sprint 1 atomic write | （共享）|
| Approval 共享逻辑 | InboxStore | （共享）|
| EdgeClassifier 核心 | Sprint 1 atomic write | （核心）|
| **TUI（textual）— 主审核入口** | InboxStore + Approval | TUI |
| 3 个 MCP 工具 | InboxStore + Sprint 1 mtime reload | MCP |
| `classify-edges` CLI | EdgeClassifier | （核心）|
| `inbox review` CLI | TUI | TUI |
| `inbox` CLI（list/show/approve/reject/approve-all）| InboxStore + Approval | CLI |
| v5.1a SymbolLinker 守卫 | Sprint 1 lifecycle_class enum | （核心）|

**Sprint 2 验收**：
- Sprint 1 协议层测试不允许失败
- 跑首次 `prism-rag classify-edges`：0 promoted、0 queued、0 discarded（MIGRATION_PENDING 全短路）
- 跑一次 `prism-rag ingest`（覆盖部分文件）后再跑 classify-edges：MIGRATION_PENDING 哨兵被覆盖的文件对应 entries 进入正常分类，预计 15-30 条进 inbox
- **TUI 验收**：`prism-rag inbox review` 启动 → textual 全屏 → j/k/a/r/n/q 键全部生效 → 退出后 atomic 写一次 inbox.jsonl + graph.json → server 500ms 内看到新边
- **Jei MCP 验收**：用户 "列出待审" → Jei 调 list_pending_edges → "details on first one" → Jei 调 get_pending_edge_context → "approve" → Jei 调 review_pending_edge → 500ms 内 server 看到新 mentions_symbol 边
- **CLI 验收**：`prism-rag inbox approve <id> --note "..."` 幂等；二次调同 id 报 already-decided
- **三入口共享一致性**：TUI 写入后，CLI `inbox --status approved` 立刻能列出；同一条 entry 不能被两个入口同时审（最后写赢，但 status 转换检查防止 approved → reject）
- **Regression**：approve → classifier 重跑（模拟 probe 数据漂移）→ ANCHORED 边和 probe entry 仍存在

### 为什么协议先行

Sprint 1 的 atomic write + mtime reload + lifecycle_class 是 PrismRag 通用基础设施，任何后续工作都受益。Sprint 2 在这之上构建，不会反过来改 Sprint 1。这避免了"业务逻辑变更倒推协议变更"的雪崩。
