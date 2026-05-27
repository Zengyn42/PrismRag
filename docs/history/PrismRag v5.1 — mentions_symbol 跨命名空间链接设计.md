---
title: "PrismRag v5.1 — mentions_symbol 跨命名空间链接设计"
tags:
  - "#PrismRag"
  - "#GraphRAG"
  - "#architecture"
  - "#cross-namespace"
  - "#design"
status: implemented
created: 2026-05-02
last_audited: 2026-05-02
milestone: v5.1a
github_issue: "https://github.com/Zengyn42/PrismRag/issues/10"
note: "v5.1a = 边创建层（精确匹配）；v5.1b = L2 语义层（推迟，待真实数据驱动）；v5.2 = CrossNamespaceProbe 观察层（待激活条件满足）"
related:
  - "设计细节/PrismRag v5.0 — 详细设计与任务分解.md"
  - "设计细节/PrismRag v5.0 — 通用图引擎架构设计.md"
---

# PrismRag v5.1 — `mentions_symbol` 跨命名空间链接设计

## 动机

PrismRag v5.0 实现了 `nimbus::` 和 `code::` 两个 namespace 的独立索引与联合搜索（`scope` 参数、FederatedGraph embedding bridge）。但两个 namespace 之间没有**结构性连接**：系统不知道"这篇设计文档描述的是这个函数"。

这导致：
- 无法检测设计文档与代码实现之间的漂移
- BFS 从文档节点无法跳转到它描述的代码节点（反之亦然）
- 代码变更时无法定位哪些文档可能已过时

`mentions_symbol` 边解决这个问题：在 vault 文档提到代码符号名时，建立一条从文档节点到代码节点的跨 namespace 边。

---

## 核心概念

```
nimbus::设计细节/PrismRag-Jei整合路线图.md
    │
    │  mentions_symbol (INFERRED, score=0.9)
    ▼
code::framework/loader.py::build_graph
```

有了这条边之后：
- `search_knowledge("build_graph 设计背景")` 的 BFS 能从代码节点跳到设计文档
- `impact(target="build_graph", direction="upstream", allowedEdgeKinds=["mentions_symbol"])` 能找到所有描述它的文档
- 增量 ingest 检测到 `build_graph` 代码变化时，能自动给相关文档打 `stale_refs` 标记

---

## 匹配策略（两级）

### Level 1：Wikilink 精确引用（EXTRACTED / 1.0）

vault 笔记里出现 `[[build_graph]]` 或 `[[ClaudeSDKNode]]`——作者的**显式意图声明**，与 AST `import` 语义等价，最高置信度。

```markdown
详见 [[build_graph]] 的实现。
```

> **为什么是 EXTRACTED**：wikilink 是作者的主动声明，不是系统推断。这与 Tree-sitter 解析 `import` 语句的认识论地位相同——解析是确定性的，意图是显式的。

### Level 2：Symbol 名称词边界匹配（INFERRED / 0.9）

正文里出现符号名，使用正则词边界匹配：`\bsymbol_name\b`

**过滤条件（避免通用短词爆炸）：**
- symbol 名长度 **≥ 5 字符**
- 排除 Python 内置名黑名单：`range`, `print`, `super`, `self`, `True`, `False`, `None`, `class`, `async`, `yield` 等
- 匹配结果在当前 code namespace 内**全局唯一**：同名符号存在于多个模块时，降级为 AMBIGUOUS → inbox

> **为什么是 INFERRED 而非 EXTRACTED**：精确字符串匹配是确定性的，但"这篇文档**描述**这个函数"的语义是推断——文档可能只是在对比或批评这个符号。安全性依赖的是 code namespace 的项目边界约束，而非文本本身的显式声明。铁律不破。

### Level 3：Embedding 相似度

→ **不做**。已由 FederatedGraph serve-time embedding bridge 覆盖，不重复实现。

---

## 边 Schema

```python
# wikilink 匹配
EdgeRecord(
    source_id        = "nimbus::设计细节/PrismRag-Jei整合路线图.md",
    target_id        = "code::framework/loader.py::build_graph",
    kind             = "mentions_symbol",
    confidence_tier  = "EXTRACTED",
    confidence       = 1.0,
    weight           = 1.0,
    evidence         = ["wikilink: [[build_graph]]"],
)

# 精确符号名匹配（全局唯一）
EdgeRecord(
    source_id        = "nimbus::设计细节/PrismRag-Jei整合路线图.md",
    target_id        = "code::framework/loader.py::build_graph",
    kind             = "mentions_symbol",
    confidence_tier  = "INFERRED",
    confidence       = 0.9,
    weight           = 1.0,
    evidence         = ["exact match: 'build_graph' (occurrences=3)"],
)

# 歧义匹配（同名符号存在于多个模块）
EdgeRecord(
    source_id        = "nimbus::设计细节/...",
    target_id        = "code::...",   # 所有候选 target 各建一条
    kind             = "mentions_symbol",
    confidence_tier  = "AMBIGUOUS",
    confidence       = 0.2,
    weight           = 1.0,
    evidence         = ["ambiguous: 'parse' matched 3 symbols"],
)
# → 同时写入 inbox，等人工确认
```

---

## 边的存储：直接写入 vault `graph.json`

### 决策

mentions_symbol 边作为 vault 文档节点的 **outbound edge**，直接存入 vault 端的 `graph.json`。

**不使用独立的 cross_links.json 文件。**

### 理由

独立文件方案存在"幽灵边"问题：
```
文档删除 → re-ingest 刷新 vault graph.json（文档节点消失）
                                    ↓
                       cross_links.json 仍残留指向已删节点的悬空边
                       → 每次局部 ingest 需捆绑全局 link pass 才能清理
```

边写入 vault graph.json 后，生命周期与文档**完全绑定**：
- 文档删除 → re-ingest → vault graph.json 重建 → mentions_symbol 边自动消失
- 零额外文件，零同步问题，零缓存一致性负担

### FederatedGraph 集成

```python
# federated.py serve 时加载——无需改动
fg = FederatedGraph.load([vault_graph_path, code_graph_path])
# mentions_symbol 边已在 vault_graph_path 的 graph.json 中
# FederatedGraph 加载时自动合并进 unified DiGraph，BFS/DFS 可跨 namespace 遍历
```

`load_cross_links()` 方法**不需要添加**。

---

## 计算时机：独立的 link pass

两个 namespace 独立 ingest，互不知道对方的节点。`mentions_symbol` 边是第三方产物，在两边都 ingest 完成后计算，结果**写回** vault graph.json。

```
prism-rag ingest          → nimbus graph.json（vault 文档）
prism-rag ingest-code     → code  graph.json（代码符号）
         ↓
prism-rag link-symbols \              ← 新 CLI 命令（幂等）
  --vault-graph /path/nimbus/graph.json \
  --code-graph  /path/zenith_code/graph.json
  # 输出：在 vault graph.json 中追加/更新 mentions_symbol 边
```

`link-symbols` 设计为**幂等**：重复运行只覆盖，不重复追加。

---

## 增量 stale_refs 标记

```
prism-rag ingest-code（增量，检测到 content_hash 变化）
    ↓
对每个 hash 变化的 code 节点
    ↓
加载 FederatedGraph（vault + code graph.json）
    ↓
fg.unified.in_edges(changed_code_node_id, data=True)
    过滤 kind == "mentions_symbol"
    ↓
对每个 source vault 节点，patch frontmatter：

stale_refs:
  - symbol: build_graph
    changed_at: 2026-05-02
    code_node: "code::framework/loader.py::build_graph"
```

Jei 读到含 `stale_refs` 的文档时，知道需要核查代码与文档是否仍然一致。

**无需反向索引缓存文件**：FederatedGraph 内存图的 `in_edges()` 查询已足够，offline CLI 工具加载万级节点 graph.json 仅需毫秒级。

---

## 实现模块

| 模块 | 路径 | 说明 | 行数估计 |
|--|--|--|--|
| `SymbolLinker` | `prism_rag/ingest/symbol_linker.py` | 核心匹配逻辑（wikilink + 精确符号名）+ mentions_symbol 边生成，写回 vault graph.json | ~150 行 |
| `link-symbols` CLI | `prism_rag/cli.py` | 新 typer 命令（幂等） | ~30 行 |
| 增量 stale 标记 | `prism_rag/ingest/incremental.py` | hash 变化时触发，用 FederatedGraph.in_edges() 查询，写 stale_refs frontmatter | ~40 行 |
| 测试 | `tests/test_symbol_linker.py` | linker 单元测试 + stale 标记集成测试 | ~80 行 |
| **合计** | | | **~300 行，约 2 天** |

---

## 实现顺序

```
Step 1  SymbolLinker 核心匹配逻辑
        - 从 code graph.json 提取所有 symbol 名（function/class/module label）
        - 构建符号词典：{short_name → [qualified_id, ...]}
        - 对每个 vault 节点 content，做：
            a. wikilink 扫描：[[X]] → EXTRACTED/1.0
            b. 词边界正则匹配（≥5字符，非黑名单，全局唯一检查）→ INFERRED/0.9
            c. 歧义符号 → AMBIGUOUS/0.2 + inbox
        - 输出 EdgeRecord 列表 → 写回 vault graph.json（追加/覆盖同类边）

Step 2  link-symbols CLI 命令
        - 接受 --vault-graph / --code-graph 参数
        - 调用 SymbolLinker，打印统计（X 篇文档，Y 个符号，Z 条边，W 条歧义进 inbox）

Step 3  增量 stale 标记
        - incremental.py hash 变化检测后，加载 FederatedGraph
        - in_edges() 查 mentions_symbol，调 patch_note 写 frontmatter
```

---

## 已知边界与风险

| 风险 | 缓解 |
|--|--|
| 短符号名误匹配（`load`, `run`）| 名称长度 ≥ 5 + 内置名黑名单 |
| 同名符号歧义（两个 `parse`）| 歧义符号降级 AMBIGUOUS，进 inbox，不建 INFERRED 边 |
| 文档只是"提到"而非"描述"符号 | INFERRED/0.9（非 EXTRACTED），BFS min_confidence 可过滤 |
| vault graph.json 重建覆盖 mentions_symbol 边 | `prism-rag ingest` 结束时自动检测 code graph.json 是否存在，存在则自动触发 link-symbols；否则打印提示引导用户手动运行 |
| vault 未 ingest | link-symbols 检查两个 graph.json 都存在，否则报错退出 |
| 大量 cross-link 边导致 graph.json 体积膨胀 | L1 只产出精确匹配边，数量有限；万级节点 JSON 仍在毫秒级 |

---

## v5.1b — L2 语义层（推迟）

> **启动条件**：NimbusVault 实际 ingest 完成后，用真实数据评估 L1 的召回率盲区，再驱动 L2 设计。

L2 设计方向（留存参考，不在 v5.1a 实现）：

```
L2 输入：L1 未命中的"孤立 block"（无任何 mentions_symbol 边）
    ↓
BM25 召回：block 文本 vs 代码节点 {name + docstring}，取 top-K
    +
图扩展：从同文档已有 L1 边出发，code graph 1-2 跳邻域
    ↓
LLM 确认（候选集收敛至 10-50 个 symbol）
    score ≥ 0.7 → INFERRED mentions_symbol 边
    score < 0.7 → AMBIGUOUS → inbox
```

**不预留 MatchStrategy 抽象层**。等 L2 真正落地时再按实际需求抽象（Rule of Two）。

---

## 与 CrossNamespaceProbe（v5.2）的关系

```
mentions_symbol 边创建（v5.1a）← CrossNamespaceProbe 观察的对象之一（v5.2）
```

v5.2 依赖 v5.1a：没有结构性跨 namespace 边，探测器无东西可观察。
实现顺序：v5.1a → v5.1b（可选）→ v5.2。

v5.2 激活条件：nimbus 节点 > 200 **且** cross_namespace_log > 50 条 **且** 至少 10 条经人工确认有价值。

---

## 启动条件

NimbusVault 尚未 ingest（Jei 集成路线图 Step 4 待做）。
`link-symbols` 实现可以提前完成（用合成测试数据验证），但端到端验证需要 vault ingest 完成后进行。
