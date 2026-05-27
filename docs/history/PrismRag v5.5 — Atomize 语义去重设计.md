---
title: PrismRag v5.5 — Atomize 语义去重设计
created: 2026-05-16
tags: [prismrag, atomize, dedup, design]
status: design
---

# PrismRag v5.5 — Atomize 语义去重设计

## 一、背景与问题

v5.4 的 atomize pipeline（scan → propose → apply）只有 ID 层幂等保护：

- **同 proposal 内**：`atomize_propose` 按 `knowledge_id` 去重，first wins
- **崩溃恢复**：`atomize_apply` 读源文档 `atomized_nodes` frontmatter，已有 KNOW-ID 跳过写文件

缺口：**语义重复**。同一文档 atomize 两次、或两篇文档描述同一概念，都会生成内容高度相似的独立 KNOW 节点，图里出现语义孪生节点，靠 `semantically_similar_to` 边（score ≈ 0.9）标记但不合并。

---

## 二、总体方案

**在 propose 阶段做 batch embedding 检索，附加 `similar_existing` 软警告，Jei 分片决策（每 5 claims 一组），reuse 时建立溯源边 + 快照回滚点。**

```
atomize_scan
     ↓
atomize_propose
  ├─ batch embed claims (body)
  ├─ batch ANN search vs existing KNOW nodes
  ├─ similar_existing > threshold → 附加到 claim 结果
  └─ Jei 分片判断（每 5 claims）：reuse / 新建
     ↓
atomize_apply
  ├─ reuse 路径：建立 MENTIONS 边 + CONTEXT_REF 节点
  ├─ 新建路径：正常写 KNOW 文件
  └─ 每次 reuse 写 dedup_snapshot（可回滚）
```

---

## 三、五个议题定案

### P1 — 阈值

**不硬编码。默认 fallback 0.90，支持 `prism calibrate` 动态标定。**

- 冷启动（知识图谱 < 100 节点）：直接用 fallback 0.90，不做自标定，避免冷启动死锁
- 节点量达到阈值后：`prism calibrate` 抽样标注对（重复 vs 非重复），拟合最优阈值
- 配置文件记录 `calibrated_model`，embedding 模型变更时强制重新标定

```toml
# prism_rag.toml
[dedup]
threshold = 0.90          # fallback，calibrate 后自动更新
calibrated_model = ""     # 空 = 未标定，变更时触发重新标定
min_nodes_for_calibration = 100
```

### P2 — 检查时机

**propose 阶段，batch 模式。**

- 一次 embedding API 调用（所有 claim body 打包）
- 一次 batch ANN 查询（vs EmbeddingStore 中 knowledge namespace）
- 无冲突的 claims 直接 pass-through，零额外开销
- 仅 `similar_existing` 非空的 claims 触发后续 Jei 判断

预防优于事后修复：apply 之后合并是破坏性操作（需重定向边、废弃 ID、清缓存），propose 阶段的「不建」是零成本。

### P3 — Jei 决策能力

**可靠，但必须分片：每次最多 5 个 claims 对比判断。**

- 40 claims 并发判断有 Lost-in-the-Middle 认知超载风险
- 5 claims/轮 ≈ 2000 tokens 对比上下文，远低于注意力衰减区间
- 大笔记拆成多轮 LLM 调用，牺牲少量延迟换取判断精度

Jei 判断时收到的信息结构：

```json
{
  "claim": {
    "knowledge_id": "KNOW-000031",
    "title": "LangGraph 节点与边",
    "body": "LangGraph 是基于有向图的 LLM 编排框架..."
  },
  "similar_existing": [
    {
      "id": "KNOW-000008",
      "score": 0.93,
      "title": "PrismRag Embedding 层次",
      "body_preview": "PrismRag 支持两个层次的多模态 emb..."
    }
  ]
}
```

Jei 的决策：
- **reuse**：不新建，返回 `{"action": "reuse", "reuse_id": "KNOW-000008"}`
- **新建**：认为概念不同，返回 `{"action": "create"}`

### P4 — 跨文档重复

**reuse 时建立溯源边，用 CONTEXT_REF 中间节点隔离局部关系，防止超级节点膨胀。**

- 建立 `(源文档 note 节点)-[:MENTIONS]->(existing_KNOW)` 边，标记跨文档引用关系
- 跨文档的局部关系（"在本文语境下 X 依赖 Y"）挂在轻量 `CONTEXT_REF` 中间节点，而非直接修改被复用节点的拓扑
- `CONTEXT_REF` 携带 `source_doc`、`context_note` 字段，不影响被复用 KNOW 节点的全局语义

```
[note:A] -[:MENTIONS]-> [KNOW-000008]
[note:B] -[:MENTIONS]-> [KNOW-000008]
              ↑
         CONTEXT_REF_A (source=note:A, context="在 v5.3 设计中 X 依赖此概念")
         CONTEXT_REF_B (source=note:B, context="在多模态路线图中提及")
```

### P5 — 人工审核

**交互模式：展示 diff，超时 auto-create（保守不合并）。Bulk 模式：全自动 + dedup_snapshot 可回滚。**

| 模式 | 行为 |
|------|------|
| 交互（单篇） | 有冲突时展示 claim vs existing 的内容 diff；超时自动新建（保守策略） |
| Bulk | Jei 全自动决策；每次 reuse 写带时间戳的 `dedup_snapshot` |

`dedup_snapshot` 格式（非纯 log，可回滚操作记录）：

```json
{
  "decision_id": "uuid",
  "timestamp": "2026-05-16T12:00:00Z",
  "action": "reuse",
  "claim_title": "LangGraph 节点与边",
  "reused_id": "KNOW-000008",
  "source_doc": "实验/atomize-test-doc.md",
  "similarity_score": 0.93,
  "pre_state": { "mentions_edges": [] }
}
```

回滚命令：`prism rollback --decision-id <uuid>`

---

## 四、实现变更点

| 文件 | 变更 |
|------|------|
| `prism_rag/ingest/atomize.py` | `atomize_propose_impl`：加 batch embed + ANN 检索，claim 结果附加 `similar_existing` |
| `prism_rag/mcp_server/server.py` | `atomize_propose` tool 返回结构更新，文档说明 `similar_existing` 字段 |
| `prism_rag/mcp_server/server.py` | `atomize_apply` 支持 `reuse` action：建 MENTIONS 边 + CONTEXT_REF 节点 |
| `prism_rag/ingest/dedup_log.py` | 新建：`dedup_snapshot` 写入 / 读取 / 回滚逻辑 |
| `prism_rag/config.py` | `PrismRagSettings` 加 `[dedup]` 配置块 |
| `prism_rag/cli/calibrate.py` | 新建：`prism calibrate` 命令 |

---

## 五、风险与缓解

| 风险 | 缓解 |
|------|------|
| Jei 幻觉导致误 reuse（Bulk 模式） | dedup_snapshot 支持 `prism rollback --decision-id` 精确单条回滚 |
| Embedding 模型更换后阈值失效 | `calibrated_model` 字段记录，模型变更时强制重新标定 |
| 分片多轮调用增加延迟 | 仅 `similar_existing` 非空 claims 触发 LLM 判断，无冲突零开销 |
| CONTEXT_REF 节点膨胀 | CONTEXT_REF 不参与 Leiden 聚类，不计入 community 统计 |

---

## 六、破局点

这个方案的真正价值不在于「去重」，而在于让知识图谱从「文档的被动映射」进化为「概念的主动织网」。每一次 reuse + MENTIONS 边的建立，都是在不同文档间架设一座桥梁——这是纯 ID 幂等永远做不到的事。语义去重只是手段，跨文档知识网络才是目的。

---

## 七、Decision Log

| ID | 决策 | 理由 |
|----|------|------|
| D1 | 检查时机选 propose 而非事后看板 | 事后合并是破坏性操作，预防成本为零 |
| D2 | 阈值 0.90 fallback + 动态标定 | 硬编码在模型更换后失效；冷启动 <100 节点直接用 fallback |
| D3 | Jei 分片 5 claims/轮 | 40 claims 并发有 Lost-in-the-Middle 风险；5 claims ≈ 2000 tokens 安全区间 |
| D4 | CONTEXT_REF 中间节点隔离局部关系 | 直接挂边会制造超级节点；中间节点隔离跨文档局部语义 |
| D5 | 软警告非硬阻断 | 相似内容不代表重复；Jei 应有最终判断权 |
| D6 | dedup_snapshot 可回滚 | Bulk 模式无人审核，需要精确撤销能力 |
