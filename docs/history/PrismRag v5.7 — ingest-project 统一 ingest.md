---
title: "PrismRag v5.7 — ingest-project 统一 ingest"
tags:
  - "#PrismRag"
  - "#ingest"
  - "#architecture"
status: confirmed
created: 2026-06-01
---

# PrismRag v5.7 — ingest-project 统一 ingest

## 动机

v5.6 之前，PrismRag 有两条独立的 ingest 路径：

| 命令 | 处理内容 | 缺口 |
|------|----------|------|
| `prism-rag ingest` | Markdown vault（vault_loader） | 无代码节点，无 symbol link |
| `prism-rag ingest-code` | Python 代码（Tree-sitter） | 无可视化输出，无文档节点 |

要得到一个同时包含代码和文档的统一知识图谱，用户需要分别运行两个命令，然后手动合并。两者的 embed cache 也互不相通。

## 设计

新增 `prism-rag ingest-project` 命令，一次执行完成：

1. **Pass 1a** — Markdown AST 抽取（`vault_loader`），生成 doc 节点
2. **Pass 1b** — Python 代码 AST（Tree-sitter），生成 code 节点
3. **Pass 2** — Leiden 社区发现（合并后的完整图）
4. **Pass 3a** — Embedding（Ollama / Gemini），命中已有 code 节点的 embed cache
5. **Pass 3b** — 相似边生成
6. **Pass 3c** — Symbol links（doc → code `mentions_symbol` 边）
7. **Pass 4** — 持久化 `graph.json` + 生成 `graph.html` 交互式可视化

所有输出写入同一个 `<target>/.prismrag/<namespace>/` 目录。

## 核心改进

- **单一 KnowledgeGraph**：code 节点和 doc 节点合并进同一个图，社区发现覆盖全部节点
- **Embed cache 复用**：embed cache 按 `(node_id, content_hash)` 键控，已有代码节点复用向量，只需对新文档增量计算
- **可视化闭环**：`ingest-project` 自动生成 `graph.html`（之前 `ingest-code` 无此功能）
- **文档覆盖**：code-only 图现在也包含 docs 目录中的 Markdown 节点（之前 `ingest-code` 忽略文档）

## 首次验证

Pulsify 项目 ingest 结果：1,684 nodes（1,643 code + 41 docs），95 communities，单一 graph.json + graph.html。

## 文件变更

- `prism_rag/cli.py` — 新增 `ingest-project` CLI 命令
- `prism_rag/ingest/project_ingest.py` — 统一 ingest pipeline 实现
- 现有 `ingest` 和 `ingest-code` 命令保持不变（向后兼容）
