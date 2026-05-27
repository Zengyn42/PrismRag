---
tags:
  - "#PrismRag"
  - "#GraphRAG"
  - "#architecture"
  - "#roadmap"
  - "#implementation-plan"
status: active
created: 2026-04-28
last_audited: 2026-05-04
milestone: v5.0
supersedes: "设计细节/PrismRag v5.0 — 通用图引擎架构设计.md"
---

# PrismRag v5.0 — 详细设计与任务分解

> 本文是 [[PrismRag v5.0 — 通用图引擎架构设计]] 的实施细化版。
> 上层概念见概要文档，本文专注于"怎么做"和"做什么"。

---

## 零、当前实现状态（2026-05-04 审计）

> 审计基准：PrismRag 代码库，320 个测试全绿，通过 GitNexus 代码分析验证。
> **任务追踪权威来源：本节。** `docs/superpowers/plans/` 下的旧 plan 文件已删除（之前标注为 STALE）。

| Phase | 核心任务 | 状态 | 备注 |
|-------|---------|------|------|
| **P1** OllamaEmbedder | OllamaEmbedder class (bge-m3, dim=1024) | ✅ 完成 | embedder.py |
| **P1** | embed_backend config (ollama/gemini) | ✅ 完成 | config.py |
| **P1** | EmbeddingStore 维度安全 | ✅ 完成 | 自动 detect 已有表的 dim，不再 drop+recreate |
| **P2** Hybrid Search | BM25Index.build() / search() | ✅ 完成 | store/bm25_index.py |
| **P2** | hybrid_search() + RRF fusion | ✅ 完成 | retrieve/hybrid.py |
| **P2** | search_knowledge MCP 接入 hybrid_search | ✅ 完成 | mcp_server/server.py L204 |
| **P3** Parser 抽象 | Parser ABC + ParseResult Pydantic | ✅ 完成 | base_parser.py, parse_result.py |
| **P3** | ObsidianParser | ✅ 完成 | obsidian_parser.py |
| **P3** | ConvParser iron law validator | ✅ 完成 | parse_result.py (conv:: 禁止 EXTRACTED) |
| **P4** Edge Confidence | BFS/DFS min_confidence + allowed_tiers | ✅ 完成 | bfs.py, dfs.py |
| **P4** | AMBIGUOUS 边默认排除（BFS） | ✅ 完成 | bfs_traverse / dfs_traverse 默认 frozenset({EXTRACTED,INFERRED})（2026-05-04） |
| **P4** | tag_co→INFERRED/0.7 co-occurrence 边 | ✅ 完成 | ast_extractor.py step 5 |
| **P5** CodeParser | CodeParser + 跨文件 calls 边 | ✅ 完成 | code_parser.py |
| **P5** | MRO 继承链查找 | ✅ 完成 | _mro_lookup() |
| **P5** | 相对 import 解析 | ✅ 完成 | _resolve_relative_import() |
| **P5** | 执行流检测 (kind="flow") | ✅ 完成 | _detect_flows() |
| **P5** | `ingest-code` CLI 命令 | ✅ 完成 | cli.py `ingest-code` 命令 |
| **P5** | code:: 接入 FederatedGraph | ✅ 完成 | store/federated.py |
| **P5** | CrossNamespaceProbe class | ✅ 完成 | mcp_server/server.py (hook-based) |
| **P5** | list_cross_namespace_edges MCP tool | ✅ 完成 | Tool 8, mcp_server/server.py L666 |
| **§5.1** MCP scope 参数 | search_knowledge / trace_path / impact 等加 scope + min_confidence | ✅ 完成 | trace_path 补全于 2026-05-04 |
| **P6** Impact | impact_bfs() + path_score_fn | ✅ 完成 | retrieve/impact.py |
| **P6** | impact MCP tool | ✅ 完成 | mcp_server/server.py |
| **P6** | 跨 namespace impact | ✅ 完成 | code→vault（已有）+ vault→code 对称增强（2026-05-04） |
| **P7** ConvParser | ConvParser | ❌ 预期未做 | iron law validator 已就位 |

### 当前状态（2026-05-04 更新）

**v5.0 全部可编码任务已完成（含 §5.1 trace_path、P4 BFS 默认 tier、P6 跨图 impact 三项收尾）。** 320 tests pass。

> ✅ 本轮完成（2026-05-04）：trace_path 加 scope/min_confidence 参数、bfs_traverse/dfs_traverse 默认排除 AMBIGUOUS、impact 工具补全 vault→code 方向的 mentions_symbol 跨图增强（含 code 节点 signature 元数据）。
>
> ✅ 上一轮完成（2026-05-01）：tag_co 边、search_knowledge 接入 hybrid_search、list_cross_namespace_edges、EmbeddingStore dim 安全、hybrid_search stale-ID 过滤、qwen3-embedding:8b、ingest-code on ZenithLoom（1886 embeddings, 4096-dim）

---

## 一、设计目标与约束

### 1.1 目标

1. **Parser 插件化**：把数据解析（Markdown/代码/对话）与图算法完全解耦
2. **Hybrid Search**：BM25 + Embedding + RRF 三路融合，替换现有单一 BFS 起点选择
3. **Edge Confidence**：所有边带置信度，支持过滤和加权遍历
4. **CodeParser**：接入代码仓库（ZenithLoom 优先），产出 `code::` namespace
5. **Impact 分析**：定向、带置信度、分深度的影响面分析工具
6. **ConvParser**：从 agent 对话提取候选事实，写入 vault inbox

### 1.2 硬性约束

- **向后兼容**：现有 17 个 MCP 工具的调用方式不变，只增不改
- **零外部系统依赖**：不引入 GitNexus、mem0 作为运行时依赖
- **渐进迁移**：每个 Phase 独立可部署，不需要等全部完成才能用
- **测试覆盖**：每个新组件必须有对应的 pytest 测试，通过率 100%
- **现有 174 个测试不回归**

### 1.3 技术选型

| 组件 | 选型 | 原因 |
|---|---|---|
| BM25 | `rank_bm25`（BM25Okapi） | 轻量，纯 Python，无需独立进程 |
| 代码解析 | `tree-sitter` + `tree-sitter-python` | GitNexus 同款，成熟稳定 |
| 图数据库 | 继续用 NetworkX | 够用，零依赖；KuzuDB 留 Phase 8 评估 |
| Embedding（查询时）| bge-m3（Ollama） | 本地，无 API 限速，索引/查询一致 |
| 向量存储 | 继续用 LanceDB | 已有，不变 |

---

## 二、全局数据流

```
数据来源
  ├── NimbusVault/*.md  ──MarkdownParser──→┐
  ├── ZenithLoom/*.py   ──CodeParser──────→├──→ list[Node] + list[Edge]
  └── 对话记录           ──ConvParser──────→┘
                                            │
                                     ingest pipeline
                                            │
                              ┌─────────────▼─────────────┐
                              │        KnowledgeGraph      │
                              │  NetworkX DiGraph          │
                              │  + BM25 Index（新）        │
                              │  + LanceDB embeddings      │
                              │  + community_id（Leiden）  │
                              └─────────────┬─────────────┘
                                            │
                              ┌─────────────▼─────────────┐
                              │       FederatedGraph       │
                              │  namespace: nimbus::       │
                              │  namespace: code::         │
                              │  namespace: conv::         │
                              │  + bridge edges            │
                              └─────────────┬─────────────┘
                                            │
                              ┌─────────────▼─────────────┐
                              │     Hybrid Search 层       │
                              │  BM25 排名 A               │
                              │  Embedding 排名 B          │
                              │  RRF 融合 → top-K 起点    │
                              │  BFS/DFS 展开 → 子图      │
                              └─────────────┬─────────────┘
                                            │
                                    MCP 工具接口
                              （search/explain/trace/impact）
                                            │
                                  Hani / Jei / Asa
```

---

## 三、核心数据结构升级

### 3.1 Node（变化最小）

```python
@dataclass
class Node:
    id: str                    # 不变
    label: str                 # 不变
    kind: str                  # 扩展：新增 "function"|"class"|"fact"
    source_file: str           # 不变
    content: str               # 不变
    content_hash: str          # 不变
    tokens: int                # 不变
    frontmatter: dict          # 不变
    community_id: str          # 不变
    maturity: str              # 不变
    confidence: float          # 【新增】节点可信度，默认 1.0
    namespace: str             # 【新增】"nimbus"|"code"|"conv"，默认 "nimbus"
    ontology_type: str         # 不变
    actionability: str         # 不变
```

向后兼容：`confidence` 默认 1.0，`namespace` 默认 "nimbus"，
现有序列化/反序列化加 `get(key, default)` 处理。

### 3.2 Edge（最重要的变化）

```python
@dataclass
class Edge:
    id: str
    source: str
    target: str
    kind: str               # 现有种类 + 新增 "calls"|"imports"|"inherits"|"implements"
    weight: float           # 不变
    confidence: float       # 【新增】连续置信度 [0.0, 1.0]
    confidence_tier: str    # 【新增】PRIMARY 过滤维度："EXTRACTED"|"INFERRED"|"AMBIGUOUS"
    evidence: list[str]     # 【新增】置信度来源说明（借鉴 GitNexus ResolutionEvidence）

# 置信度两轴约定：
#
# kind                       tier        confidence
# ─────────────────────────────────────────────────────────────────
# wikilink（显式声明）        EXTRACTED   1.0
# tag_ref（文件→标签）        EXTRACTED   0.9
# Tree-sitter calls/imports  EXTRACTED   1.0（结构事实）
# tag 共现（tag_co）          INFERRED    0.7
# Tree-sitter 推断类型        INFERRED    0.80（constructor 推断）
# 相似度边                   INFERRED    cosine_similarity 值（0.4–0.95）
# LLM 推断的 depends_on      INFERRED    0.65–0.90
# embedding bridge           INFERRED    cosine_similarity 值
# LLM 对话推断事实           INFERRED    embedding 相似度（≥0.92 去重）
# 存疑/模糊边                AMBIGUOUS   0.1–0.3（标记，不默认遍历）
#
# ⚠️ 铁律：认识论来源不可由频率覆写。
#    ConvParser 产出的边最高只能是 INFERRED，无论被重复提取多少次。
#    EXTRACTED 的唯一合法来源是确定性解析器（AST / Tree-sitter / wikilink）。
```

### 3.3 Parser 抽象接口

```python
# prism_rag/ingest/base_parser.py

from abc import ABC, abstractmethod
from pathlib import Path

class Parser(ABC):
    """
    数据源解析器基类。
    子类负责把原始数据转成 ParseResult（Pydantic 验证后输出），
    不关心后续的 embedding / Leiden / 持久化。
    """

    @abstractmethod
    def parse(self, source: Path) -> "ParseResult":
        """
        解析数据源，返回经 Pydantic 校验的 ParseResult。
        source 可以是：
          - 单个文件路径（.md / .py）
          - 目录路径（整个 vault / 整个 repo）
        Pydantic validator 在 Parser 出口处即拦截自相矛盾数据
        （如 tier=EXTRACTED 但 confidence=0.3），不留到 writer 层才爆。
        """
        ...

    @property
    @abstractmethod
    def namespace(self) -> str:
        """返回该 parser 产出节点的 namespace（nimbus / code / conv）。"""
        ...
```

### 3.4 ParseResult — Parser 输出契约（Pydantic）

> LLM 驱动的非确定性 parser（ConvParser）必须在出口处做运行时校验。
> CodeParser 路径在 Tree-sitter 输出格式稳定后可 bypass，但合约定义不变。

```python
# prism_rag/ingest/parse_result.py

from __future__ import annotations
from datetime import datetime
from typing import Any, Literal, Self
from pydantic import BaseModel, Field, model_validator

ConfidenceTier = Literal["EXTRACTED", "INFERRED", "AMBIGUOUS"]

# tier ↔ confidence 合法区间
_TIER_RANGES = {
    "EXTRACTED":  (0.85, 1.0),
    "INFERRED":   (0.30, 0.94),
    "AMBIGUOUS":  (0.0,  0.30),
}

class NodeRecord(BaseModel):
    id: str
    namespace: Literal["nimbus", "code", "conv"]
    kind: str                             # note/knowledge/function/class/module/fact/...
    label: str
    content: str
    source_file: str
    confidence_tier: ConfidenceTier = "EXTRACTED"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: dict[str, Any] = {}

    @model_validator(mode="after")
    def tier_float_consistency(self) -> Self:
        lo, hi = _TIER_RANGES[self.confidence_tier]
        if not (lo <= self.confidence <= hi):
            raise ValueError(
                f"Node {self.id}: confidence_tier={self.confidence_tier} "
                f"但 confidence={self.confidence} 不在合法区间 [{lo}, {hi}]"
            )
        return self

class EdgeRecord(BaseModel):
    source_id: str
    target_id: str
    kind: str
    confidence_tier: ConfidenceTier = "EXTRACTED"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    weight: float = 1.0
    evidence: list[str] = []              # 借鉴 GitNexus ResolutionEvidence

    @model_validator(mode="after")
    def tier_float_consistency(self) -> Self:
        lo, hi = _TIER_RANGES[self.confidence_tier]
        if not (lo <= self.confidence <= hi):
            raise ValueError(
                f"Edge {self.source_id}→{self.target_id}: confidence_tier={self.confidence_tier} "
                f"但 confidence={self.confidence} 不在合法区间 [{lo}, {hi}]"
            )
        # ⚠️ 铁律：conv:: 来源的边永远不能是 EXTRACTED
        if self.confidence_tier == "EXTRACTED":
            for ev in self.evidence:
                if "conv::" in ev or "ConvParser" in ev:
                    raise ValueError(
                        "ConvParser 产出的边不能是 EXTRACTED。"
                        "认识论来源不可由频率或任何其他信号覆写。"
                    )
        return self

class ParseResult(BaseModel):
    nodes: list[NodeRecord]
    edges: list[EdgeRecord]
    parser_id: str          # "MarkdownParser"|"CodeParser"|"ConvParser"
    namespace: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### 3.4 BM25 Index

```python
# prism_rag/store/bm25_index.py

from rank_bm25 import BM25Okapi
import jieba  # 中文分词

class BM25Index:
    def __init__(self):
        self._index: BM25Okapi | None = None
        self._node_ids: list[str] = []

    def build(self, graph: KnowledgeGraph) -> None:
        """从图的所有节点 content 构建 BM25 索引。"""
        corpus = []
        self._node_ids = []
        for node_id, data in graph.g.nodes(data=True):
            content = data.get("content", "") + " " + data.get("label", "")
            tokens = self._tokenize(content)
            corpus.append(tokens)
            self._node_ids.append(node_id)
        self._index = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """返回 [(node_id, score), ...] 按分数降序。"""
        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)
        ranked = sorted(zip(self._node_ids, scores),
                        key=lambda x: x[1], reverse=True)
        return [(nid, score) for nid, score in ranked[:top_k] if score > 0]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # 中英混合分词：jieba 处理中文，空格分词处理英文
        return list(jieba.cut_for_search(text))
```

### 3.5 Hybrid Search（RRF 融合）

```python
# prism_rag/retrieve/hybrid.py

from collections import defaultdict

def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[str]:
    """
    RRF 融合多个排名列表。
    score(d) = Σ 1 / (k + rank_i(d))
    k=60 是标准常数，防止头部名次过度权重。
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, node_id in enumerate(ranking, start=1):
            scores[node_id] += 1.0 / (k + rank)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def hybrid_search(
    query: str,
    graph: KnowledgeGraph,
    bm25_index: BM25Index,
    embed_fn,           # Callable[[str], list[float]]
    embedding_store,    # EmbeddingStore
    top_k: int = 10,
    namespace: str = "",
) -> list[str]:
    """
    三路检索 → RRF 融合 → 返回 top-K node_id 列表作为 BFS 起点。

    embed_fn: 把 query 字符串转成向量（用 bge-m3 或 Gemini）
    """
    # 路 1：BM25
    bm25_results = [nid for nid, _ in bm25_index.search(query, top_k=top_k*3)]

    # 路 2：Embedding 相似度
    query_vec = embed_fn(query)
    emb_results = embedding_store.nearest(query_vec, top_k=top_k*3)

    # 路 3（可选）：精确 ID 匹配
    exact = [nid for nid in graph.g.nodes if query.lower() in nid.lower()]

    # RRF 融合
    fused = reciprocal_rank_fusion([bm25_results, emb_results, exact])

    # namespace 过滤
    if namespace:
        fused = [nid for nid in fused
                 if graph.g.nodes[nid].get("namespace", "nimbus") == namespace]

    return fused[:top_k]
```

---

## 四、各 Parser 详细设计

### 4.1 MarkdownParser（重构现有逻辑）

```
输入：vault 目录（Path）
输出：list[Node] + list[Edge]

节点种类（kind）：
  note      — 普通 markdown 文件
  tag       — #tag（从文件里提取的标签节点）
  category  — 目录节点

边种类（kind）：
  wikilink  — [[文件名]] 显式引用，confidence=1.0
  tag_ref   — 文件 → #tag，confidence=0.9
  tag_co    — 两个共用同一 tag，confidence=0.7
  in_dir    — 文件在目录里，confidence=1.0
```

实现：把现有 `ast_extractor.py` + `vault_loader.py` 的逻辑包装进 `MarkdownParser(Parser)`，
外部接口不变，内部实现不动。**这一步是纯重构，行为完全不变。**

### 4.2 CodeParser（新建）

```
输入：代码仓库目录（Path）
输出：list[Node] + list[Edge]

节点种类（kind）：
  function  — 函数/方法
  class     — 类
  module    — 模块文件（.py）
  variable  — 模块级变量/常量（可选）

边种类（kind）：
  calls     — 函数 A 调用函数 B，confidence=1.0（结构事实）
  imports   — 模块 A 导入模块 B，confidence=1.0
  inherits  — 类 A 继承类 B，confidence=1.0
  defined_in — 函数/类 定义在 模块里，confidence=1.0
  inferred_call — 通过 constructor 推断的类型调用，confidence=0.80
```

**Tree-sitter 解析流程：**

```python
# prism_rag/ingest/code_parser.py

import tree_sitter_python as tspython
from tree_sitter import Language, Parser as TSParser

PY_LANGUAGE = Language(tspython.language())

class CodeParser(Parser):
    namespace = "code"

    def parse(self, source: Path) -> tuple[list[Node], list[Edge]]:
        nodes, edges = [], []
        py_files = list(source.rglob("*.py"))
        for fpath in py_files:
            if any(p in fpath.parts for p in ["__pycache__", ".venv", "node_modules"]):
                continue
            file_nodes, file_edges = self._parse_file(fpath, source)
            nodes.extend(file_nodes)
            edges.extend(file_edges)
        return nodes, edges

    def _parse_file(self, fpath: Path, repo_root: Path):
        """解析单个 .py 文件，提取函数/类/import。"""
        source_code = fpath.read_bytes()
        parser = TSParser(PY_LANGUAGE)
        tree = parser.parse(source_code)
        rel_path = str(fpath.relative_to(repo_root))

        nodes, edges = [], []
        module_id = f"code::{rel_path}"

        # 模块节点
        nodes.append(Node(
            id=module_id,
            label=fpath.stem,
            kind="module",
            source_file=rel_path,
            content=source_code.decode(errors="replace")[:5000],
            namespace="code",
            confidence=1.0,
        ))

        # 遍历 AST，提取 class/function/import
        for child in tree.root_node.children:
            if child.type == "class_definition":
                cls_node, cls_edges = self._extract_class(child, rel_path, module_id)
                nodes.append(cls_node)
                edges.extend(cls_edges)
            elif child.type == "function_definition":
                fn_node, fn_edges = self._extract_function(child, rel_path, module_id)
                nodes.append(fn_node)
                edges.extend(fn_edges)
            elif child.type in ("import_statement", "import_from_statement"):
                import_edges = self._extract_imports(child, module_id)
                edges.extend(import_edges)

        return nodes, edges
```

**Node ID 命名约定（code:: namespace）：**

```
模块：   code::framework/nodes/llm/claude.py
类：     code::framework/nodes/llm/claude.py::ClaudeSDKNode
方法：   code::framework/nodes/llm/claude.py::ClaudeSDKNode::__call__
函数：   code::framework/nodes/llm/claude.py::compute_embeddings
```

**内容字段：** 存函数的完整源码（含 docstring），作为 embedding 和 BM25 的输入。

### 4.3 ConvParser（长期，Phase 7）

```
输入：对话历史（list[Message]）
输出：list[Node]（候选 fact 节点，namespace="conv"）

流程：
  1. LLM 单次调用：从对话提取候选事实（JSON 数组）
  2. 每个候选 fact 与已有 KNOW notes 做 embedding 相似度比较
  3. 相似度 ≥ 0.92 → 判为同一概念 → NONE（跳过）
  4. 相似度 0.75–0.91 → LLM 判断 ADD / UPDATE / supersede
  5. 相似度 < 0.75 → ADD（新概念）
  6. 输出写入 NimbusVault/未分类/inbox/<timestamp>-<slug>.md
```

LLM 提取 prompt（仿 mem0 FACT_RETRIEVAL_PROMPT）：
```
从以下对话中提取独立的、可验证的事实性主张。
每条主张须满足：
  - 能独立成句，无需上下文
  - 是关于系统设计、架构决策或技术事实的陈述
  - 不是过渡句或推理中间步骤

输出 JSON 数组：
[{"claim": "...", "confidence": 0.0-1.0, "type": "fact|decision|concept"}]
```

---

## 五、MCP 工具升级

### 5.1 现有工具加 scope 参数（向后兼容）

```python
# 所有工具加同一个可选参数
@mcp.tool()
def search_knowledge(
    query: str,
    scope: str = "",      # "" = 全局；"nimbus::" / "code::" / "conv::"
    mode: str = "bfs",
    budget: int = 4000,
    min_confidence: float = 0.0,   # 【新增】边置信度过滤
) -> str: ...

@mcp.tool()
def trace_path(
    source: str,
    target: str,
    scope: str = "",
    min_confidence: float = 0.7,
) -> str: ...
```

### 5.2 新增 impact 工具

```python
@mcp.tool()
def impact(
    target: str,                             # 节点 ID（支持 namespace:: 前缀）
    direction: str = "downstream",           # "downstream" | "upstream" | "both"
    max_depth: int = 3,
    min_confidence: float = 0.7,
    allowed_tiers: list[str] = ["EXTRACTED", "INFERRED"],  # tier 硬过滤（PRIMARY）
    allowed_edge_kinds: list[str] | None = None,           # None=全部；["calls","imports"]=只走代码边
    path_score_fn: str = "weakest_link",     # "weakest_link" | "cumulative_decay"
    scope: str = "",
) -> str:
    """
    分析修改 target 节点的影响范围。

    downstream: 谁依赖 target（target 改了，谁会受影响）
    upstream:   target 依赖谁（target 改了，谁被它拉动）

    path_score_fn:
      weakest_link    — 路径分 = min(edge.confidence)，适合 Impact 分析（瓶颈决定整体）
      cumulative_decay — 路径分 = Π tier_decay[edge.tier]，适合溯源分析（每跳增加不确定性）

    tier_decay 默认值 {EXTRACTED:1.0, INFERRED:0.6, AMBIGUOUS:0.2}（初始估计，待实验校准）

    返回按深度分层的影响节点列表：
    Depth 1: DIRECTLY AFFECTED
    Depth 2: LIKELY AFFECTED
    Depth 3+: MAY BE AFFECTED
    """
```

**impact 实现（定向 BFS + tier 硬过滤 + 双维度过滤）：**

```python
_TIER_DECAY_DEFAULT = {"EXTRACTED": 1.0, "INFERRED": 0.6, "AMBIGUOUS": 0.2}

def _impact_bfs(graph, start_id, direction, max_depth,
                min_confidence, allowed_tiers, allowed_edge_kinds,
                path_score_fn, tier_decay=None):
    if tier_decay is None:
        tier_decay = _TIER_DECAY_DEFAULT
    allowed_tiers_set = set(allowed_tiers)

    results = {}        # depth → list[(node_id, path_score)]
    visited = {start_id}
    queue = deque([(start_id, 0, 1.0)])   # (node, depth, path_score)

    while queue:
        current, depth, path_score = queue.popleft()
        if depth >= max_depth:
            continue

        neighbors = (
            list(graph.successors(current))   if direction in ("downstream", "both") else []
        ) + (
            list(graph.predecessors(current)) if direction in ("upstream", "both") else []
        )

        for neighbor in neighbors:
            edge_data = graph.get_edge_data(current, neighbor) or {}
            tier = edge_data.get("confidence_tier", "INFERRED")
            conf = edge_data.get("confidence", 1.0)
            kind = edge_data.get("kind", "")

            # 双维度过滤：tier 硬过滤（PRIMARY）+ confidence 阈值
            if tier not in allowed_tiers_set:
                continue
            if conf < min_confidence:
                continue
            if allowed_edge_kinds is not None and kind not in allowed_edge_kinds:
                continue

            # 路径评分策略
            decay = tier_decay.get(tier, 1.0)
            if path_score_fn == "weakest_link":
                new_score = min(path_score, conf)
            else:  # cumulative_decay
                new_score = path_score * decay

            if neighbor not in visited:
                visited.add(neighbor)
                results.setdefault(depth + 1, []).append((neighbor, new_score))
                queue.append((neighbor, depth + 1, new_score))

    return results
```

---

## 六、Task Breakdown（按 Phase）

### Phase 1 — Embedding 补全（立即可做，约 2 小时）

- [x] **T1.1** 安装 embedding 模型：`ollama pull qwen3-embedding:8b`（dim=4096，#1 MTEB 2025）
- [x] **T1.2** 在 `embedder.py` 新增 `OllamaEmbedder` 类，实现同 `compute_embeddings` 接口
  - 调用 `POST http://localhost:11434/api/embed`
  - 输入：`{"model": "bge-m3", "input": text}`
  - 输出：`{"embeddings": [[...]]}`
  - 维度：1024（需同步更新 LanceDB schema，从 768 → 1024 或新建表）
- [x] **T1.3** 在 `config.py` 新增 `PRISM_EMBED_BACKEND=ollama|gemini`，默认 `ollama`
- [x] **T1.4** CLI `ingest` 命令：根据 backend 选择 embedder
- [ ] **T1.5** 运行全量 ingest，验证 103 个节点全部有 embedding
  - 验收：`SELECT count(*) FROM embeddings` = 103
- [x] **T1.6** 更新 `EmbeddingStore`：如果表已存在但维度不匹配，drop + recreate
  - `EmbeddingStore(path, dim=768)` — dim 可配置，默认 768
  - `PrismRagSettings.embedding_dim` 属性：ollama→1024，gemini→embed_dimensionality
  - `persist_embeddings`、`FederatedGraph.load`、`server.py` 均已透传 dim
- [x] **T1.7** 测试：`test_embedder_ollama.py`，mock Ollama HTTP，验证格式

**依赖**：无。**产出**：完整 embedding，解锁 Hybrid Search。

---

### Phase 2 — Hybrid Search（约 1–2 天）

- [x] **T2.1** 安装依赖：`pip install rank-bm25 jieba`（写入 `pyproject.toml`）
- [x] **T2.2** 实现 `BM25Index`（`prism_rag/store/bm25_index.py`）
  - `build(graph)` — 从图节点构建索引
  - `search(query, top_k)` — 返回 `[(node_id, score)]`
  - 中英混合分词：jieba `cut_for_search` + 空格分词组合
- [x] **T2.3** `KnowledgeGraph` 类新增 `bm25: BM25Index`，在图加载完成后 `bm25.build(self)`
- [x] **T2.4** 实现 `hybrid_search()`（`prism_rag/retrieve/hybrid.py`）
  - BM25 排名 A + Embedding 排名 B + 精确 ID 匹配 C
  - RRF 融合 → top-K node_id
  - `namespace` 过滤参数
- [x] **T2.5** 实现查询时 embedding：
  - `OllamaEmbedder.embed_query(text) -> list[float]`（单条，比 ingest 简单）
  - Gemini backend 同样实现
- [x] **T2.6** 升级 `EmbeddingStore`：新增 `nearest(vec, top_k) -> list[str]`（向量近邻搜索）
  - LanceDB：`table.search(vec).limit(top_k).to_arrow()`
- [x] **T2.7** 升级 `search_knowledge` MCP 工具：
  - 新增 `scope` 参数（默认 `""`）
  - 新增 `min_confidence` 参数（默认 `0.0`）
  - 起点改用 `hybrid_search()` 返回，BFS 展开逻辑不变
  - BM25/embedding 在 `_ensure_federated()` 启动时构建，graceful fallback
- [x] **T2.8** 测试：`test_hybrid_search.py`
  - 验证 RRF 正确融合两个排名列表
  - 验证 namespace 过滤
  - 验证中文查询能命中

**依赖**：T1.5（需要 embedding 才能做 embedding 路检索）。**产出**：更准的搜索起点。

---

### Phase 3 — Parser 抽象（约 1.5 天，纯重构 + Pydantic 契约）

- [x] **T3.1** 新建 `prism_rag/ingest/base_parser.py`，定义 `Parser` ABC（返回 `ParseResult`）
- [x] **T3.2** 新建 `prism_rag/ingest/parse_result.py`，实现 `NodeRecord / EdgeRecord / ParseResult`
  - Pydantic v2 模型，含 `tier_float_consistency` validator
  - `_TIER_RANGES`：EXTRACTED [0.85,1.0]，INFERRED [0.30,0.94]，AMBIGUOUS [0.0,0.30]
  - `EdgeRecord` validator 内嵌铁律检查：conv:: 来源禁止 EXTRACTED
- [x] **T3.3** 新建 `prism_rag/ingest/markdown_parser.py`
  - `class MarkdownParser(Parser): namespace = "nimbus"`
  - 把 `ast_extractor.py` 的 `extract_ast()` 搬进 `parse()`，输出 `ParseResult`
  - 把 `media_extractor.py` 的 `add_media_nodes()` 合并进来
- [x] **T3.4** 更新 `Node` dataclass：新增 `namespace: str = "nimbus"` 字段
- [x] **T3.5** 更新 `Edge` dataclass：新增 `confidence: float`, `confidence_tier: str`, `evidence: list[str]` 字段
- [x] **T3.6** 更新 `graph.json` 序列化/反序列化：
  - `Node.to_dict()` / `Edge.to_dict()` 输出新字段
  - `from_dict()` 用 `d.get(key, default)` 兼容旧数据（zero breaking change）
- [x] **T3.7** 更新 `cli.py` 的 `ingest` 命令：使用 `MarkdownParser`，行为完全不变
- [x] **T3.8** 运行现有 174 个测试，全部通过（零回归）
- [x] **T3.9** 新增测试 `test_parse_result.py`：
  - 验证 tier=EXTRACTED、confidence=0.3 时 validator 抛出 ValueError
  - 验证 evidence 含 "conv::" 的边不能是 EXTRACTED
  - 验证合法数据正常通过

**依赖**：无（可与 Phase 1/2 并行）。**产出**：Parser 插件槽位 + 输出契约就绪。

---

### Phase 4 — Edge Confidence + Tier（约 0.5 天）

- [x] **T4.1** 在 `ast_extractor.py`（或 `MarkdownParser`）里，按边类型同时赋 confidence + tier：
  - wikilink → `confidence=1.0, tier=EXTRACTED`（已完成）
  - tag_ref → `confidence=0.9, tier=EXTRACTED`（已完成）
  - tag_co（共现）→ `confidence=0.7, tier=INFERRED`（✅ T4.1b 完成，step 5 in extract_ast）
- [x] **T4.2** 在 `similarity_linker.py`：相似度边的 `confidence=cosine_similarity, tier=INFERRED`
- [x] **T4.3** 更新 `bfs.py` 和 `dfs.py`：接受双维度过滤参数：
  - `min_confidence: float = 0.0`
  - `allowed_tiers: set[str] | None = None`（None = 全部通过）
  - AMBIGUOUS 边默认不参与遍历，除非调用方显式 `allowed_tiers={"AMBIGUOUS"}`
- [x] **T4.4** 更新 `search_knowledge` 把过滤参数传递给 BFS
  - `federated_bfs` / `federated_dfs` 默认 `allowed_tiers={"EXTRACTED","INFERRED"}`，AMBIGUOUS 默认排除
  - `min_confidence` 从 `search_knowledge` 透传到遍历层
- [x] **T4.5** 验证：`test_edge_confidence.py`
  - 验证 `min_confidence=0.8` 时 tag_co 边不被遍历
  - 验证 `allowed_tiers={"EXTRACTED"}` 时 INFERRED 相似度边不被遍历
  - 验证 AMBIGUOUS 边默认不出现在 BFS 结果里

**依赖**：T3.5（Edge dataclass 有 confidence_tier 字段）。

---

### Phase 5 — CodeParser（约 3–5 天）

- [x] **T5.1** 安装依赖：`pip install tree-sitter tree-sitter-python`（写入 `pyproject.toml`）
- [x] **T5.2** 新建 `prism_rag/ingest/code_parser.py`，实现 `CodeParser(Parser)`
  - `namespace = "code"`
  - `parse(repo_path: Path)` → 递归处理所有 `.py` 文件
  - 跳过：`__pycache__`, `.venv`, `node_modules`, `*.pyc`
- [x] **T5.3** 实现 `_parse_file(fpath)` — 解析单个 Python 文件：
  - 提取 `class_definition` 节点
  - 提取 `function_definition` 节点
  - 提取 `import_statement` / `import_from_statement` 边
  - Node content = 函数/类的完整源码（含 docstring），截断到 5000 chars
  - 含相对 import 解析（`_resolve_relative_import()`）
- [x] **T5.4** 实现 `_extract_class(node, ...)` — 提取类节点 + `inherits` 边 + 方法节点
  - 含 MRO 继承链查找（`_mro_lookup()`）
- [x] **T5.5** 实现 `_extract_function(node, ...)` — 提取函数节点 + `defined_in` 边
- [x] **T5.6** 实现 `_extract_calls(node, ...)` — 在函数体内找 `call_expression`
  - 直接调用（`func_name(...)`）→ `confidence=1.0`
  - 方法调用（`self.func(...)`）→ 尝试类型推断，`confidence=0.8`
  - 含执行流检测（`_detect_flows()`），产出 `kind="flow"` 节点 + `step_of` 边
- [x] **T5.7** CLI 新增 `ingest-code` 命令：
  ```
  prism-rag ingest-code \
    --repo /home/kingy/Foundation/ZenithLoom \
    --data-dir /home/kingy/Foundation/PrismRag/data/code \
    --namespace code
  ```
- [x] **T5.8** FederatedGraph：加载 `code::` namespace 的 graph.json
  - `serve` 命令通过 `PRISM_GRAPHS` env var 指定多个 namespace
- [x] **T5.9** Bridge edges（embedding 跨 namespace）：
  - 在 `FederatedGraph.build_bridges()` 里，对 `nimbus::` 和 `code::` 做跨 namespace 余弦相似度
  - `similarity ≥ 0.85` → 添加 `implements` 或 `related_to` 边，`confidence=similarity, tier=INFERRED`
  - bridge 阈值 0.85（比 within-graph 0.75 更严，减少误连）
- [x] **T5.10** 测试：`test_code_parser.py`（46 个测试，含 relative imports / MRO / execution flows）
  - 用 ZenithLoom 的一个小模块（如 `framework/debug.py`）做 fixture
  - 验证函数节点被正确提取
  - 验证 import 边被正确提取
  - 验证 namespace 为 "code"
- [x] **T5.11** 新建 `prism_rag/store/cross_namespace_probe.py`：`CrossNamespaceProbe` 类
  ```python
  class CrossNamespaceProbe:
      def on_edge_created(self, edge: EdgeRecord) -> None:
          """边创建钩子：跨 namespace 边写入 cross_namespace_log"""

      def list_cross_edges(self,
                           min_confidence: float = 0.0,
                           allowed_tiers: list[str] | None = None
                          ) -> list[CrossEdgeEntry]: ...

      def list_new_cross_edges(self, since: datetime) -> list[CrossEdgeEntry]:
          """增量查询：本月新发现了哪些跨域连接？"""

      def list_edges_from_node(self, node_id: str) -> list[CrossEdgeEntry]:
          """按节点查询：这个代码模块和哪些文档有关联？"""
  ```
  每条 `CrossEdgeEntry` 包含：`edge_id, source_node, target_node, edge_kind,
  confidence_tier, confidence, first_seen_at, evidence[]`
- [x] **T5.12** 在 `FederatedGraph.build_bridges()` 里注入 `CrossNamespaceProbe.on_edge_created()`
- [x] **T5.13** 注册 MCP 工具 `list_cross_namespace_edges(since?, scope?)`

**依赖**：T3.1（Parser ABC），T3.4（Node 有 namespace 字段）。

**v5.1 激活条件**（届时实现社区检测 + 惊喜边评分）：
nimbus 节点 > 200 **且** cross_namespace_log > 50 条 **且** 至少 10 条经人工确认有价值。

---

### Phase 6 — Impact 分析（约 1.5 天）

- [x] **T6.1** 实现 `impact_bfs()` 函数（`prism_rag/retrieve/impact.py`）
  - 完整参数：`target_id, graph, direction, max_depth, min_confidence,
    allowed_tiers, allowed_edge_kinds, path_score_fn, tier_decay`
  - 返回：`dict[int, list[tuple[str, float]]]`（depth → [(node_id, path_score)]）
  - 支持两种 `path_score_fn`：`weakest_link`（默认）和 `cumulative_decay`
- [x] **T6.2** 注册 `impact` MCP 工具（`prism_rag/mcp_server/server.py`）
  - 完整 API 签名（见 5.2 节）
  - 深度标签：Depth 1 = "DIRECTLY AFFECTED"，Depth 2 = "LIKELY AFFECTED"，Depth 3+ = "MAY BE AFFECTED"
  - 输出附带每个节点的 `path_score` 和 `confidence_tier`
- [ ] **T6.3** 跨 namespace 支持：`scope=""` 时在 FederatedGraph 的 `unified_view` 上跑
  - ⚠️ 当前：单图内完整；FederatedGraph 跨图路径未支持
- [x] **T6.4** 测试：`test_impact.py`
  - 构造 3 层依赖的测试图（含 EXTRACTED、INFERRED、AMBIGUOUS 边）
  - 验证 downstream / upstream 方向
  - 验证 `allowed_tiers={"EXTRACTED"}` 时 INFERRED 边不被遍历
  - 验证 `allowed_edge_kinds=["calls"]` 时非代码边被过滤
  - 验证 `path_score_fn="weakest_link"` vs `"cumulative_decay"` 结果差异
  - 验证 `tier_decay` 可覆盖默认值

**依赖**：T4.3（BFS 支持 tier + confidence 双维度过滤）。

---

### Phase 7 — ConvParser（长期，约 3–5 天）

> **设计基线**：mem0 v1.1 的 additive-only 路线（单次 LLM 提取 + embedding 去重），
> 不采用四操作模式（mem0 自己已弃用，UPDATE/DELETE 判断在弱模型上容易出错）。

**三路径 promote 机制：**

| 路径 | 触发条件 | 入图 tier | 审核方式 |
|------|----------|-----------|----------|
| 规则自动 | fact 可被 CodeParser/正则/schema 约束验证 | EXTRACTED | 无需人工 |
| 批量审核 | count ≥ 2（独立信源）→ 进 Jei 优先队列 | INFERRED | Jei 批量确认 |
| 单条审核 | count == 1 | INFERRED | Jei 逐条审核 |

独立信源定义：(a) `session_user` 不同；或 (b) 同一用户但 `session_id` 间隔 > 24h。

**⚠️ 铁律（不设例外）**：ConvParser 产出的 fact 最高只能是 INFERRED。
EXTRACTED 的唯一合法来源是确定性解析器。频率、重复次数、用户级别均不能覆写此规则。

- [ ] **T7.1** 设计 LLM 提取 prompt（additive-only，中英双语）
  - 单次调用：从对话提取候选 facts → JSON 数组 `[{claim, type, confidence}]`
  - 显式要求：每条 claim 须独立成句、可验证、无需上下文
- [ ] **T7.2** 实现 embedding 去重逻辑：
  - `≥ 0.92`：与已有 KNOW 重复 → NONE，跳过
  - `0.75–0.91`：可能是 UPDATE → inbox 标注 `status: possible_update, linked_id: <KNOW-XXX>`
  - `< 0.75`：新概念 → inbox 标注 `status: new`
- [ ] **T7.3** 实现 `ConvParser(Parser)`（`prism_rag/ingest/conv_parser.py`）
  - 输出 `ParseResult`（tier 强制为 INFERRED，Pydantic validator 自动校验）
- [ ] **T7.4** inbox 输出：`NimbusVault/未分类/inbox/<timestamp>-conv-fact.md`
  - frontmatter 含：`count, sessions[], status, linked_id?, first_seen, last_seen`
- [ ] **T7.5** 30 天 TTL：count==1 且超 30 天未审核 → 自动标注 `status: expired`（不删除，不入图）
- [ ] **T7.6** `promote_batch(fact_ids: list[str])` API：
  - Jei 可一次性确认多条 fact
  - 高频 fact（count ≥ 2）自动附带"证据包"：聚合所有来源会话的相关片段
- [ ] **T7.7** ZenithLoom discord 集成：`on_message` 完成后异步调用 ConvParser（不阻塞响应）
- [ ] **T7.8** Jei skill `curate-inbox`：
  - 显示优先队列（count ≥ 2 的先展示）
  - 支持 `promote_batch()`，单条审核，标注 expired
- [ ] **T7.9** 测试：`test_conv_parser.py`
  - 验证重复 fact（≥ 0.92）不产生新节点
  - 验证 possible_update 正确标注 linked_id
  - 验证 ParseResult 中所有边的 tier 均为 INFERRED（铁律测试）

**依赖**：T1.5（embedding 完整），Phase 3（Parser + ParseResult），Phase 4（Edge tier）。

---

### Phase 8 — KuzuDB 评估（可选）

- [ ] **T8.1** Benchmark：用 ZenithLoom 代码仓库（code:: namespace）对比
  - NetworkX BFS 查询耗时
  - KuzuDB Cypher 同款查询耗时
- [ ] **T8.2** 如果 Cypher 使用需求明确 → 实现 KuzuDB adapter
  - 实现 `KuzuGraph` 类，接口与 `KnowledgeGraph` 相同
  - `ingest-code` 写 KuzuDB 格式
  - MCP 工具通过 Cypher 查询

**依赖**：Phase 5 完成后才有足够数据评估。

---

## 七、依赖关系图

```
T1（embedding）──────────────────────────────────────────────────────→ T2（Hybrid Search）
                                                                                ↓
T3（Parser 抽象 + ParseResult Pydantic）→ T4（Edge Confidence + Tier）→ T5（CodeParser + CrossNamespaceProbe）
                                                                                ↓
                                                                         T6（Impact 分析）
                                                                                ↓
                                                                         T7（ConvParser）→ T8（KuzuDB，可选）
```

T1 和 T3 可以完全并行开始。
T2 需要 T1 的 embedding 完整。
T5 需要 T3 的 Parser ABC + ParseResult 就绪，T4 的 tier 过滤就绪。
T6 需要 T4 的双维度过滤。
T7 需要 T1（embedding）+ T3（ParseResult + 铁律 validator）+ T4（tier）。

---

## 八、验收标准汇总

| Phase | 关键验收条件 |
|---|---|
| 1 | 103 个节点全部有 bge-m3 embedding，维度 1024 |
| 2 | `search_knowledge("context explosion")` 中文查询命中正确节点，BM25+emb 两路均贡献结果 |
| 3 | 174 个现有测试零回归；`ingest` 命令行为不变；tier=EXTRACTED+confidence=0.3 时 validator 抛错 |
| 4 | `allowed_tiers={"EXTRACTED"}` 时 INFERRED tag_co 边不出现在 BFS 结果里 |
| 5 | `ingest-code ZenithLoom` 产出 ≥ 100 个函数/类节点；`ClaudeSDKNode` 节点存在；`list_new_cross_edges()` 返回至少 1 条跨域边 |
| 6 | `impact(target="LlmNode", direction="upstream", allowed_edge_kinds=["inherits"])` 正确返回 ClaudeSDKNode、GeminiCLINode；`path_score_fn` 两种策略结果可复现 |
| 7 | 对话结束后 10 秒内 inbox 出现候选 fact 文件；所有 ParseResult edge.tier 均为 INFERRED（铁律测试通过） |

---

## 九、风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| bge-m3 维度（1024）与现有 LanceDB（768）冲突 | 高 | T1.6 处理：drop + recreate；旧 embedding 数据丢弃 |
| Tree-sitter Python binding 版本不兼容 | 中 | 锁定 `tree-sitter==0.21.*`；先验证再写 CI |
| 中文 BM25 分词质量差 | 中 | jieba + 空格分词组合；对专有名词（KNOW-042）做精确匹配补充 |
| code:: namespace embedding 与 nimbus:: bridge 误连 | 中 | bridge 阈值 0.85+（比 within-graph 0.75 更严）；tier=INFERRED，可用 allowed_tiers 过滤 |
| ConvParser LLM 提取事实噪声大 | 高 | additive-only + embedding 去重；inbox TTL 30 天；铁律：conv 边永不 EXTRACTED |
| Pydantic 在 CodeParser 批量处理时性能开销 | 中 | CodeParser 路径在 Tree-sitter 输出稳定后可 bypass validator；ConvParser 路径强制校验 |
| tier_decay 默认值 {0.6, 0.2} 不准确 | 低 | 标注为"初始估计，待实验校准"；上线后用人工评估结果调参 |
| inbox 积压导致 Jei 审核瓶颈 | 中 | 优先队列（count≥2 先展示）+ promote_batch API + 30 天 TTL 自动归档 |
