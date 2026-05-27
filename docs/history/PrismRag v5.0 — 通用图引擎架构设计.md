---
tags:
  - "#PrismRag"
  - "#GraphRAG"
  - "#architecture"
  - "#roadmap"
status: superseded
created: 2026-04-27
superseded_by: "设计细节/PrismRag v5.0 — 详细设计与任务分解.md"
milestone: v5.0
---

> **[SUPERSEDED]** 本文已被 [[PrismRag v5.0 — 详细设计与任务分解]] 取代。概念层描述仍有参考价值，但具体任务分解和实现状态请看新文档。

# PrismRag v5.0 — 通用图引擎架构设计

> **核心洞察**：图算法是通用的，数据来源是可插拔的。
> BM25 + embedding + RRF、BFS/Leiden/Impact 分析，对代码和文档都适用。
> 区别只在于接入的是 Tree-sitter（代码）还是 Markdown parser（文档）。

---

## 1. 当前架构（v4.0）的局限

PrismRag v4.0 是"Markdown-first"的设计：parser 和算法层耦合在一起，
只能处理 vault 里的 `.md` 文件。想接入代码仓库或对话提取，需要大改。

同期参考系统的算法对比：

| 能力 | PrismRag v4.0 | GitNexus | mem0 |
|---|---|---|---|
| 搜索 | BFS/DFS 只此一种 | BM25 + embedding + RRF | BM25 + embedding + entity |
| 边置信度 | 无 | 有 | 有（相似度阈值） |
| 任意图查询 | 无 | Cypher | 无 |
| Impact 分析 | 无 | 有（方向 + 深度） | 无 |
| 数据来源 | Markdown only | 代码 only | 对话 only |

---

## 2. v5.0 目标架构

```
┌─────────────────────────────────────────────────────┐
│              图算法层（完全领域无关）                  │
│                                                     │
│  搜索：BM25 + Embedding + RRF 融合                  │
│  遍历：BFS / DFS / Cypher（KuzuDB）                 │
│  聚类：Leiden 社区检测                               │
│  Impact：定向遍历 + 置信度过滤 + 深度分层            │
│  存储：NetworkX（内存）+ LanceDB（向量）             │
└──────────────────────┬──────────────────────────────┘
                       │  统一 Node + Edge 格式
          ┌────────────┼────────────┐
          │            │            │
   ┌──────▼───┐  ┌─────▼────┐  ┌───▼──────┐
   │ Markdown │  │   Code   │  │   Conv   │
   │  Parser  │  │  Parser  │  │  Parser  │
   │          │  │          │  │          │
   │ wikilinks│  │Tree-sitter│  │ LLM 提取 │
   │ tags     │  │ symbols  │  │ 事实     │
   │frontmatter  │ call graph  │ ADD/UPDATE│
   │ 中文语义 │  │ imports  │  │ /DELETE  │
   └──────────┘  └──────────┘  └──────────┘
   namespace:    namespace:    namespace:
   nimbus::      code::        conv::
      │              │              │
      └──────────────┴──────────────┘
              FederatedGraph
                    │
           统一 MCP 工具接口
         Hani / Jei / Asa 调用
```

---

## 3. 统一数据格式

两种 parser 输出相同结构，上层算法无需感知来源：

```python
@dataclass
class Node:
    id: str
    label: str
    kind: str          # "note"|"concept"|"function"|"class"|"fact"
    content: str
    source_file: str
    namespace: str     # "nimbus"|"code"|"conv"
    metadata: dict     # 存放 parser 特有字段
    community_id: str
    confidence: float  # 新增：节点可信度

@dataclass
class Edge:
    source: str
    target: str
    kind: str          # "wikilink"|"calls"|"imports"|"supersedes"|"implements"
    confidence: float  # 新增：边置信度（GitNexus 借鉴）
    weight: float
```

**Markdown parser 输出示例：**
```
Node(id="KNOW-042", kind="concept", namespace="nimbus",
     source_file="设计细节/PrismRag-v4.0-设计文档.md")
Edge(src="KNOW-042", dst="KNOW-038", kind="depends_on", confidence=0.9)
```

**Code parser 输出示例：**
```
Node(id="ClaudeSDKNode", kind="class", namespace="code",
     source_file="framework/nodes/llm/claude.py")
Edge(src="ClaudeSDKNode", dst="LlmNode", kind="inherits", confidence=1.0)
```

---

## 4. Parser 插件接口

```python
from abc import ABC, abstractmethod

class Parser(ABC):
    @abstractmethod
    def parse(self, source: Path) -> tuple[list[Node], list[Edge]]:
        """解析数据源，返回节点和边列表。"""
        ...

class MarkdownParser(Parser):
    """现有 ast_extractor.py 逻辑封装。"""

class CodeParser(Parser):
    """Tree-sitter 解析代码仓库，借鉴 GitNexus 思路。"""

class ConvParser(Parser):
    """LLM 从对话中提取事实，借鉴 mem0 的四操作模式
    (ADD / UPDATE / supersede / NONE)。"""
```

Ingest pipeline 变成完全通用的：
```python
def ingest(source: Path, parser: Parser, graph: KnowledgeGraph):
    nodes, edges = parser.parse(source)
    graph.add(nodes, edges)
    # Pass 3（embedding）、Pass 4（Leiden）、Pass 5（report）完全不变
```

---

## 5. 算法升级

### 5.1 Hybrid Search（借鉴 GitNexus + mem0）

```
查询
 ├── BM25 检索           → 排名列表 A（关键词精确命中）
 ├── Embedding 检索      → 排名列表 B（语义相近）
 └── RRF 融合            → score = Σ 1/(k + rank_i)
                            取 top-N 作为 BFS 起点
```

替换现有 `search_knowledge` 的起点选择逻辑，图遍历部分不变。

### 5.2 Edge Confidence（借鉴 GitNexus）

所有边带置信度，BFS 可以按 `minConfidence` 过滤：
- wikilinks：1.0（显式声明）
- similarity 边：余弦相似度值
- LLM 推断的 `depends_on`：0.6–0.9
- Tree-sitter 解析的 `calls`：1.0

### 5.3 Impact 分析（新增工具）

```
impact(target="KNOW-042", direction="downstream", maxDepth=3,
       minConfidence=0.7,
       allowedTiers=["EXTRACTED", "INFERRED"],   # tier 硬过滤（PRIMARY）
       allowedEdgeKinds=None,                    # None=全部；["calls","imports"]=只走代码边
       pathScoreFn="weakest_link")               # 或 "cumulative_decay"

→ Depth 1: KNOW-038 (path_score=0.9), KNOW-051 (path_score=0.85)   [DIRECTLY AFFECTED]
→ Depth 2: KNOW-067 (path_score=0.72)                               [LIKELY AFFECTED]
```

对知识库的用途：改了 KNOW-042 之后，哪些 KNOW notes 可能需要同步更新？

`pathScoreFn` 两种策略：
- `weakest_link`（默认）：路径分 = min(edge.confidence)，适合 Impact 分析（瓶颈决定整体）
- `cumulative_decay`：路径分 = Π tier_decay[edge.tier]，适合溯源分析（每跳增加不确定性）

### 5.4 CrossNamespaceProbe（v5.2）

> 整体推迟到 v5.2，不在 v5.0/v5.1 任务分解范围内。

带时序的跨域探测器，记录所有跨 namespace 边及其首次发现时间：

```python
# 三个查询 API
probe.list_cross_edges(min_confidence=0.0, allowed_tiers=None)    # 全量
probe.list_new_cross_edges(since=datetime(2026,5,1))              # 增量（探测器核心）
probe.list_edges_from_node("code::framework/nodes/llm/claude.py") # 按节点
```

`list_new_cross_edges(since)` 回答："知识图谱最近发现了哪些新的跨域连接？"

v5.2 启动条件：
- nimbus 节点 > 200 **且** cross_namespace_log > 50 条 **且** 至少 10 条经人工确认有价值

### 5.4 考虑 KuzuDB 替代 NetworkX

KuzuDB 是嵌入式图数据库，支持 Cypher，无需独立进程，可替代 NetworkX 作为图存储。
开放此决策：NetworkX 够用且零依赖，KuzuDB 带来 Cypher 但增加复杂度。

---

## 6. MCP 工具接口升级

现有 17 个工具加 `scope` 参数，无需改变工具语义：

```
search_knowledge(query="embedding", scope="code::")    → 代码里的实现
search_knowledge(query="embedding", scope="nimbus::") → 知识库里的设计
search_knowledge(query="embedding")                    → 跨库联合搜索

trace_path(from="KNOW-042", to="ClaudeSDKNode")        → 跨 namespace 路径
impact(target="KNOW-042")                              → 新增工具
```

---

## 7. 与现有系统的关系

| 系统 | 角色 | 接入方式 |
|---|---|---|
| GitNexus | 灵感来源（不直接集成） | 自己实现 CodeParser（Tree-sitter）|
| mem0 | 灵感来源（不直接集成） | 自己实现 ConvParser（四操作模式）|
| NimbusVault | nimbus:: namespace | 现有 MarkdownParser（Pass 1）|
| ZenithLoom 代码仓库 | code:: namespace | 新增 CodeParser |
| Jei / Hani 对话 | conv:: namespace | 新增 ConvParser（长期）|

---

## 8. 迁移路径（v4.0 → v5.0）

不是大爆炸重写，而是渐进迁移：

```
Step 1（近期）  补全 embedding（bge-m3，103 节点）
Step 2（近期）  Hybrid Search 接入 BM25 + RRF
Step 3（中期）  Parser 接口抽象，MarkdownParser 封装现有逻辑
Step 4（中期）  Edge confidence 字段上线
Step 5（中期）  CodeParser + code:: namespace（ZenithLoom 仓库先试点）
Step 6（长期）  Impact 分析工具
Step 7（长期）  ConvParser（Jei/Hani 对话事实提取）
Step 8（长期）  KuzuDB 评估（视 Cypher 需求决定）
```

---

## 9. 决策记录

- **D1: 自己实现 CodeParser，不依赖 GitNexus，不写 GitNexusAdapter** —
  评估过三条路：
  (a) 自己实现 CodeParser（Tree-sitter Python），
  (b) 用 GitNexusAdapter 把 GitNexus 输出转成 ParseResult，
  (c) 直接集成 GitNexus 作为数据源。
  选 (a)，拒绝 (b)(c)。原因：
  - Adapter 是永久负债：GitNexus 升级改 schema → adapter 断；GitNexus 进程不在 → code:: namespace 失效
  - 自己实现是一次性工作：ParseTree + ParseResult + StorageBackend 框架已就绪，
    CodeParser 只剩"从 Tree-sitter AST 提取节点"这一件事，约 2–3 天
  - ZenithLoom 是 Python 为主，Python-first 覆盖 95% 用例，其他语言后续按需加
  - ParseTree 天然产出层级（module → class → function），`contains` 边自动生成；
    GitNexus 输出平铺节点，需要额外重建层级
  **GitNexus 的定位**：对比基准。CodeParser 实现后，用同一个 ZenithLoom 仓库
  跑两边，对比节点数、边数、查询质量，验证实现正确性。GitNexus 已安装（v1.6.3），
  ZenithLoom 已索引（7,003 nodes / 11,890 edges / 202 clusters），随时可用于对比
- **D2: 自己实现 ConvParser，不集成 mem0** — 同理；mem0 是完整记忆系统，
  只需要它的四操作提取模式
- **D3: FederatedGraph 作为统一入口** — 现有设计已支持多 namespace，
  不需要新增联邦层
- **D4: MCP 工具接口向后兼容** — 加 scope 参数但保持默认行为不变，
  现有 Jei/Hani 调用不需要修改
- **D5: Pydantic 作为 Parser 输出契约，非 Protocol 也非完整 IR** —
  LLM 驱动的非确定性 parser（ConvParser）必须运行时校验；Protocol 在运行时等于 dict，
  完整 IR 层过重。Pydantic 模型（NodeRecord/EdgeRecord/ParseResult，~100 行）在 Parser
  出口处拦截自相矛盾数据，writer 层直接消费，不做 schema 转换
- **D6: 认识论来源不可由频率覆写（铁律）** —
  ConvParser 产出的边最高只能是 INFERRED，无论被重复提取多少次。
  EXTRACTED 的唯一合法来源是确定性解析器（AST / Tree-sitter / wikilink）。
  没有这条铁律，tier 系统就是装饰品。此规则不设例外、不设升级通道
- **D7: ConvParser 从 additive-only 启动，不采用四操作模式** —
  mem0 本身已在 v1.1 中弃用四操作（UPDATE/DELETE 在弱模型上容易出错）。
  用 embedding 相似度（≥0.92 去重，0.75-0.91 → inbox 标 possible_update）替代 LLM 判断。
  人工审核 inbox 决定 promote，铁律保证所有 conv 边永远是 INFERRED
