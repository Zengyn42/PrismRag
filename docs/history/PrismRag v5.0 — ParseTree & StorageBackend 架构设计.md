---
title: PrismRag v5.0 — ParseTree & StorageBackend 架构设计
date: 2026-04-28
status: superseded
superseded_by: "设计细节/PrismRag v5.0 — 详细设计与任务分解.md"
tags: [prism-rag, architecture, parse-tree, kuzu, storage-backend]
---

> **[SUPERSEDED]** ParseTree / StorageBackend 抽象已实现并合并到主设计。具体实现见 `prism_rag/ingest/base_tree.py`, `parse_result.py`。任务状态见 [[PrismRag v5.0 — 详细设计与任务分解]]。

# ParseTree & StorageBackend 架构设计

## 核心洞察

代码和文档的结构本质上是同一件事——树：

```
Markdown 文档                    代码文件
─────────────────                ──────────────────
note (文件根节点)                 module (文件根节点)
├── section (H2)                  ├── class
│   ├── block (paragraph)         │   ├── function (method)
│   └── block (code_block)        │   └── function (method)
└── section (H2)                  └── function (top-level)
    └── block
```

父子关系都是 `contains`，遍历逻辑完全复用。

## 为什么 ParseTree 不能直接决定怎么存

ParseTree 如果自己调 `to_graph_elements()` 输出到 NetworkX，将来换 Kuzu 时：

- NetworkX：通用 Node dataclass，`kind` 字段区分类型
- Kuzu：每种 kind 对应独立的 node table，schema 强类型

两者之间有结构差异。如果 ParseTree 耦合了存储，迁移要改两层。

## 分层架构

```
Source file
    ↓  Parser.parse()
ParseTree              ← 中间层，与存储无关
    ↓
StorageBackend.write_tree(tree)
    ↓              ↓
NetworkXBackend    KuzuBackend
(今天)             (将来)
```

ParseTree 只产出树，不知道怎么存。存储由 `StorageBackend` 负责，可以热换。

---

## TreeNode Schema

```python
@dataclass
class TreeNode:
    # ── 通用字段（所有 kind 都有）────────────────────────────────────
    id: str                          # 命名空间内唯一，slug 格式
    kind: NodeKind                   # 见下方 kind 列表
    label: str                       # 人类可读名称
    content: str                     # 全文内容（用于 embedding 和 BM25）
    namespace: str                   # "nimbus" | "code" | "conv"
    source_file: str                 # 来源路径（vault 相对路径或绝对路径）
    tokens: int = 0                  # token 数（BFS budget 用）
    content_hash: str = ""           # SHA1，增量 ingest 用

    # ── 树结构 ───────────────────────────────────────────────────────
    children: list["TreeNode"] = field(default_factory=list)

    # ── Kind-specific 字段 ───────────────────────────────────────────
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata 的 key 按 kind 规范，见下方各 kind 的 schema
```

### Kind 列表与 metadata schema

#### 文档类（nimbus namespace）

| kind | metadata keys |
|------|--------------|
| `note` | `frontmatter: dict`, `tags: list[str]` |
| `knowledge` | `frontmatter: dict`, `tags: list[str]`, `know_id: str` |
| `section` | `heading_level: int` (1–6) |
| `block` | `block_type: str`, `callout_type: str \| None` |
| `tag` | — |
| `category` | — |

`block_type` 枚举值：`"paragraph"` / `"callout"` / `"code_block"` / `"list"` / `"table"`

#### 代码类（code namespace）

| kind | metadata keys |
|------|--------------|
| `module` | `language: str`, `line_count: int` |
| `class` | `language: str`, `line_start: int`, `line_end: int`, `is_exported: bool`, `bases: list[str]`, `docstring: str` |
| `function` | `language: str`, `line_start: int`, `line_end: int`, `signature: str`, `is_exported: bool`, `is_async: bool`, `parameters: list[str]`, `return_type: str \| None`, `docstring: str` |

#### 对话提取类（conv namespace）

| kind | metadata keys |
|------|--------------|
| `fact` | `source_session: str`, `extracted_at: str`, `speaker: str \| None` |

---

## StorageBackend 接口

```python
class StorageBackend(ABC):

    @abstractmethod
    def write_tree(self, tree: ParseTree) -> None:
        """将一棵 ParseTree 写入存储（新增或覆盖）。"""
        ...

    @abstractmethod
    def delete_by_source(self, source_file: str, namespace: str) -> None:
        """删除某文件产出的所有节点（用于增量 ingest 的清理阶段）。"""
        ...

    @abstractmethod
    def has_changed(self, source_file: str, namespace: str, content_hash: str) -> bool:
        """判断文件是否有变化（hash 比对）。"""
        ...
```

`NetworkXBackend` 实现：TreeNode 展平 → `Node` + `Edge(relation="contains")` → DiGraph。  
`KuzuBackend` 实现：TreeNode 按 `kind` 写入对应 node table，containment 写入 REL table。

---

## Kuzu 兼容性设计

### 节点表映射

```
TreeNode.kind       →  Kuzu node table
─────────────────────────────────────
"function"          →  Function { id, label, content, source_file, namespace,
                                   language, line_start, line_end, signature,
                                   is_exported, is_async, docstring }
"class"             →  Class    { id, label, content, source_file, namespace,
                                   language, line_start, line_end, is_exported, bases }
"module"            →  Module   { id, label, content, source_file, namespace, language }
"section"           →  Section  { id, label, content, source_file, namespace, heading_level }
"note"              →  Note     { id, label, content, source_file, namespace, tags }
...
```

`metadata` dict 的 key 直接对应 Kuzu table 的列名，`KuzuBackend` 按 kind 取对应 key 写入。

### 关系表映射

```
TreeNode 父子关系    →  Kuzu REL table
─────────────────────────────────────────────────────
module → class       →  CONTAINS (FROM Module TO Class)
class  → function    →  CONTAINS (FROM Class TO Function)
note   → section     →  CONTAINS (FROM Note TO Section)
section → block      →  CONTAINS (FROM Section TO Block)
```

Wikilink / calls 等横向关系由 Parser 直接产出为 Edge，不来自树结构。

### 查询示例（未来 Kuzu）

```cypher
-- 某函数被哪些模块间接依赖（最多 5 跳）
MATCH (target:Function {id: $id})<-[:CALLS*1..5]-(caller)
RETURN caller, length(path) AS depth ORDER BY depth

-- 某 section 下所有 block 的内容
MATCH (s:Section {id: $id})-[:CONTAINS]->(b:Block)
RETURN b.content
```

---

## 为什么现在就要按这个设计

- `TreeNode.metadata` 的 key 现在就要稳定 → Kuzu table 列名后续不用重新设计
- `kind` 值现在就要稳定 → 直接对应 Kuzu table 名，后续不用映射层
- `StorageBackend` 接口 → 今天实现 `NetworkXBackend`，将来加 `KuzuBackend` 只是一个新类

ParseTree 本身不会因为换存储后端而改变。

---

## Parser 输出契约（Pydantic）

`Parser.parse()` 最终返回 `ParseResult`（而非裸 `ParseTree`）。
`ParseResult` 是 Pydantic 验证模型，在 Parser 出口处即校验数据正确性：

```
Parser.parse()
    ↓
ParseTree              ← 树结构，与存储无关
    ↓
ParseResult.from_tree(tree)   ← Pydantic 校验（NodeRecord/EdgeRecord/tier_float_consistency）
    ↓
StorageBackend.write_result(result)
    ↓
NetworkXBackend / KuzuBackend
```

**为什么在 ParseTree 和 StorageBackend 之间加 Pydantic 层？**

- `ParseTree` 是纯数据结构，不做校验
- `StorageBackend` 是写入逻辑，不应该是校验的责任方
- `ParseResult` 是两者之间的契约：保证"进入存储层的数据在语义上是自洽的"
- 具体校验：`confidence_tier=EXTRACTED` 但 `confidence=0.3` → 在 ParseResult 处即失败，不留到 KuzuDB writer 才爆炸
- 铁律嵌入：`ConvParser` 产出的 edge 中如有 `EXTRACTED` tier，校验器直接拒绝

## ⚠️ 铁律

> **认识论来源不可由频率覆写。**
>
> ConvParser 产出的节点和边最高只能是 `INFERRED`。
> `EXTRACTED` 的唯一合法来源是确定性解析器：
> - `MarkdownParser`：wikilink、tag_ref、frontmatter 显式关系
> - `CodeParser`：Tree-sitter AST 解析的 calls / imports / inherits
>
> 无论一条 fact 被独立提取多少次，无论重复出现在多少个会话中，
> 它的 `confidence_tier` 永远是 `INFERRED`。
>
> 这条规则不设例外，不设升级通道，不可由任何代码路径绕过。
> `EdgeRecord.tier_float_consistency()` validator 内嵌此检查，
> 任何违反都在运行时立即抛出 `ValueError`。

## 实现顺序

1. `prism_rag/ingest/parse_result.py` — `NodeRecord / EdgeRecord / ParseResult`（Pydantic，~100 行）
2. `base_tree.py` — `TreeNode` dataclass + `ParseTree` 封装
3. `storage/backend.py` — `StorageBackend` ABC（接受 `ParseResult` 而非裸树）
4. `storage/networkx_backend.py` — `NetworkXBackend`（展平 ParseResult → DiGraph）
5. `Parser` ABC 改为返回 `ParseResult`
6. `NimbusParser` 改造为产出 `ParseResult`
7. 将来：`storage/kuzu_backend.py` — `KuzuBackend`
