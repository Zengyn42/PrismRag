---
title: "PrismRag v5.4 — Atomize 体验精打磨"
tags:
  - "#PrismRag"
  - "#GraphRAG"
  - "#architecture"
  - "#knowledge-graph"
  - "#design"
status: spec
created: 2026-05-12
last_audited: 2026-05-12
review_notes: "初稿。覆盖 P1–P4 四项体验修复：vault 路径注入 MCP instructions、explain_node KNOW-ID 先置检查、list_knowledge_nodes body_preview、KNOW 节点 label 改用 frontmatter title。R1：P2 正则 \\d{6} → \\d{6,}（与 v5.3 D22 对齐）；P4 文件名现状描述修正（{kid}-{slug}.md 格式）；P4 代码片段补充 kind 来源说明；对话测试标注手动执行 + @pytest.mark.skip。R2（辩论 R1+R2 结论）：P1 保持单一 namespace 静态注入不变，多 namespace 动态注入作为 v5.5 技术债；P2 升级为四层方案——移除 explain_node 先置检查（死代码），改为双侧 description + search_knowledge 纯 ID soft hint（宽松正则 + 按位数分支文案 + ⚠️ 阻断性语义）+ explain_node 智能错误回执（区分含杂质 ID vs 纯自然语言，防乒乓死循环）；P3 body_preview 从 100 字降至 50 字，加 max_results=20 默认分页，返回值补充 total/returned 字段；P4 新建 label_resolver.py（title→clean_slug→stem 三层 fallback），incremental + vault_loader 都调用；Jei 冒烟测试 3 条固定路径（S1/S2/S3），不进 CI。"
milestone: v5.4
related:
  - "设计细节/PrismRag v5.3 — Vault Phase 2、atomize_document 与 Embedding 增强.md"
  - "设计细节/PrismRag-Jei整合路线图.md"
---

# PrismRag v5.4 — Atomize 体验精打磨

> v5.3 完成了 atomize 三阶段工具族（scan / propose / apply）和 embedding 增强。
> 功能可用，但 v5.3 测试阶段暴露了 4 个体验短板，影响 Jei 的实际使用效果。
> v5.4 修复这 4 项，不引入新工具，不改变任何参数签名。

---

## 零、v5.3 测试发现的问题（v5.4 动机）

v5.3 完成后，对 Jei 进行了两轮对话测试：基础流程测试（T1–T9）和扩展场景测试（E1–E10）。以下是触发 v5.4 四项修复的关键发现。

### E1 — 新 session 首轮路径推断出错（→ P1）

**测试场景**：新开 Jei session，第一轮让她调用 `read_note` 读取一个 vault 文档。

**Jei 的行为**：报告 vault 路径不匹配。Gemini CLI 的 CWD 是 `/home/kingy/Foundation/PrismRag/`，而实际 vault 在 NimbusVault。Jei 将 CWD 当作默认工作空间，拼出的路径不存在。

**续 session 后**：路径推断恢复正常（system prompt 已注入足够上下文）。

**结论**：MCP server 的 `instructions` 字段未声明 vault 绝对路径，新 session 冷启动时 Jei 没有足够的路径线索。

---

### E8 — KNOW-ID 精确查找失败（→ P2）

**测试场景**：让 Jei 查找节点 `KNOW-000008` 的内容和社区归属。

**Jei 的行为**：调用 `search_knowledge("KNOW-000008")`，语义向量搜索对数字 ID 字符串无能为力，结果不稳定或返回不相关节点。

**追问**：如果让 Jei 改用 `explain_node("KNOW-000008")` 呢？

**结果**：`explain_node` 已有精确 ID 匹配路径，直接命中，返回完整节点信息。

**结论**：工具可以工作，但 Jei 不知道应该用 `explain_node` 而非 `search_knowledge` 做 ID 查找。问题在工具描述，不在实现。同时需要加防御性代码，保证未来 label 改变后 ID 查找仍然可靠。

---

### E10 — 知识库主题聚类分析受限（→ P3 + P4）

**测试场景**：让 Jei 用 `list_knowledge_nodes` 获取所有 KNOW 节点列表，然后分析：哪两个节点主题最相近？哪个节点最孤立？

**Jei 的行为**：拿到列表后，发现每个节点只有 `id / label / namespace / tokens` 四个字段。

**追问**：你能基于这些信息做分析吗？

**Jei 的回答**：`label` 字段显示的是 `"KNOW-000043-fresh-per-call-decision"` 这样的带 slug 文件名，没有可读的标题，也没有内容摘要。只能靠 slug 的文字片段猜测主题，分析质量很差。如果有节点的正文摘要，分析会有依据得多。

**结论**：两个问题叠加——label 不可读（P4），且缺乏内容摘要（P3）。两者都需要修复，Jei 才能真正做主题分析。

---

## 一、定位与边界

**一句话**：让 Jei 更可靠地通过 PrismRag 工具理解知识库，消除路径推断错误和 ID 查找失败。

**范围**：仅 PrismRag 仓库内改动，共涉及 4 个源文件（server.py、label_resolver.py 新建、incremental.py、vault_loader.py）+ 2 个测试文件。不改变 MCP 工具接口签名，无 breaking change。

| # | 问题 | 来源 |
|---|---|---|
| P1 | Gemini CLI 新 session 首轮路径推断出错 | v5.3 测试 E1 |
| P2 | `search_knowledge("KNOW-000008")` 无法可靠命中 | v5.3 测试 E8 |
| P3 | `list_knowledge_nodes` 无内容摘要，Jei 无法做主题分析 | v5.3 测试 E10 |
| P4 | KNOW 节点 label 为文件名（"KNOW-000004"）而非实际标题 | v5.3 测试发现 |

---

## 二、各修复详情

### 2.1 P1 — MCP instructions 注入 vault 路径

**问题**：Gemini CLI 将 CWD（`/home/kingy/Foundation/PrismRag/`）当作默认工作空间。MCP server 的 `instructions` 字段未声明 vault 绝对路径，Jei 第一轮工具调用时路径推断出错，续 session 后才恢复正常。

**修复**：`server.py` 顶部（`mcp = FastMCP(...)` 之前），从 `PrismRagSettings` 读取主 vault 路径并注入 instructions。

```python
try:
    _startup_settings = PrismRagSettings()
    _vault_hint = f"Vault root: {_startup_settings.vault_path}. "
except Exception:
    _vault_hint = ""

mcp = FastMCP(
    "PrismRag",
    instructions=(
        "PrismRag: graph-first RAG system for Obsidian vaults and code repositories. "
        f"{_vault_hint}"
        "File paths in atomize_scan/read_note/write_note are relative to vault root. "
        ...
    ),
)
```

Settings 在 server 启动时加载一次，无副作用。`try/except` 保证设置缺失时不影响 server 启动。

**范围说明**：v5.4 只注入主 vault namespace 路径。多 namespace 动态路径注入（需要 `contextvars` 或 session-scoped config）标记为 v5.5 技术债，不在本版本范围内（见 D5）。

---

### 2.2 P2 — KNOW-ID 工具选择引导

**问题**：E8 的根因是 Jei **选错了工具**——调用了 `search_knowledge("KNOW-000008")` 而非 `explain_node`。`search_knowledge` 走向量语义路径，数字 ID 字符串不携带语义信息，结果不稳定。`explain_node` 本身已可精确命中 KNOW-ID（`resolve_entry_points` 已有精确 ID 匹配路径），问题完全在工具描述不够清晰。

**修复**：四层方案，不新增工具，不改参数签名。

**（a）`explain_node` 描述改为动作导向**

docstring 首句改为：
> "根据节点 ID 或名称读取完整节点内容及邻边。**当你持有 KNOW-NNNNNN ID 时，必须使用此工具，而非 search_knowledge**——语义搜索无法可靠匹配数字 ID。"

**（b）`search_knowledge` 描述加交叉引用**

docstring 末尾加一行：
> "如果你已有 KNOW-ID（如 'KNOW-000008'），请改用 `explain_node` 直接读取，不要用此工具做 ID 查找。"

**（c）`search_knowledge` 内部纯 ID soft hint**

当 query **精确匹配** `^KNOW-\d+$`（纯 ID，不含其他文字）时，返回 soft hint 而非执行搜索。hint 文案按位数分支（廉价拦截器，宽松正则避免假阴性）：

```python
import re
_q = query.strip()
_kid_match = re.match(r'^(KNOW-(\d+))$', _q, re.IGNORECASE)
if _kid_match:
    digits = _kid_match.group(2)
    kid_upper = _kid_match.group(1).upper()
    if len(digits) >= 6:
        hint_msg = (
            f"⚠️ 未执行搜索。检测到精确节点 ID 格式（{kid_upper}），"
            f"请立即调用 explain_node(node='{kid_upper}') 查询此节点。"
            "results 为空不代表该知识不存在。"
        )
    else:
        hint_msg = (
            f"⚠️ 未执行搜索。检测到疑似节点 ID 但格式不完整（{kid_upper}，"
            "KNOW-ID 至少需要 6 位数字），请确认完整 ID 后调用 explain_node。"
        )
    return json.dumps({"hint": hint_msg, "results": []}, ensure_ascii=False)
# 正常搜索逻辑继续...
```

**Schema 说明**：`{"hint": "...", "results": []}` 最小契约不变。hint 文案必须满足三要素：① 声明搜索未执行（防止 LLM 将空 results 误判为"无结果"）；② 明确指定目标工具名；③ ⚠️ + 强制性动词（`立即`、`必须`）。

**关键区分**：
- 纯 ID（`KNOW-000008`）→ 明确工具误用，soft hint 无误杀风险
- 混合 query（`对比 KNOW-000008 与 database 的设计`）→ 合法语义搜索，不触发 hint，正常放行

**（d）`explain_node` 错误回执智能分支**（防乒乓死循环）

`explain_node` 不加 KNOW-ID 格式校验（职责边界不变）。但当 `resolve_entry_points` 无法精确匹配时，错误回执需区分两种情况：

```python
if not entries:
    # 尝试从输入中提取合法 KNOW-ID（处理"KNOW-123456 的架构"等混合输入）
    _embedded = re.search(r'KNOW-(\d{6,})', node, re.IGNORECASE)
    if _embedded:
        clean_id = f"KNOW-{_embedded.group(1)}"  # \d{6,} 已保证 6+ 位，无需 zfill
        return json.dumps({
            "error": (
                f"输入包含额外文本，无法精确匹配节点。"
                f"请仅传入纯 ID 重新调用：explain_node(node='{clean_id}')"
            )
        }, ensure_ascii=False)
    else:
        return json.dumps({
            "error": (
                f"未找到精确匹配的节点：{node!r}。"
                "若需语义搜索请使用 search_knowledge。"
            )
        }, ensure_ascii=False)
```

这打破了 search→explain→search 的乒乓死循环：含杂质的 ID 被要求清洗后重试 explain_node（而非踢回 search_knowledge）。

---

### 2.3 P3 — `list_knowledge_nodes` 内容摘要

**问题**：`list_knowledge_nodes` 只返回 `id / label / namespace / tokens`，Jei 无法依据内容对节点做主题对比或聚类分析（E10 场景），只能靠 label 猜测。

**修复**：每个节点追加 `body_preview` 字段，取 content 前 **50 字符**（约 25–40 个中文字，足够辨识节点主题），不加控制参数，始终返回；同时加 `max_results=20` 默认分页上限。

```python
"body_preview": data.get("content", "")[:50],
```

```python
# list_knowledge_nodes 签名增加默认分页（不是新参数，是现有参数的默认值调整）
def list_knowledge_nodes(
    namespace: str = "",
    ktype: str = "",
    status: str = "active",
    max_results: int = 20,   # 默认返回最多 20 条，防止大 vault token 膨胀
) -> str:
```

Token 估算：20 节点 × 50 字 ≈ 1,000 token 增量，可控。

同步更新 docstring，加入 `body_preview` 字段说明及 `max_results` 参数说明。

**关于 `ktype` 和 `status` 参数**：这两个参数在 v5.3 已定义为 reserved（未实装），v5.4 继续保持 reserved 状态，不实装过滤逻辑。v5.5 或后续版本按需实装。

---

### 2.4 P4 — KNOW 节点 label 使用 frontmatter title

**问题**：`VaultDocument.label` = `path.stem`（文件名去掉扩展名）。v5.3 `atomize_apply` 生成的 KNOW 文件名格式为 `{kid}-{slug}.md`（如 `KNOW-000043-fresh-per-call-decision.md`），导致图中 label = `"KNOW-000043-fresh-per-call-decision"`。在力导向图 10-15 字符截断下所有 KNOW 节点显示为 `KNOW-00000...`，图谱可用性为零。

**修复**：新建 `ingest/label_resolver.py`，实现三层 fallback：`title → clean_slug → stem`。`incremental.py` 和 `vault_loader.py` 都调用此 resolver（两个 loader 都负责 KNOW 节点 ingest）。

```python
# prism_rag/ingest/label_resolver.py

def _clean_slug(stem: str) -> str:
    """从 'KNOW-000043-fresh-per-call-decision' 提取可读 slug。"""
    parts = stem.split('-', 2)          # ['KNOW', '000043', 'fresh-per-call-decision']
    return parts[2].replace('-', ' ') if len(parts) > 2 else ''

def resolve_knowledge_label(frontmatter: dict, stem: str) -> str:
    """三层 fallback：frontmatter title → slug 清洗 → 文件名 stem。"""
    return (
        frontmatter.get('title')
        or _clean_slug(stem)
        or stem
    )
```

在 `incremental.py` / `vault_loader.py` 的 Node 构建处：

```python
from prism_rag.ingest.label_resolver import resolve_knowledge_label

# kind 判断：用 frontmatter.get('knowledge_id') 而非 doc.kind
# （doc.kind 的生命周期在 incremental 路径中不确定，frontmatter 在 parse 时已确定可用）
is_knowledge = bool(doc.frontmatter.get('knowledge_id'))
label = resolve_knowledge_label(doc.frontmatter, doc.path.stem) if is_knowledge else doc.label
```

**注意**：改动仅影响新 ingest 的节点。已入图节点的 label 需下一次 `prism-rag ingest --full` 才更新。无需专门迁移命令，full ingest 自然覆盖。P2 的 soft hint 通过 query 字符串匹配，P4 不影响 P2。

---

## 三、测试策略

**原则**：能确定性测试的确定性测试。凡涉及 LLM 工具选择或知识理解的行为，必须用 Jei 对话测试作为验收标准——PrismRag 的最终目的是让 LLM 更好地理解知识集，代码通过单元测试不等于 Jei 能正确使用这些工具。

所有测试脚本持久化到 `tests/`（非 `/tmp/`）。

### Fixture 设计

现有测试模式（来自 `test_knowledge_mcp_tools.py`）：
- `reset_federated` autouse fixture — 每个测试前后重置 `mcp_server._federated` 等全局状态
- `_make_fake_federated(tmp_path)` helper — 构建最小 FederatedGraph stub，直接设置 `mcp_server._federated`

**关键架构决策**：soft hint 逻辑必须插在 `_ensure_federated()` 之前（纯输入校验，无需图）。这决定了测试 fixture 的分组：

| 测试组 | 需要 FederatedGraph | 理由 |
|------|------|------|
| P2 soft hint（KNOW-000008、KNOW-12） | **否** | 提前返回，不触达图加载 |
| P2 混合 query（含自然语言） | 是 | 走向量搜索路径 |
| P2 `explain_node` 混合输入错误 | 是（含 KNOW 节点） | `resolve_entry_points` 需要图 |
| P3 `body_preview` / `max_results` | 是（节点含 content） | `list_knowledge_nodes` 遍历图 |
| P4 `resolve_knowledge_label` | **否** | 纯函数，只测逻辑 |

**`_make_fake_federated` 扩展**：P3/P4 测试需要节点携带 `content`（用于 body_preview）和 `knowledge_id`（用于 explain_node 测试）。现有 helper 已有 `content="A concept"`，无需大改，补全 `knowledge_id` 字段即可。

### 确定性测试（`tests/test_v54_polish.py`）

| 用例 | 需要 Federated | 验证内容 |
|------|------|---------|
| `test_search_knowledge_pure_id_6digit_hint` | **否** | `search_knowledge("KNOW-000008")` 返回 ⚠️ hint，`results=[]` |
| `test_search_knowledge_pure_id_case_insensitive` | **否** | `search_knowledge("know-000008")` 小写同样触发 hint（`re.IGNORECASE`）|
| `test_search_knowledge_pure_id_short_hint` | **否** | `search_knowledge("KNOW-12")` 返回「格式不完整」hint，`results=[]` |
| `test_search_knowledge_mixed_query_no_hint` | 是 | `search_knowledge("对比 KNOW-000008 与 database")` 正常搜索，不触发 hint |
| `test_search_knowledge_know_no_digits_no_hint` | **否** | `search_knowledge("KNOW-")` 无数字，正则不匹配，不触发 hint |
| `test_search_knowledge_hint_id_uppercased` | **否** | `search_knowledge("know-000008")` 触发 hint，且 hint 文案中 ID 为大写 `KNOW-000008`（非 `know-000008`）|
| `test_explain_node_knowid_direct` | 是 | `explain_node("KNOW-000008")` 返回正确节点 |
| `test_explain_node_knowid_case_insensitive` | 是 | `explain_node("know-000008")` 同样命中 |
| `test_explain_node_mixed_input_error` | 是 | `explain_node("KNOW-000008 的架构")` 返回「含额外文本」错误，且错误消息中包含 `explain_node(node='KNOW-000008')` |
| `test_list_knowledge_nodes_has_body_preview` | 是 | 每个节点有 `body_preview`，长度 ≤ 50 |
| `test_list_knowledge_nodes_preview_truncates_at_50` | 是 | content 为 51 字符的节点，`body_preview` 长度恰好等于 50 |
| `test_list_knowledge_nodes_preview_empty_ok` | 是 | 无 content 节点的 `body_preview` 为 `""` 不报错 |
| `test_list_knowledge_nodes_max_results_default` | 是 | 不传 `max_results` 时默认返回 ≤ 20 条，`total` 字段反映真实总数 |
| `test_list_knowledge_nodes_max_results_one` | 是 | `max_results=1` 时返回 1 条，`total` 仍为真实总数 |
| `test_resolve_knowledge_label_title` | **否** | `resolve_knowledge_label({"title": "测试标题"}, "KNOW-000001-some-slug")` = `"测试标题"` |
| `test_resolve_knowledge_label_clean_slug` | **否** | `resolve_knowledge_label({}, "KNOW-000043-fresh-per-call-decision")` = `"fresh per call decision"` |
| `test_resolve_knowledge_label_fallback_stem` | **否** | `resolve_knowledge_label({}, "KNOW-000001")` = `"KNOW-000001"`（无 slug 时退回 stem）|

### Jei 冒烟测试（`tests/test_jei_v54_smoke.py`）

> **执行方式**：手动运行，不进 CI，不阻塞实现。标记 `@pytest.mark.skip(reason="manual: requires live Jei session")`。验收时由开发者执行 3 条固定路径，判定标准：工具选择正确 = pass，不做统计分析。

| 编号 | query / 操作 | 预期行为 | 覆盖 |
|------|------------|---------|------|
| S1 | 新 session，第一轮 `read_note <已知文档>` | 路径推断正确，不报错 | P1 |
| S2 | `KNOW-000008 的内容是什么？` | 断言A：`search_knowledge` 返回含 hint 的 JSON 且 `results==[]`；断言B：Jei 最终调用 `explain_node` 并呈现节点内容。两层都 pass 才整体 pass | P2 |
| S3 | `列出所有 KNOW 节点，分析哪两个主题最相近` | Jei 调用 `list_knowledge_nodes`，回复包含具体 body 内容依据 | P3/P4 |

---

## 四、文件变更一览

| 文件 | 变更 |
|------|------|
| `prism_rag/mcp_server/server.py` | P1 vault 路径注入 instructions；P2 `explain_node` 描述动作导向 + `search_knowledge` 描述交叉引用 + 纯 ID soft hint + explain_node 智能错误回执；P3 `list_knowledge_nodes` 加 `body_preview`（50字）+ `max_results=20` 默认 |
| `prism_rag/ingest/label_resolver.py` | **新建**：P4 `resolve_knowledge_label(frontmatter, stem)`，三层 fallback（title → clean_slug → stem）|
| `prism_rag/ingest/incremental.py` | P4 Node 构建处调用 `resolve_knowledge_label` |
| `prism_rag/ingest/vault_loader.py` | P4 Node 构建处调用 `resolve_knowledge_label` |
| `tests/test_v54_polish.py` | 新建：P2/P3/P4 确定性测试（含 soft hint 正反用例、max_results 默认值、resolver 三层 fallback）|
| `tests/test_jei_v54_smoke.py` | 新建：P1/P2/P3 Jei 冒烟测试（3 条固定路径，手动执行，不进 CI）|

---

## 五、不在 v5.4 范围内

- **多 namespace 动态 vault 路径注入**（需 `contextvars` 或 session-scoped config）→ v5.5 技术债
- `atomize_status` 工具（独立功能，延后）
- P5 测试关键词脆弱性（测试基础设施，独立处理）
- 已入图节点的 label 批量回填（需专门迁移工具）
- 多模态 embedding（参见《PrismRag Multimodal Embedding — 设计方向与路线图》）

---

## 六、决策日志

- **D1：P2 最终方案 = 双侧 description + `search_knowledge` soft hint + `explain_node` 智能错误回执** — v5.3 P2 建议新增 `get_node` 工具；v5.4 初稿改为在 `explain_node` 内加先置检查（prior check）。辩论 R1 发现：E8 路径是 Jei→`search_knowledge`，explain_node 内的 prior check 是死代码。辩论 R2 进一步发现：LLM 可能传入混合输入（`"KNOW-000008 的架构"`），若 explain_node 的错误回执直接踢回 search_knowledge，会造成 search→explain→search 乒乓死循环。最终四层方案：① explain_node 描述动作导向；② search_knowledge 描述加交叉引用；③ search_knowledge 纯 ID soft hint（宽松正则 `^KNOW-\d+$`，按位数分支文案，⚠️ + 阻断性语义）；④ explain_node 错误回执智能分支（含杂质 ID → 要求清洗重试 explain_node，纯自然语言 → 引导 search_knowledge）。正则选宽松而非 `\d{6,}` 是为了捕获格式不完整的误用（廉价拦截后给不同文案）。
- **D2：`body_preview` 固定 50 字符（非 100 字），加 `max_results=20` 默认分页** — 初稿 100 字源于 v5.3 P3 建议（200 字压缩版），辩论后推翻：`list_knowledge_nodes` 无相关度排序，50 字已足够辨识主题（≈25-40 个中文字），更少 token 更可靠。同时 `max_results=20` 防止大 vault 时 token 膨胀（20 节点 × 50 字 ≈ 1,000 token 增量可控）。不加 `include_summary` 控制参数——参数越少越好，始终返回避免 Jei 需要额外传参才能获取基础信息。
- **D3：P4 新建 `label_resolver.py`，incremental + vault_loader 都调用；判断用 `frontmatter.get('knowledge_id')` 而非 `doc.kind`** — 辩论发现两处设计都比初稿更优：① 两个 loader 都负责 KNOW 节点 ingest，若只改 incremental.py，vault_loader 路径的 KNOW 节点 label 不会更新；② `doc.kind` 在 incremental 路径中的生命周期就绪性无法在纯讨论中确认，而 `frontmatter.get('knowledge_id')` 在文件 parse 时已确定可用，是零风险选项。注：v5.3 已为 `vault_loader` 路径定义了完整的 `kind` 生命周期（知识节点设为 `NodeKind.KNOWLEDGE`），但 `incremental` 路径是否同步保证不能仅凭设计文档确认，需查实现代码。使用 `frontmatter.get('knowledge_id')` 是对两条路径都成立的最稳健判断，不依赖 `kind` 的设置时机。fallback 链升级为三层（title → slug 清洗 → stem），slug 清洗（`split('-', 2)` + `replace('-', ' ')`）在力导向图截断场景下是图谱可用性的 0/1 分界，而非"可读性微幅提升"。`VaultDocument.label` 不改，影响面不变。
- **D4：Jei 验收测试 = 3 条冒烟测试（非全流程对话测试）** — PrismRag 的最终受益者是调用它的 LLM，代码正确但 LLM 选错工具等于未修复，必须有 LLM 行为验证。但全流程对话测试依赖 Gemini CLI session 和 Quota，不适合 CI。改为 3 条固定路径冒烟测试（S1/S2/S3），验收时手动执行，判定标准是工具选择正确与否，不做统计分析。标记 `@pytest.mark.skip` 不进 CI。
- **D5：P1 多 namespace 动态 vault 路径注入推迟至 v5.5** — v5.4 只注入主 namespace 路径（单 vault 场景覆盖绝大多数实际用例）。多 namespace 场景需要 `contextvars` 或 session-scoped config 才能在请求级别动态注入正确路径，架构变化较大。v5.3 测试 E1 的失败发生在单 namespace 场景下（路径完全缺失），v5.4 的静态注入已修复该场景。多 namespace 动态注入标记为 v5.5 技术债。
- **D6：soft hint 逻辑必须插在 `_ensure_federated()` 之前** — `search_knowledge` 首行即调用 `_ensure_federated()` 加载整个图。soft hint 是纯输入校验（query 字符串匹配），无需图数据。若插在之后，每个 soft hint 测试都需构建 fake FederatedGraph，测试复杂度大幅增加，且会为明显误用查询触发不必要的图加载。插在之前：① 测试无需 fixture，极简；② 快速失败语义清晰；③ 避免无效图加载。`label_resolver` 同理，纯函数直接单元测试，无需任何 fixture。
