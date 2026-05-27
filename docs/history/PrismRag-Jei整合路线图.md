---
title: PrismRag → Jei 整合路线图
type: design
status: completed-step-1-2-3
scope: ZenithLoom + PrismRag + NimbusVault
created: 2026-04-18
updated: 2026-04-18
authors: [frankwings, claude]
related:
  - PrismRag Phase 1 MVP 实现详情.md
  - PrismRag Phase 2 实现详情.md
  - atomize-document skill — Jei 文档原子化能力设计.md
  - ../knowledge/PrismRag-v4.0-设计文档.md
  - ../../ZenithLoom/docs/vault/architecture/prism-rag-inbox-design.md
  - ../../ZenithLoom/docs/vault/architecture/knowledge-graph-design.md
tags: [prismrag, jei, mcp, integration, roadmap]
---

# PrismRag → Jei 整合路线图

> **目的**：把 PrismRag 从独立工具变成 Jei（Knowledge Curator）的唯一 vault 后端。替换原 `knowledge_shelf` 子图和 `obsidian-vault` MCP server。
>
> **状态**（2026-04-18）：Step 1 / Step 2 / Step 3 **全部完成并 merge 到各自 repo 的 master**。
> 下一步：真实环境下重启 Jei，端到端验证（T1 → T4 测试矩阵）。

---

## 三步演进

```
Step 1 ✅ PrismRag 自身完整性（完成 2026-04-18 午）
    │
    ▼
Step 2 ✅ Obsidian MCP 合并进 PrismRag（完成 2026-04-18 晚）
    │
    ▼
Step 3 ✅ 移除 knowledge_shelf，Jei 切换到 prism-rag MCP（完成 2026-04-18 晚）
    │
    ▼
Step 4 🔜 真实环境验证（重启 Jei、端到端测试）
```

---

## Step 1 — PrismRag 自身完整性（✅ 已完成）

**分支**：`feat/step1-completion` → merged into master (2026-04-18)
**Spec**：`PrismRag/docs/superpowers/specs/2026-04-18-prismrag-step1-completion-design.md`
**Plan**：`PrismRag/docs/superpowers/plans/2026-04-18-prismrag-step1-completion.md`

### 完成项

- **Section 1** — `prism-rag serve` 启动（stdio/SSE transport 验证）
- **Section 2** — Pass 2 PDF 媒体抽取（pypdf）
- **Section 3** — `knowledge_id` 分支（VaultDocument.id + NodeKind + `relations:` frontmatter → EXTRACTED 边 + `embed: false` 开关）
- **Section 4** — 命名撞车分层仲裁（knowledge_id > canonical > knowledge/ dir > AMBIGUOUS）
- **Section 5** — `ontology_type` 字段 + MCP 按类型过滤
- **Section 6** — vault 写入路径加固（CASConflict + atomic_write + audit JSONL）
- **Section 7** — CLI 集成测试（10 个测试，< 9 秒）
- **Post-review fixes** — write_note TOCTOU 修复；incremental ingest 补 Am 属性

**交付**：27 commits，~1,617 LOC，115 测试通过。

### Step 1 后 PrismRag 的 MCP tool 表

| Tool | 类型 | 说明 |
|---|---|---|
| `search_knowledge` | 查询 | 入口解析 + BFS/DFS 图遍历 |
| `explain_node` | 查询 | 节点 + 邻居 + 社区 |
| `trace_path` | 查询 | 两节点最短路径 |
| `list_communities` | 查询 | Leiden 社区列表 |
| `explore_community` | 查询 | 社区成员 + 密度 + god-node |
| `list_namespaces` | 查询 | 联邦图命名空间 |
| `read_note` | CRUD | 读 markdown + frontmatter + cas_hash |
| `write_note` | CRUD | 写文件（CAS + atomic + audit） |

---

## Step 2 — Obsidian MCP 合并进 PrismRag（🚧 待执行）

### 动因

Jei 当前通过 `knowledge_shelf` 子图调用 `mcp_servers/obsidian/server.py`（在 ZenithLoom 仓库里），而 PrismRag 是独立系统。要让 PrismRag 成为唯一后端，必须把 Obsidian MCP 的 11 个 tool 合并进 PrismRag。

### 能力差距（2026-04-18 clocked）

PrismRag 相对于 Obsidian MCP：
- **查询侧超集** — 多了 5 个图查询工具
- **CRUD 侧子集** — 缺以下 6 个关键能力：

| 缺失工具 | 对 Jei 的影响 |
|---|---|
| `patch_note`（按 section 局部修改） | 改一段要全文重写，费 token、易错 |
| `move_note` / `delete_note` | 无法整理 vault 目录结构 |
| `list_files` | 无法列目录浏览 |
| `get_frontmatter` / `update_frontmatter` | 无法原子修改单个 frontmatter 字段 |
| `manage_tags` | 无法加/删 tags |
| `search_files`（关键词）| PrismRag 只有语义图遍历，没有 grep-style 关键词搜 |
| `get_links` | PrismRag 通过 `explain_node` 覆盖（已有） |

### 关键发现：核心层已经是超集

PrismRag 的 `vault_ops/` 已经包含等价的 `vault.py`、`markdown_ops.py`、`errors.py`，而且 `cas.py` 和 `audit_log.py` 比 Obsidian MCP 版本更新（我们 Step 1 加了 CASConflict、atomic_write、JSONL 审计）。

**→ Step 2 不是改核心层，是把 tool handler 从 ZenithLoom 搬到 PrismRag**。

### 源文件映射（from ZenithLoom → to PrismRag）

| ZenithLoom 路径 | 行数 | 目标 PrismRag 路径 | 动作 |
|---|---|---|---|
| `mcp_servers/obsidian/tools/read.py` | 97 | `prism_rag/mcp_server/vault_tools_read.py` | 复制 + 改 import |
| `mcp_servers/obsidian/tools/write.py` | 227 | `prism_rag/mcp_server/vault_tools_write.py` | 复制 + 改 import + 用新 atomic_write |
| `mcp_servers/obsidian/tools/manage.py` | 255 | `prism_rag/mcp_server/vault_tools_manage.py` | 复制 + 改 import |
| `mcp_servers/obsidian/tools/search.py` | 146 | `prism_rag/mcp_server/vault_tools_search.py` | 复制 + 改 import |
| `mcp_servers/obsidian/tests/*.py` | ~9 files | `PrismRag/tests/test_vault_tools_*.py` | 复制 + 改 import |
| `mcp_servers/obsidian/server.py` | 131 | (merge into) `prism_rag/mcp_server/server.py` | 把 `@mcp.tool()` 注册搬过去 |
| `mcp_servers/obsidian/__main__.py` | ? | (N/A) | 不需要，PrismRag 用 `prism-rag serve` |

**估计规模**：~725 行 tool code + ~500 行测试。

### Step 2 后的 PrismRag MCP tool 表（19 个工具）

查询侧（6 个，Step 1 已有）：
- `search_knowledge`, `explain_node`, `trace_path`, `list_communities`, `explore_community`, `list_namespaces`

CRUD 侧（11 个，Step 2 合并进来，替换现有 read_note/write_note）：
- 读：`read_note`, `list_files`, `get_frontmatter`
- 写：`write_note`, `patch_note`, `update_frontmatter`
- 管理：`move_note`, `delete_note`, `manage_tags`
- 搜索：`search_files`（关键词）, `get_links`（反向链接）

### 冲突处理

现有 PrismRag 的 `read_note` 和 `write_note` 与 Obsidian MCP 的同名工具**功能重叠但实现不同**。合并策略：
- **采用 Obsidian MCP 版本作为 canonical**（它们更成熟、有完整测试）
- 替换时保留 PrismRag 特性：写入后自动触发 `incremental.ingest_file` 更新图；返回的 response 里加 `graph_update` 字段
- 删除 PrismRag `server.py` 里的 Step 1 版 `read_note` / `write_note`

### 设计决策

**D1 — `patch_note` 在 `write_note` 后同样触发增量 ingest**
- 修改 frontmatter / section 本质上改变了 node 的 content_hash，图需要重算这个节点的边
- 实现：所有写路径（write / patch / update_frontmatter / manage_tags）都调 `ingest_file(path)` 更新图
- `move_note` 特殊：改变 node ID（路径变了），需要 `graph.rename_node(old_id, new_id)` — 或简单地删除旧 node 重新 ingest 新路径
- `delete_note` 特殊：把 node 从图里标记为 `status: invalidated`（Phase 2 lifecycle 字段 — 目前先直接 `graph.remove_node`）

**D2 — 保持 MCP tool 命名风格一致**
- Obsidian MCP tool 用 `obsidian_*` 前缀。合并后**去掉前缀**，因为现在它们是 PrismRag 的原生工具
- 即 `obsidian_patch_note` → `patch_note`；`obsidian_search_files` → `search_files`
- 这与 PrismRag 现有命名（`read_note` / `write_note` 已经不带前缀）一致

**D3 — `mcp_servers/obsidian/` 最终删除**
- Step 2 完成并验证后，ZenithLoom 仓库里的 `mcp_servers/obsidian/` 整个目录删掉
- `systemd` 单元、启动脚本中的 obsidian MCP 引用一并清理

**D4 — tests 迁移但不合并**
- 原 `mcp_servers/obsidian/tests/test_tools_*.py` 迁到 PrismRag 的 `tests/` 下，文件名加前缀 `test_vault_tools_*.py` 避免和现有测试冲突
- 测试复用 PrismRag 的 `tests/test_vault_ops_write.py` 里已建的 fixture 约定

### Step 2 不做的事

- **不**改 core（vault / markdown_ops / errors）— PrismRag 版本已经够用
- **不**加新 CRUD 特性 — 只搬运已有能力
- **不**做 batch write（`ingest_batch`）— 那是 Step 2 的可选扩展，可以单独做
- **不**清理 ZenithLoom 仓库 — 那是 Step 3 的事

### Step 2 的风险

1. **测试覆盖的历史差异** — Obsidian MCP 的测试可能依赖 `mcp_servers.obsidian.core` 路径。迁移时要重写 import
2. **graph 更新失败不能阻断 CRUD** — `patch_note` 成功但 `ingest_file` 失败时，要返回 `status: ok` + `graph_update: {error: ...}`，不能让 graph 更新失败导致文件写成功但 response 报错
3. **Jei 的 PROTOCOL.md 路由 JSON 名字需要更新** — 现在说 `{"route": "knowledge_shelf", ...}`，Step 3 切换时要改为直接调工具

---

## Step 3 — 切换 Jei：移除 knowledge_shelf，接入 prism-rag（📋 待执行，依赖 Step 2）

### 前提

Step 2 完成：PrismRag 有完整的 11 个 CRUD tool + 6 个查询 tool。

### 要改的文件

#### 1. `ZenithLoom/blueprints/role_agents/knowledge_curator/entity.json`

**当前状态**（2026-04-18 快照）：
```json
{
  "mcp": [
    {
      "name": "obsidian-vault",
      "module": "mcp_servers.obsidian.server",
      "module_args": ["--transport", "sse", "--port", "8101", "--vault", "/home/kingy/Foundation/Vault"],
      "url": "http://localhost:8101/sse",
      "shared": true
    }
  ],
  "graph": {
    "nodes": [
      { "id": "gemini_main", ... },
      { "id": "validate", ... },
      { "id": "knowledge_shelf", "agent_dir": "blueprints/functional_graphs/knowledge_shelf" },
      { "id": "render_slides", ... },
      ...
    ],
    "edges": [
      ...,
      { "id": "e_validate_2_shelf", "type": "routing_to", "from": "validate", "to": "knowledge_shelf" },
      ...,
      { "id": "e_shelf_2_gemini", "from": "knowledge_shelf", "to": "gemini_main" }
    ]
  }
}
```

**变更**：
- 替换 `mcp.obsidian-vault` 为 `mcp.prism-rag`（路径指向 PrismRag 的 server）
- 删除 `knowledge_shelf` 节点
- 删除 2 条边：`e_validate_2_shelf` 和 `e_shelf_2_gemini`
- **决策点**：Jei 是否还用子图封装？两种方案：
  - (α) 直接用：gemini_main 直接调 prism-rag 的工具（去掉"plan 模式只路由"的约束）
  - (β) 保留子图：新建 `prism_rag` 子图（类似原 knowledge_shelf，但用 prism-rag MCP）
  - **推荐 (α)**：简化结构；Gemini CLI 的 `bypassPermissions` 已允许直接调 MCP

#### 2. `ZenithLoom/blueprints/role_agents/knowledge_curator/PROTOCOL.md`

- 删除 `{"route": "knowledge_shelf", "context": "..."}` 一节
- 改写为「直接调用 PrismRag tools」的说明段
- 保留其他路由（`render_slides`, `render_docs`, `gws_*`）

#### 3. `ZenithLoom/blueprints/functional_graphs/knowledge_shelf/` — **整个目录删除**

#### 4. `ZenithLoom/framework/schema/base.py`

- 删除 `knowledge_result: str` 字段（只被 knowledge_shelf 使用）
- 检查：framework 其他地方如果有读这个字段要一并清理

#### 5. `ZenithLoom/mcp_servers/obsidian/` — **整个目录删除**（Step 2 的 D3）

#### 6. 测试清理

- `test_e2e_mcp.py`、`tests/test_gemini_routing.py` 里如果提到 knowledge_shelf 要更新或删除相关用例

#### 7. Gemini CLI 配置

- `ZenithLoom/.gemini/settings.json` 的 `mcpServers.obsidian-vault` 改成 `prism-rag`
- 或者（推荐）让 ZenithLoom 的 mcp_manager 统一注册，`.gemini/settings.json` 用不上

### Step 3 的风险

1. **Jei 可能突然没 vault 能力** — 如果 Step 2 有 bug，Step 3 切过去 Jei 就瘫了。**建议**：切换前手工跑 `prism-rag serve` 用 MCP inspector 验证所有 11 个工具能跑
2. **systemd 重启需要小心** — Jei 是 systemd unit，重启有风险。先在 staging 验证再改 prod
3. **Jei 的记忆 / session** — PROTOCOL.md 改了，Jei 的系统 prompt 变了，她的历史 session 引用 `knowledge_shelf` 路由可能困惑；重启后问题不大

---

## 测试矩阵（Step 2 + Step 3 完成后）

| 测试层 | 内容 | 目的 |
|---|---|---|
| **L1 单元** | `tests/test_vault_tools_*.py`（从 obsidian 迁移） | 每个工具独立正确性 |
| **L2 集成** | `prism-rag serve` + MCP inspector | 端到端 JSON-RPC 协议 |
| **L3 CLI** | `prism-rag ingest && prism-rag serve` 在 tiny vault 上跑 | 索引 + 查询协同 |
| **L4 Jei** | Discord 里给 Jei 发真实问题，观察她用哪些工具 | 真实 agent 集成 |

---

## 后续（Step 4+，超出本路线图）

1. **Inbox 模式** — 任何 LLM 写 md 到 `Vault/inbox/`，Jei 异步消费、原子化、索引
   - 参见 `ZenithLoom/docs/vault/architecture/prism-rag-inbox-design.md`
2. **Vault Phase 2 数据模型** — 引入 `knowledge_id` frontmatter、REGISTRY、`relations:` schema
   - 参见 `ZenithLoom/docs/vault/architecture/knowledge-graph-design.md`
3. **atomize-document skill** — Jei 把长文档拆成原子 knowledge nodes
   - 参见 `设计细节/atomize-document skill — Jei 文档原子化能力设计.md`
4. **PrismRag future work C** — 碰撞仲裁 LLM / ontology 自动分类（feature-flag 关闭）
   - 参见 `PrismRag Phase 2 实现详情.md`

---

## 决策日志

- **2026-04-18 D1** — Step 1 选择独立 merge 到 master（不 PR）—— 小团队 + 单人维护 + 内部项目
- **2026-04-18 D2** — Obsidian MCP 合并方向：从 ZenithLoom 搬到 PrismRag，**而非反过来** —— PrismRag 是自包含的 RAG 工具，vault 编辑是它的自然职责
- **2026-04-18 D3** — 保留 `vault_ops/` 核心层不动 —— 它已经是 Obsidian MCP core 的超集
- **2026-04-18 D4** — Jei 切换到 prism-rag MCP 后走方案 (α)：gemini_main 直接调工具，不要子图包装 —— 简化架构，现在也没别的 agent 调用 knowledge_shelf
- **2026-04-18 D5** — 工具名去掉 `obsidian_` 前缀 —— 它们现在是 PrismRag 原生工具
- **2026-04-18 D6** — 暂不删除 `ZenithLoom/mcp_servers/obsidian/` —— 作为 rollback 保险，等 Jei 在真实环境下验证通过再清理
- **2026-04-18 D7** — `framework/schema/base.py` 的 `knowledge_result` 字段保留不动 —— 是 dead field 但 framework 多处引用，移除需要协调 schema + subgraph_init_node 改动，非 Step 3 scope

---

## 实际执行记录（2026-04-18）

### PrismRag 提交（master 分支）

Step 1（merged via `08ce44a`）：
- 27 feature commits，115 tests 通过
- 交付：serve 验证、PDF 抽取、knowledge_id 分支、撞名仲裁、ontology_type、atomic write、CAS + audit JSONL、CLI 测试

Step 2（merged via `9f...` merge commit）：
- 5 feature commits，158 tests 通过
- 交付：11 个 Obsidian CRUD 工具搬进 `prism_rag/mcp_server/vault_tools.py`，写路径自动触发增量 ingest

Post-Step-2 增强（`49a95d6`）：
- `serve` 命令加 `--port`、`--vault`、`--data-dir`，方便 ZenithLoom MCP manager 当子进程启动

### ZenithLoom 提交（master 分支）

Step 3（merged via `caac923`）：
- 6 个文件改动：entity.json / PROTOCOL.md / ROLE.md / .gemini/settings.json / 删除 knowledge_shelf/ / test_gemini_routing.py
- Jei 的 MCP server 从 `obsidian-vault` 切到 `prism-rag`
- knowledge_shelf 子图完全移除

### 最终架构

```
Jei (knowledge_curator)
  └── gemini_main
        ├── MCP: prism-rag @ :8102 (SSE)      ← 17 工具（6 查询 + 11 CRUD）
        │     └── 内部 vault_ops / embedder / Leiden / 图遍历
        │
        └── 路由出口（通过 validate 节点）
              ├── render_slides  (Presenton)
              ├── render_docs    (Pandoc)
              ├── gws_slides     (Google Slides API)
              └── gws_docs       (Google Docs API)
```

---

## Step 4 — 真实环境验证清单（🔜 未开始）

执行顺序：

1. **首次 ingest**：`PRISM_GEMINI_API_KEY=... prism-rag ingest --vault /home/kingy/Foundation/Vault --output /home/kingy/Foundation/PrismRag/data`
2. **独立 serve 测试**：`prism-rag serve --transport sse --port 8102 --vault ... --data-dir ...` 起来后用 `curl localhost:8102/sse` 或 MCP inspector 验证
3. **重启 Jei systemd unit**：`systemctl --user restart jei`（注意是 jei 不是 hani；如不确定先 `systemctl --user list-units | grep jei`）
4. **T1 连通性**：Discord 里给 Jei 发 "调用 list_communities 看看"
5. **T2 图查询**：给 Jei 发 "搜一下 fresh_per_call 相关的知识"
6. **T3 对比**：让 Jei 同时用 `search_knowledge` 和 `search_files` 搜同一关键词并比对
7. **T4 写入路径**：让 Jei 创建一个 test note（验证 atomic write + audit JSONL + 自动入图）

### 已知风险

1. **`mcp_manager.py` 启动 prism-rag 子进程时的 env 继承** — 目前 PrismRag CLI 已支持 `--vault` / `--data-dir` 覆盖，entity.json 的 `module_args` 也传了值。但 `PRISM_GEMINI_API_KEY` 还依赖 ZenithLoom 进程 env 继承。若查询侧够用（不需要重新 embed）应无影响。
2. **端口 8102 被占** — obsidian-vault 曾占用 8101；prism-rag 用 8102 避免冲突。但 PrismRag 旧的 ad-hoc 测试可能留着 8102 的僵尸进程，启动前 `lsof -i:8102` 检查。
3. **gemini_main 之前在 "plan 模式"（PROTOCOL.md 明写）** — 已更新 PROTOCOL.md 告诉 Jei "Vault 操作直接调工具"，但 Gemini CLI 对从"必须路由"到"可直接调"的切换可能需要几轮对话 adaptation。验证时多发几条消息观察。

---

## Step 5 清理（待 Step 4 验证通过后做）

- 删除 `ZenithLoom/mcp_servers/obsidian/`（整个目录，~1,333 行 + 测试）
- 清理 `framework/schema/base.py` 的 `knowledge_result` 字段 + 相关 framework 代码
- `test_e2e_mcp.py` 移除 knowledge_shelf 相关的 fixture / 路径检查
- 清理 `docs/superpowers/plans/2026-04-12-unified-subgraph-integration.md` 中的 `knowledge_result` 示例（或标记为历史）
