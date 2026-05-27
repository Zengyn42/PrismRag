---
name: PrismRag version status
description: PrismRag release history, what each version added, and test coverage status
type: project
originSessionId: 6d54d2e1-ac3a-472e-9e85-aff67972652c
---
PrismRag MCP server lives at `/home/kingy/Foundation/PrismRag/`, running SSE mode on port 8102.

## v5.2 — md↔code 跨命名空间链接（COMPLETE, tested 2026-05-08）

**What:** Built edges between nimbus namespace (vault markdown docs) and code namespace (source code symbols).

Two edge types:
- `mentions_symbol` — explicit wikilink/symbol references from vault docs to code nodes (7 edges)
- `embedding_similar` — semantic similarity via embeddings (129 edges)

Total: 136 cross-namespace edges.

**Test coverage (F/G/H series, all passed):**
- F-series 20/20 — baseline MCP tool correctness (code search, graph stats, vault CRUD, multi-hop, semantic)
- G-series 5/5 — md↔code link existence and quality (mentions_symbol accuracy, explain_node both directions, embedding_similar quality, bidirectional traversal)
- H-series 4/4 — link usability (impact cross-namespace, trace_path cross-namespace, pending_edges state)

**Known gap:** `review_pending_edge` approval flow not exercised (pending list was empty — all v5.2 edges auto-approved at ingest time). Needs re-test when next ingest produces pending edges.

**MCP tool descriptions:** Rewritten to Anthropic standard (what/returns/NOT/siblings) in commit `2e94d98`. No behavioral instructions in descriptions.

## v5.3 — COMPLETE (2026-05-09)

Three sprints, all implemented and tested.

### Sprint A — Embedding Enhancements
- `embed_cache.jsonl` checkpoint/resume: `_load_embed_cache`, `_append_cache_entry`, `gc_embed_cache` in `embedder.py`
- Timeout 300→60s
- `detect_model_device` (Ollama `/api/ps` → gpu/cpu/unknown)
- `embed-status` CLI command (per-namespace progress)
- `load_vault` now returns `(docs, live_sha_set)` for GC — all callers updated
- Tag: `v5.3-sprint-a`

### Sprint B — Vault Phase 2 Data Model
- `Node.knowledge_id: str | None = None` first-class field (was only in metadata)
- `prism_rag/store/registry.py` — `Registry` class with thread+flock-safe `alloc_id()`, `batch_alloc()`
- MCP tools: `alloc_knowledge_id(count=1)`, `list_knowledge_nodes(namespace="")`
- Tag: `v5.3-sprint-b`

### Sprint C — atomize_document MCP Tool Suite
- `prism_rag/ingest/atomize.py`: `atomize_scan_impl`, `atomize_propose_impl`, `atomize_apply_impl`, `StaleDocError`, `ScanExpiredError`
- MCP tools: `atomize_scan`, `atomize_propose`, `atomize_apply` registered in `server.py`
- `prism_rag/cli_atomize.py`: `prism-rag atomize list/show/apply` CLI group
- Three-phase protocol: scan→propose→apply with crash recovery (doc self-proof idempotency)
- Scan cache: `data/atomize-proposals/scan_cache/<scan_id>.json` (24h TTL)
- Proposals: `data/atomize-proposals/pending/<id>.json` → `applied/` on success
- Tag: `v5.3-sprint-c`

**Test coverage:** 443 unit tests pass (5 pre-existing failures unchanged)

**I+J series Jei integration tests (all passed):**
- I1: regression — search_knowledge still works
- J1: alloc_knowledge_id allocated KNOW-000001 through KNOW-000003
- J2: list_knowledge_nodes returned 0 nodes (vault not yet atomized — correct)
- J3: atomize_scan scanned real vault doc, returned 7 sections with headings + token estimates
- J4: atomize_propose created proposal `2241f426`, claim_count=1

**Key operational note:**
PrismRag SSE server runs on port 8102. When code changes, must kill old process and restart:
`kill $(pgrep -f "prism_rag.cli serve")` then restart with `.venv/bin/python -m prism_rag.cli serve --transport sse --port 8102`

Jei test runner must use unique session names + `ctrl.switch_session()` to force fresh Gemini KV cache (avoids stale tool schemas from old sessions).

## v5.4 — Atomize Polish（COMPLETE）

- P1+P3 实现
- P2 KNOW-ID 路由
- P4 label resolver（`resolve_knowledge_label` 应用到 `VaultDocument.label`）
- 16 个新确定性测试
- Tag: `v5.4.0`

## v5.5 — Semantic Dedup（COMPLETE）

- `atomize_propose` 加入语义去重（cosine similarity 阈值防重复 KNOW 节点）
- `atomize_apply` reuse path
- MCP tools: `rollback_dedup`, `list_dedup_log`
- 3 个 dedup bug 修复（smoke test 发现）
- Tags: `v5.5.0`, `v5.5.1`

## v5.6 — Graph Visualization（COMPLETE）

- Obsidian URI 支持
- Portal nodes
- Federation meta-graph（已在同版本 refactor 移除，defer 到 v6.0）
- 当前测试：**511 tests**
- Tag: HEAD（无独立 v5.6 tag）

## v5.7 — ingest-project 统一 ingest（COMPLETE）

- 新增 `prism-rag ingest-project` CLI 命令
- 一次命令同时处理：Python 代码（Tree-sitter）+ Markdown 文档（vault_loader）
- 合并进同一个 `KnowledgeGraph` → 一个 `graph.json` + 一个 `graph.html`
- embed cache 命中已有代码节点，只需对新文档增量计算
- 修复了原来 `ingest-code` 无 viz、`ingest` 无代码 的两个缺口
- 首次 ingest：Pulsify 1,684 nodes（1,643 code + 41 docs），95 communities

## v6.0 — 设计规划（未实现）

### By-demand Graph Loading
- **Graph Registry 文件** (`graph_registry.json`) — `ingest-code` 跑完自动注册，Server 启动时合并加载
- **`FederatedGraph.add_namespace(src)`** — 运行时热加载新 namespace
- **MCP tool `mount_graph(namespace, data_dir, vault_path=None)`** — Agent 可在对话中动态挂载任意 graph，可选持久化到 registry
- 目标：无需修改 `.env` / `PRISM_GRAPHS`，ingest 完即可 on-demand 查询

### 背景
- 触发：Pulsify 项目 ingest 后，讨论如何让 MCP Server 动态加载各项目 graph
- 当前限制：`PRISM_GRAPHS` 静态配置，`FederatedGraph` 只支持已有 namespace 的 mtime 热重载

## GeminiCLINode cold-start behavior (learned during testing)

Each `ctrl.run()` call spawns a fresh `gemini` subprocess. Cold start with Jei's ~10k token context (persona + 19 MCP tool schemas) takes 120-180s on fresh session. With `--resume session_id`, API has KV cache → 30-60s per call.

Test runner pattern that works: single session across all questions (no thread_id rotation). See `/tmp/jei_ftest_runner.py` for reference.
