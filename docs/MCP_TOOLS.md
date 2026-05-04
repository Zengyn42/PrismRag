# PrismRag MCP Tools Reference

Server: SSE transport, default port 8102. Start with `PRISM_GRAPHS` env var (see `.prism_env`).

All tools accept an optional `namespace=""` / `scope=""` parameter to target a specific federated graph (e.g. `"nimbus"`, `"code"`). Omit to search all.

---

## 图查询

| Tool | Signature | 用途 |
|------|-----------|------|
| `search_knowledge` | `(query, scope="", mode="bfs", budget=4000, ontology_type="")` | 图遍历检索，返回相关节点子图（含社区归属） |
| `explain_node` | `(node, scope="")` | 节点详情 + 邻居 + 所属社区 |
| `trace_path` | `(from_node, to_node, max_length=5)` | 两节点间最短关系路径 |
| `list_communities` | `(ontology_type="")` | 所有 Leiden 社区及其代表节点 |
| `explore_community` | `(community, ontology_type="")` | 社区成员 + 密度 + god-node |
| `list_namespaces` | `()` | 联邦图命名空间列表（含 indexed_dirs、sample_node_ids） |

## 读写 Vault

| Tool | Signature | 用途 |
|------|-----------|------|
| `read_note` | `(path, namespace="")` | 读笔记（返回 content + frontmatter + cas_hash + mtime） |
| `list_files` | `(directory="", pattern="*.md", recursive=False, namespace="")` | 列目录 |
| `get_frontmatter` | `(path, namespace="")` | 仅取 frontmatter |
| `write_note` | `(path, content, cas_hash="", namespace="")` | 全量写（新建或覆盖；需要 CAS） |
| `patch_note` | `(path, section_heading, new_content, cas_hash="", namespace="")` | 按 heading 改一段 |
| `update_frontmatter` | `(path, updates, cas_hash="", namespace="")` | 合并 frontmatter 字段 |
| `move_note` | `(source, dest, cas_hash="", namespace="")` | 移动/重命名 |
| `delete_note` | `(path, cas_hash="", namespace="")` | 软删到 `.trash/` |
| `manage_tags` | `(path, add=[], remove=[], cas_hash="", namespace="")` | 管理 frontmatter tags |
| `search_files` | `(query, directory="", case_sensitive=False, filename_only=False, max_results=50, namespace="")` | 关键词搜索 |
| `get_links` | `(path, namespace="")` | 该笔记的 outgoing + incoming wikilinks |

## 维护工具

| Tool | Signature | 用途 |
|------|-----------|------|
| `check_drift` | `(namespace="code")` | 扫描 mentions_symbol 边，验证目标节点是否还存在于代码图（检测 post-link drift） |

---

## 节点 ID 格式

- **Vault (nimbus) 节点**：`relative/path/to/note`（无扩展名，`vault_loader` 用 `Path.with_suffix("")` 统一处理 `.md`、`.pdf`、`.png` 等）
- **Code 节点**：`code::relative/path/to/file.py::SymbolName`（带命名空间前缀和 `::` 分隔符）

> `explain_node` / `search_knowledge` 等工具接受带扩展名的路径，`resolve_entry_point` 会自动剥离扩展名后再查找。

## CAS（乐观并发）

所有写操作（`write_note`、`patch_note`、`update_frontmatter`、`move_note`、`delete_note`、`manage_tags`）都支持 `cas_hash` 参数。先用 `read_note` 获取 `cas_hash`，再写入，服务器会验证文件未被并发修改。

写操作完成后自动触发图增量 ingest，改完即可查。
