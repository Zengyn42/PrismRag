# PrismRag MCP Tools Reference

Server: supports stdio and SSE transports, default SSE port 8102. Start with `prism-rag serve`.

All tools accept an optional `namespace=""` / `scope=""` parameter to target a specific federated graph (e.g. `"nimbus"`, `"code"`). Omit to search all.

---

## Search & Query

### search_knowledge

**Parameters**: `query` (str), `budget` (int, 4000), `mode` (str, "bfs"), `scope` (str, ""), `ontology_type` (str, ""), `min_confidence` (float, 0.0)

Search the knowledge graph by topic, symbol name, or natural language query. Ranking uses BM25 keyword + embedding vector + exact name match, fused via RRF. Returns JSON with entry node and traversed neighbors up to the token budget.

- `scope="nimbus"` for vault/design-doc nodes; `scope="code"` for code symbols; `scope=""` for both.
- `mode="bfs"` for broad topic context; `mode="dfs"` to follow call/reference chains.
- `ontology_type` filters to a specific type e.g. "decision", "concept", "fact".
- Pure KNOW-ID queries are rejected with a hint to use `explain_node` instead.

**Returns**: JSON with `entry_point`, `namespace`, `total_nodes`, `total_tokens`, and `nodes` list (each with id, label, kind, ontology_type, tokens, community, content).

### explain_node

**Parameters**: `node` (str), `scope` (str, "")

Read full node content and edges by node ID or name. When you have a KNOW-NNNNNN ID, use this tool instead of `search_knowledge`.

**Returns**: JSON with `node` (full metadata + content), `namespace`, `outgoing_edges`, `incoming_edges`, and `community` info.

### trace_path

**Parameters**: `from_node` (str), `to_node` (str), `max_length` (int, 5), `scope` (str, ""), `min_confidence` (float, 0.0)

Find the shortest undirected path between two graph nodes, including cross-namespace paths via bridge edges.

**Returns**: JSON with `path_length` and `steps` list (each step has node metadata and `edge_to_next` relation). Error if no path exists within `max_length`.

### communities

**Parameters**: `namespace` (str, ""), `community_id` (str, ""), `ontology_type` (str, "")

List all Leiden communities in a namespace, or drill into a specific community's members and bridge edges. Merges the former `list_communities` and `explore_community` tools.

- No `community_id` -> returns community list with label, member_count, god_nodes, internal_density.
- With `community_id` -> returns members list and top_bridge_edges to neighboring communities.
- `community_id` accepts exact ID (e.g. "community_000"), namespace::ID, or label substring match.

**Returns**: JSON with community overview or detailed member/edge listing.

### impact

**Parameters**: `target` (str), `direction` (str, "upstream"), `max_depth` (int, 3), `min_confidence` (float, 0.7), `scope` (str, ""), `allowed_tiers` (str, "EXTRACTED,INFERRED"), `allowed_edge_kinds` (str, ""), `path_score_fn` (str, "weakest_link")

Return the blast radius of changing a graph node: affected symbols grouped by depth, confidence tiers, and cross-namespace vault mentions. Read-only.

- `direction`: "upstream" (callers/dependents), "downstream" (dependencies), or "both".
- `allowed_tiers`: comma-separated confidence tiers to include; default excludes AMBIGUOUS.
- `allowed_edge_kinds`: comma-separated edge kind filter e.g. "calls,imports"; empty includes all.
- `path_score_fn`: "weakest_link" (min edge confidence) or "cumulative_decay" (multiply scores).

**Returns**: Markdown report with affected symbols by hop depth, plus vault mentions section for cross-namespace references.

---

## Graph Management

### generate_graph

**Parameters**: `namespace` (str, "")

Generate an interactive HTML knowledge graph visualization. Produces `graph.html` in the namespace data directory.

**Returns**: JSON with `paths` list of generated HTML file path(s).

### list_namespaces

**Parameters**: *(none)*

Return all loaded knowledge graph namespaces with node/edge/community counts, indexed directories, and sample node IDs. Reflects state at server startup.

**Returns**: JSON with `namespaces` list (per-namespace stats), `bridges` count, and `total_nodes`.

---

## Knowledge Atomization

### atomize_scan

**Parameters**: `doc_path` (str)

Scan a vault document and return its section structure for atomization. Content is NOT returned, only structure. Accepts paths relative to vault root.

**Returns**: JSON with `scan_id`, `doc_sha`, and a list of `sections` (heading, section_id, token_estimate).

### atomize_propose

**Parameters**: `scan_id` (str), `claims` (list[dict])

Submit semantic claims for a scanned document to create an atomization proposal. Each claim must have: `section_id` (from atomize_scan), `knowledge_id` (from alloc_knowledge_id), `title`, `body`, `ontology_type`.

Semantic deduplication: when embedding store is available, similar existing nodes are flagged with `similar_existing` and `claim_status='needs_review'`. Set `action='reuse'` and `reuse_id` on a claim to reuse an existing node.

**Returns**: JSON with `proposal_id` and claim summaries.

### atomize_apply

**Parameters**: `proposal_id` (str)

Apply an atomization proposal: create `knowledge/*.md` files and patch source document. Idempotent: already-applied KNOW-IDs are skipped (crash recovery). Reuse claims add MENTIONS edges instead of creating new files.

**Returns**: JSON with `applied_count`, `reused_count`, `knowledge_files`, and `reused_nodes`. Error if source document changed since propose (StaleDocError).

### alloc_knowledge_id

**Parameters**: `count` (int, 1)

Allocate one or more globally unique KNOW-IDs (format: KNOW-NNNNNN) for new knowledge nodes. IDs are never reused.

**Returns**: JSON with `ids` list.

### list_knowledge_nodes

**Parameters**: `namespace` (str, ""), `ktype` (str, ""), `status` (str, "active"), `max_results` (int, 20)

List knowledge nodes (kind="knowledge") in the graph. Iterates all knowledge nodes then truncates. Use `search_knowledge()` for topic queries.

**Returns**: JSON with `nodes` list (id, label, knowledge_id, namespace, tokens, body_preview), `total`, and `returned` count.

---

## Edge Management

### pending_edges

**Parameters**: `edge_id` (str, ""), `top_n` (int, 10), `sort_by` (str, "confidence")

List or inspect cross-namespace edges awaiting human review in the EdgeClassifier inbox.

- No `edge_id` -> returns pending queue (top_n entries sorted by confidence, created_at, or consecutive_seen).
- With `edge_id` -> returns vault_context and code_context for that entry.

**Returns**: JSON with pending queue or detailed vault/code context for a specific edge.

### review_pending_edge

**Parameters**: `edge_id` (str), `decision` (str), `note` (str, "")

Apply an approve or reject decision to a pending cross-namespace edge. Approved edges are committed to the vault graph immediately and saved. `decision` must be "approve" or "reject".

**Returns**: JSON with `status`, `edge_id`, `decision`, and `note`.

---

## Drift & Dedup

### check_drift

**Parameters**: `vault_namespace` (str, "nimbus"), `code_namespace` (str, "code"), `include_valid` (bool, False)

Scan all `mentions_symbol` edges in the vault graph and check whether each referenced code symbol still exists in the code graph. Only catches post-link drift (symbol renamed or deleted after link creation).

**Returns**: JSON with `stale_count`, `valid_count`, and `stale_refs` list (doc label, symbol name, target_id, tier, score). Set `include_valid=True` to also return valid_refs.

### flag_drift

**Parameters**: `vault_namespace` (str, "nimbus"), `code_namespace` (str, "code")

Find stale `mentions_symbol` edges, then flag all KNOT nodes linked from those docs as `status="suspected"`. Combines `check_drift` detection with automatic KNOT lifecycle update. Idempotent. Saves the vault graph after flagging.

**Returns**: JSON with `flagged_count`, `stale_doc_count`, and per-doc `details` (doc, stale_symbols, flagged_knots).

### rollback_dedup

**Parameters**: `decision_id` (str)

Undo the graph effects of a "reuse" dedup decision recorded in `dedup_log.jsonl`. Removes the MENTIONS edge and any CONTEXT_REF nodes created during that apply run, then marks the snapshot "rolled_back". Only "reuse" decisions have graph effects; "create" decisions are no-ops.

**Returns**: JSON with `status`.

### list_dedup_log

**Parameters**: `top_n` (int, 10)

List recent dedup decisions from `dedup_log.jsonl` (newest first).

**Returns**: JSON list with `decision_id`, `action`, `claim_title`, `reused_id`, `similarity_score`, `source_doc`, `timestamp`, `rollback_status`.

---

## Community Intelligence

### generate_community_reports

**Parameters**: `vault_namespace` (str, "nimbus"), `min_members` (int, 3)

Generate LLM-written community reports for all qualifying Leiden communities. Reports are cached to `community_reports.json`. Requires Ollama running locally. Read-only (does not modify the graph).

**Returns**: JSON with `report_count`, `namespace`, and `reports` list (community_id, title, rating, summary).

### global_ask

**Parameters**: `question` (str), `vault_namespace` (str, "nimbus")

Answer a question using map-reduce over all community reports in the knowledge graph. Uses pre-cached reports if available; generates on-the-fly otherwise. Requires Ollama. Read-only.

Prefer `search_knowledge()` for targeted node lookup. Use `global_ask` for broad cross-topic questions spanning multiple communities.

**Returns**: JSON with `question` and `answer` (synthesized natural-language).

---

## Vault CRUD (11 tools)

| Tool | Signature | Description |
|------|-----------|-------------|
| `read_note` | `(path, namespace="")` | Read note content + frontmatter + cas_hash + mtime |
| `list_files` | `(directory="", pattern="*.md", recursive=False, namespace="")` | List files in directory |
| `write_note` | `(path, content, cas_hash="", namespace="")` | Full write (create or overwrite; CAS required) |
| `patch_note` | `(path, section_heading, new_content, cas_hash="", namespace="")` | Replace one heading-delimited section |
| `update_frontmatter` | `(path, updates, cas_hash="", namespace="")` | Merge frontmatter fields |
| `move_note` | `(source, dest, cas_hash="", namespace="")` | Move/rename; graph node remapped |
| `delete_note` | `(path, cas_hash="", namespace="")` | Soft-delete to `.trash/` |
| `manage_tags` | `(path, add=[], remove=[], cas_hash="", namespace="")` | Add/remove frontmatter tags |
| `search_files` | `(query, directory="", case_sensitive=False, filename_only=False, max_results=50, namespace="")` | Keyword search over filenames/content |
| `get_links` | `(path, namespace="")` | Outgoing + incoming wikilinks |
| `get_frontmatter` | `(path, namespace="")` | Return only YAML frontmatter (impl exists, not yet registered) |

All write tools support CAS (optimistic concurrency): read `cas_hash` via `read_note`, pass it on write. Writes trigger incremental graph sync automatically.

---

## Node ID Format

- **Vault (nimbus) nodes**: `relative/path/to/note` (no extension)
- **Code nodes**: `code::relative/path/to/file.py::SymbolName` (namespaced with `::` separators)

## CAS Protocol

All write operations support `cas_hash`. Read first with `read_note` to get the hash, then pass it on write. Server verifies the file has not been concurrently modified.
