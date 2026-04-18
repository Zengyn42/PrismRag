# PrismRag Step 1 Completion — Design Spec

> **Date:** 2026-04-18
> **Status:** Draft, awaiting review
> **Scope:** Complete the remaining work in PrismRag so it can be used by
> live agents. Deliberately excludes Obsidian MCP merge, ZenithLoom
> integration, and LLM-inside-PrismRag features (all deferred to later steps).

## Summary

PrismRag's v4.0 design implements a five-pass graph-first indexing pipeline
for Obsidian vaults. Passes 1, 3, 4, and 5 are production-ready. What
remains are seven discrete pieces of work that, together, bring PrismRag to
"usable by external agents over MCP, correct under realistic inputs,
forward-compatible with Vault Phase 2 (knowledge_id)."

This spec covers only those seven pieces. It assumes:

- School B (index-time embedding, zero LLM inside PrismRag) stays intact.
- Vault is in Phase 1 today. Phase 2 (knowledge_id, relations frontmatter)
  is not a precondition but is anticipated — the code is forward-compatible
  via feature-detecting branches.
- Non-Jei agents may write raw markdown to the vault. Atomization is Jei's
  responsibility via a separate skill (atomize-document) and happens out-
  of-band. PrismRag indexes whatever state the vault is in.

## Non-goals for this step

The following are intentionally excluded. They are tracked as future work.

- Merging the Obsidian MCP tools into PrismRag's MCP surface.
- Wiring PrismRag into ZenithLoom agent blueprints.
- LLM-based collision arbitration, ontology classification, or atomization
  inside PrismRag (the "F2/F3" direction). After reviewing mem0's pivot
  away from LLM-driven update arbitration, this direction is deliberately
  deferred.
- Image and audio media extraction. PDF is the only media type in scope.
- Automatic `mentioned_in` reverse-index maintenance.

## Section 1 — `prism-rag serve`: MCP Transport

**Motivation.** The eight MCP tools in `prism_rag/mcp_server/` are
implemented as FastMCP handlers but there is no command-line entry point
that starts the server. Agents cannot call PrismRag today.

**Design.**

- Add `prism-rag serve` CLI subcommand with two transports: `--transport
  stdio` (default, for subprocess agents) and `--transport sse --port N`
  (for standalone HTTP SSE).
- On startup: load `PrismRagSettings`, load `graph.json` into memory via
  `KnowledgeGraph.load()`, load `FederatedGraph` if multiple vaults are
  configured, register the eight tool handlers, start the FastMCP server.
- Exit cleanly on SIGINT / SIGTERM.

**Out of scope.** Auto-ingest on startup. Hot reload of graph.json. Reload
via MCP tool. These can be added later without breaking the interface.

## Section 2 — Pass 2: PDF Media Extraction

**Motivation.** `NodeKind` enum reserves slots for `image`, `pdf`, and
`audio`, but no pass produces such nodes. PDFs in the vault (design
references, uploaded docs) are invisible to the graph.

**Design.**

- New module `prism_rag/ingest/media_extractor.py`.
- PDF handler: `pypdf` extracts full text; pages are concatenated with
  `\n\n--- Page N ---\n\n` separators. If no extractable text
  (scan-only PDFs), log a warning and skip.
- `vault_loader.discover_markdown_files` is renamed to
  `discover_vault_files` and now returns both `.md` and `.pdf` paths.
- New dataclass `VaultMedia` parallel to `VaultDocument`. Its `id` is the
  relative path (same rule as markdown documents).
- Nodes emitted with `kind="pdf"`. Content truncated to 30k characters to
  match embedder limits.
- Image and audio handlers stubbed: module defines `extract_image` and
  `extract_audio` functions that raise `NotImplementedError`.

**Edges.** The AST extractor already recognises `![[embed.png]]` as an
embed relation; the corresponding logic for `![[report.pdf]]` already
works. When the embedded PDF is now a real node (not a stub), these
embed edges resolve cleanly. Similarity edges are produced by Pass 3b
as with markdown.

**Out of scope.** OCR for scan-only PDFs. Image understanding. Audio
transcription.

## Section 3 — `knowledge_id` Branch

**Motivation.** Phase 2 knowledge nodes will have a `knowledge_id`
frontmatter field that should become the node's identity (independent of
file path). Today, PrismRag is always file-path-keyed. The code needs a
forward-compatible branch so Phase 2 files are handled correctly when they
appear, while Phase 1 files behave exactly as before.

**Design.**

- `VaultDocument.id` property: if `frontmatter.knowledge_id` exists, return
  it (as a string); otherwise fall back to the relative-path ID (current
  behavior). Zero change for Phase 1 vaults.
- `NodeKind` adds `"knowledge"` as a value. Nodes with a `knowledge_id`
  are created with `kind="knowledge"`; others remain `kind="note"`.
- `_build_doc_index` in `ast_extractor.py` additionally registers
  `knowledge_id` as a lookup key, so `[[KNOW-042]]` resolves to the node
  whose `knowledge_id` is `KNOW-042`.
- New extractor pass reads `frontmatter.relations` and emits one
  `EXTRACTED` edge per `depends_on / supersedes / references /
  contradicts / refines` entry. `relation` field on the Edge matches the
  declared relation type (not generic `links_to`). `source_pass="ast"`.
- `embedder.py` honours `frontmatter.embed`. If `embed is False`
  (literal), Pass 3a skips that node; it still appears in the graph but
  has no embedding and therefore participates in no similarity edges.
  Default (no `embed` key) is to embed — preserves backward compatibility.

**Out of scope.** Automatic `superseded_by` reverse-population. Automatic
`mentioned_in` maintenance. Filtering by `status` in queries
(status is persisted, but query-time filtering is a future enhancement).

## Section 4 — Name Collision Resolution

**Motivation.** Two vault files can share a filename stem or alias,
especially during Phase 1→Phase 2 migration. The current `_build_doc_index`
silently overwrites on collision, giving the last-processed file
unpredictable authority.

**Design — tiered priority (no LLM in Step 1).**

Collision resolution uses a deterministic priority chain. Higher-priority
signals win:

1. Presence of `knowledge_id` (committed identity).
2. `canonical: true` in frontmatter.
3. File lives under `knowledge/` directory.
4. (Fallback) same-tier collision → emit `AMBIGUOUS` node and log warning.

Implementation approach: build the doc_index in two passes. First pass
records all candidates per key. Second pass applies the priority function
to select the winner or mark as AMBIGUOUS. Wikilinks that resolve to an
AMBIGUOUS key are logged as unresolved and dropped for Step 1 (they do not
produce an edge).

**Out of scope.** LLM-based arbitration. Cross-namespace collision handling
in federated graphs (each vault resolves independently).

## Section 5 — `ontology_type` Field

**Motivation.** Nodes currently carry a structural `kind` (note / tag /
pdf / …) but no semantic ontology (concept, decision, rule, etc.). Vault
design calls for ontology-based filtering (e.g., "show me all active
decisions about session_mode"). This requires a field on the node.

**Design.**

- Add `OntologyType` literal to `graph.py`:
  `concept | entity | process | tool | project | fact | decision | rule |
  procedure | relation | unclassified`.
- `Node` dataclass gets `ontology_type: OntologyType | None = None`.
- `ast_extractor.py` reads `frontmatter.type`. If the value is in the
  valid ontology set, it's assigned; otherwise `unclassified`.
- MCP tools `search_knowledge`, `list_communities`, `explore_community`
  accept an optional `ontology_type_filter` parameter that constrains
  results.

**Out of scope.** LLM-based classification of nodes lacking explicit
`type`. Type-aware update semantics (fact correction vs decision
supersede) — those are Jei's concern, not PrismRag's.

## Section 6 — `vault_ops` Write Path

**Motivation.** The `write_note` MCP tool currently lacks production-
quality safety: CAS conflict detection is partial, audit log is a stub,
writes are not atomic.

**Design.**

- **CAS conflict detection.** `write_with_cas(path, content,
  expected_hash)` compares the current SHA-256 of the file body (matching
  the `sha256:` prefixed format already used by `VaultDocument.content_hash`)
  to `expected_hash` before writing. Mismatch raises `CASConflict`
  carrying both hashes; caller must re-read and retry.
- **Atomic write.** Write to `path.tmp`, then `os.replace(tmp, path)`.
  POSIX guarantees atomicity. No partially-written files.
- **Audit log.** Every write appends one JSONL line to
  `data/audit.jsonl`: `{ts, op, path, old_hash, new_hash, caller, ok}`.
  Log rotates monthly by filename (`audit.jsonl`, `audit-2026-04.jsonl`);
  PrismRag does not delete old logs.
- **Path sandbox.** Existing `vault_ops/vault.py` checks are confirmed:
  path must be under `vault_root`; `.git/`, `.obsidian/`, `.trash/` are
  refused; symlink traversal is blocked.

**Out of scope.** Semantic merge of concurrent writes to disjoint
frontmatter fields. 3-way merge. Branching / propose-change workflow.
Undo and rollback (git handles this).

## Section 7 — CLI Integration Tests

**Motivation.** `test_cli.py` does not exist. The CLI commands — the
surface that agents and humans interact with — are the weakest-tested
part of the codebase.

**Design.**

- New file `tests/test_cli.py`. Covers:
  - `ingest`: empty vault, single-file vault, federated vault, incremental
    re-ingest. Verifies exit code and existence/shape of graph.json and
    GRAPH_REPORT.md.
  - `query`: results returned, token budget honored, JSON output stable.
  - `info`: stats correct, namespaces listed for federated setups.
  - `serve --transport stdio`: subprocess starts, responds to SIGTERM
    cleanly. (Protocol-level details remain in `mcp_server` unit tests.)
  - `version`: returns a version string.
- **Mock embedding backend.** To keep tests offline and deterministic:
  `embedder.py` gains a `MockEmbedder` that derives a 768-dim pseudo-vector
  from the SHA-256 of the node content (same content → same vector →
  reproducible similarity edges). Activated by
  `PRISM_GEMINI_API_KEY=TEST` or by calling `ingest` with a new
  `--no-embedding` flag (skips Pass 3 entirely).
- **Live API test.** One test marked `@pytest.mark.live_api` exercises
  the real Gemini API. Skipped by default in CI; runnable locally with
  `pytest -m live_api`.

**Out of scope.** Performance benchmarks. Tests at scale (>10k files).
Real-API tests running in CI.

## Ordering and dependencies

These sections are mostly independent. Recommended order (based on
unblock-value):

1. Section 3 (knowledge_id branch) — small, sets the structural foundation
   that Sections 4 and 5 build on.
2. Section 4 (collision resolution) — depends on 3 for `knowledge_id`
   signal.
3. Section 5 (ontology_type) — depends on 3 for frontmatter handling
   patterns.
4. Section 2 (Pass 2 PDF) — independent; can run in parallel.
5. Section 6 (vault_ops write path) — independent; small, self-contained.
6. Section 1 (`serve`) — depends on everything above being correct; it's
   the delivery surface.
7. Section 7 (CLI tests) — last, because it tests the whole stack.

## Acceptance criteria

Step 1 is considered complete when:

- [ ] `prism-rag serve --transport stdio` starts, responds to an MCP
      `list_tools` request, and exits cleanly on SIGTERM.
- [ ] A vault containing at least one PDF ingests without error, and the
      PDF appears as a `kind="pdf"` node in the resulting graph.
- [ ] A markdown file with `knowledge_id: KNOW-TEST` frontmatter ingests
      to a node whose id is `KNOW-TEST`, and `[[KNOW-TEST]]` wikilinks
      from other files resolve to it.
- [ ] A file with `embed: false` in frontmatter produces a node that has
      no outgoing `semantically_similar_to` edges.
- [ ] Two files with colliding alias "foo" — one with `knowledge_id`, one
      without — resolve to the knowledge-id owner; a log warning is
      emitted.
- [ ] A node with `type: decision` in frontmatter has
      `ontology_type == "decision"`; `search_knowledge` with
      `ontology_type_filter="decision"` excludes other types.
- [ ] Two concurrent `write_note` calls with the same `expected_hash`:
      the second one fails with `CASConflict`; `audit.jsonl` records both
      attempts.
- [ ] `pytest tests/test_cli.py` passes offline (no network), exercises
      all CLI subcommands, and runs in under 60 seconds.

## Estimated scope

| # | Section | Main code | Tests |
|---|---|---|---|
| 1 | `serve` startup | ~100 | ~40 |
| 2 | Pass 2 PDF | ~150 | ~60 |
| 3 | knowledge_id branch | ~100 | ~80 |
| 4 | Collision resolution | ~50 | ~40 |
| 5 | ontology_type | ~30 | ~30 |
| 6 | vault_ops write path | ~80 | ~50 |
| 7 | CLI integration tests | ~30 (mock) | ~250 |
| | **Total** | **~540** | **~550** |

Approximately 1,100 lines of change across roughly 12 files. Low-risk
work — each section is a bounded, locally-verifiable change.

## Future work (explicitly out of this step)

- Merge Obsidian MCP tools into PrismRag's MCP surface (Step 2).
- Wire PrismRag into ZenithLoom's knowledge_shelf / Jei blueprint
  (Step 3).
- LLM-inside-PrismRag: collision arbitration, ontology classification,
  atomization. Revisit after real usage data suggests Jei-side atomization
  is insufficient.
- Image and audio media extraction.
- Automatic `mentioned_in` reverse-index maintenance.
- Embedding model migration flow (versioned re-embed).
- Feature flag infrastructure (`[features]` in config).
- Status-based filtering in queries (superseded / invalidated).
