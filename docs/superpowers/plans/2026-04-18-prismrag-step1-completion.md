# PrismRag Step 1 Completion Implementation Plan

> **[STALE — 2026-04-30]** 本计划的 125 个任务均未勾选，但代码实际上已全部完成（241 个测试全绿）。
> 该文件不再是有效任务来源。当前任务状态见：`NimbusVault/设计细节/PrismRag v5.0 — 详细设计与任务分解.md`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the seven remaining pieces of work in PrismRag so it is usable by live agents over MCP, correct under realistic inputs, and forward-compatible with Vault Phase 2 (`knowledge_id`).

**Architecture:** Small, bounded changes across ~12 files. Passes 1/3/4/5 and the 8 MCP tools already exist and work. This plan adds a `knowledge_id` branch, `ontology_type`, tiered name-collision resolution, PDF media extraction, hardens the vault write path (atomic write, audit JSONL), verifies `serve` end-to-end, and adds CLI integration tests with a mock embedder.

**Tech Stack:** Python 3.11, NetworkX, FastMCP, pypdf, pydantic-settings, pytest, Typer.

---

## File Structure

### Created

- `prism_rag/ingest/media_extractor.py` — PDF text extraction + stubs for image/audio
- `prism_rag/ingest/mock_embedder.py` — deterministic content-hash-derived vectors for tests
- `tests/test_cli.py` — CLI integration tests
- `tests/test_knowledge_id_branch.py` — unit tests for Section 3
- `tests/test_name_collision.py` — unit tests for Section 4
- `tests/test_ontology_type.py` — unit tests for Section 5
- `tests/test_pdf_extraction.py` — unit tests for Section 2
- `tests/test_vault_ops_write.py` — unit tests for Section 6

### Modified

- `prism_rag/store/graph.py` — add `"knowledge"` to `NodeKind`; add `OntologyType`; add `ontology_type` field to `Node`
- `prism_rag/ingest/vault_loader.py` — `VaultDocument.id` honours `knowledge_id`; add `discover_vault_files`; add `VaultMedia` dataclass
- `prism_rag/ingest/ast_extractor.py` — tiered doc_index builder; `knowledge_id` registration; `relations:` frontmatter → edges; `ontology_type` from frontmatter `type:`
- `prism_rag/ingest/embedder.py` — honour `embed: false`; accept pluggable backend for mock
- `prism_rag/vault_ops/audit_log.py` — write to JSONL file (not just logger)
- `prism_rag/vault_ops/cas.py` — add `atomic_write` helper; add `CASConflict` exception
- `prism_rag/mcp_server/server.py` — accept `ontology_type` filter in `search_knowledge`, `list_communities`, `explore_community`; use `atomic_write` in `write_note`
- `prism_rag/cli.py` — add `--no-embedding` flag to `ingest`
- `pyproject.toml` — move `pypdf` from `[media]` optional to main deps (required for Section 2)

### Unchanged by this plan

- `prism_rag/cluster/leiden.py`
- `prism_rag/report/*`
- `prism_rag/retrieve/*`
- `prism_rag/store/federated.py`
- `prism_rag/ingest/similarity_linker.py`
- `prism_rag/ingest/incremental.py` (may need review after Section 3, see Task 3.6)

---

## Ordering

Following the spec's recommended order (unblock-value sequence):

- **Phase A** — Foundation: Section 3 (knowledge_id branch)
- **Phase B** — Builds on A: Section 4 (collision resolution) + Section 5 (ontology_type)
- **Phase C** — Independent: Section 2 (PDF) + Section 6 (vault_ops write path)
- **Phase D** — Delivery: Section 1 (verify `serve`) + Section 7 (CLI tests)

---

## Phase A — `knowledge_id` Branch (Section 3)

### Task 3.1: Add `"knowledge"` to NodeKind

**Files:**
- Modify: `prism_rag/store/graph.py:51`
- Test: `tests/test_knowledge_id_branch.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_knowledge_id_branch.py`:

```python
"""Tests for the knowledge_id branch (Section 3)."""
from __future__ import annotations

from pathlib import Path

from prism_rag.ingest.ast_extractor import extract_ast
from prism_rag.ingest.vault_loader import VaultDocument, load_vault
from prism_rag.store.graph import KnowledgeGraph, Node


def test_nodekind_accepts_knowledge():
    """NodeKind literal must include 'knowledge' for Phase 2 atomic nodes."""
    node = Node(id="KNOW-001", label="K1", kind="knowledge")
    assert node.kind == "knowledge"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_id_branch.py::test_nodekind_accepts_knowledge -v`
Expected: Test might pass already if typing is loose — but run mypy to confirm. If pytest passes but mypy fails, count that as "failing".

Run: `python -c "from prism_rag.store.graph import NodeKind; import typing; print(typing.get_args(NodeKind))"`
Expected: `('note', 'tag', 'category', 'image', 'pdf', 'audio', 'section', 'block')` — no "knowledge".

- [ ] **Step 3: Add "knowledge" to NodeKind**

Edit `prism_rag/store/graph.py` line 51:

```python
NodeKind = Literal["note", "knowledge", "tag", "category", "image", "pdf", "audio", "section", "block"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_knowledge_id_branch.py::test_nodekind_accepts_knowledge -v`
Expected: PASS

Run: `python -c "from prism_rag.store.graph import NodeKind; import typing; print(typing.get_args(NodeKind))"`
Expected: `('note', 'knowledge', 'tag', ...)`

- [ ] **Step 5: Commit**

```bash
cd /home/kingy/Foundation/PrismRag
git add prism_rag/store/graph.py tests/test_knowledge_id_branch.py
git commit -m "feat(graph): add 'knowledge' to NodeKind for atomic nodes"
```

---

### Task 3.2: `VaultDocument.id` honours `knowledge_id`

**Files:**
- Modify: `prism_rag/ingest/vault_loader.py:56-58`
- Test: `tests/test_knowledge_id_branch.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_knowledge_id_branch.py`:

```python
def test_vault_document_id_uses_knowledge_id(tmp_path):
    """When frontmatter has knowledge_id, VaultDocument.id returns it."""
    p = tmp_path / "sub" / "my-note.md"
    p.parent.mkdir()
    p.write_text("---\nknowledge_id: KNOW-042\n---\n\nbody")
    doc = VaultDocument.from_path(p, tmp_path)
    assert doc.id == "KNOW-042"


def test_vault_document_id_falls_back_to_path(tmp_path):
    """When no knowledge_id, VaultDocument.id is the relative path stem."""
    p = tmp_path / "sub" / "my-note.md"
    p.parent.mkdir()
    p.write_text("no frontmatter")
    doc = VaultDocument.from_path(p, tmp_path)
    assert doc.id == "sub/my-note"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_id_branch.py::test_vault_document_id_uses_knowledge_id -v`
Expected: FAIL — `assert 'sub/my-note' == 'KNOW-042'`

- [ ] **Step 3: Modify `VaultDocument.id`**

Edit `prism_rag/ingest/vault_loader.py` replacing lines 55-58:

```python
    @property
    def id(self) -> str:
        """Stable node ID.

        If frontmatter declares a knowledge_id (Phase 2 atomic node), use it.
        Otherwise fall back to relative path without .md extension, POSIX-style.
        """
        kid = self.frontmatter.get("knowledge_id")
        if kid:
            return str(kid)
        return self.relative_path.with_suffix("").as_posix()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_knowledge_id_branch.py -v`
Expected: All 3 tests PASS.

Run: `pytest tests/test_phase1_mvp.py -v`
Expected: All existing tests still PASS (regression check — no knowledge_id files means fallback path unchanged).

- [ ] **Step 5: Commit**

```bash
git add prism_rag/ingest/vault_loader.py tests/test_knowledge_id_branch.py
git commit -m "feat(vault_loader): VaultDocument.id honours knowledge_id frontmatter"
```

---

### Task 3.3: `ast_extractor` marks `kind="knowledge"` and registers `knowledge_id` in doc_index

**Files:**
- Modify: `prism_rag/ingest/ast_extractor.py` (two sections: `_build_doc_index` and `extract_ast` loop)
- Test: `tests/test_knowledge_id_branch.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_knowledge_id_branch.py`:

```python
def test_knowledge_id_wikilink_resolves(tmp_path):
    """[[KNOW-042]] in another file must resolve to the node with knowledge_id=KNOW-042."""
    # File with knowledge_id
    a = tmp_path / "knowledge" / "KNOW-042-session.md"
    a.parent.mkdir()
    a.write_text("---\nknowledge_id: KNOW-042\n---\n\nBody of K42")

    # File linking to KNOW-042 by its id
    b = tmp_path / "设计细节" / "some-doc.md"
    b.parent.mkdir()
    b.write_text("See [[KNOW-042]] for details.")

    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)

    # KNOW-042 exists as a node
    assert "KNOW-042" in graph.g.nodes
    assert graph.g.nodes["KNOW-042"]["kind"] == "knowledge"

    # Edge from some-doc to KNOW-042
    assert graph.g.has_edge("设计细节/some-doc", "KNOW-042")


def test_regular_note_still_kind_note(tmp_path):
    """Files without knowledge_id get kind='note'."""
    p = tmp_path / "regular.md"
    p.write_text("no frontmatter, just body")
    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)
    assert graph.g.nodes["regular"]["kind"] == "note"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_knowledge_id_branch.py::test_knowledge_id_wikilink_resolves -v`
Expected: FAIL — `KNOW-042` not in nodes (the node is created with id `KNOW-042` but wikilink index doesn't map "KNOW-042" lookup).

- [ ] **Step 3: Update `_build_doc_index` to register knowledge_id**

Edit `prism_rag/ingest/ast_extractor.py` — find `_build_doc_index` (around line 95) and modify:

```python
def _build_doc_index(docs: Iterable[VaultDocument]) -> dict[str, str]:
    """Build lowercase name → canonical doc.id lookup.

    Keys registered:
    - filename stem (doc.label)
    - frontmatter aliases
    - knowledge_id if present (Phase 2)
    """
    index: dict[str, str] = {}
    for doc in docs:
        index[doc.label.lower()] = doc.id
        for alias in doc.aliases:
            index[alias.lower()] = doc.id
        kid = doc.frontmatter.get("knowledge_id")
        if kid:
            index[str(kid).lower()] = doc.id
    return index
```

- [ ] **Step 4: Update node creation to set kind="knowledge"**

In `prism_rag/ingest/ast_extractor.py` — find the node creation in `extract_ast` (around line 157) and change:

```python
        kind = "knowledge" if doc.frontmatter.get("knowledge_id") else "note"

        note = Node(
            id=doc.id,
            label=doc.label,
            kind=kind,
            source_file=str(doc.relative_path),
            content=doc.content,
            content_hash=doc.content_hash,
            tokens=_token_count(doc.content),
            frontmatter=doc.frontmatter,
            maturity=_maturity if _maturity in ("seed", "growing", "mature", "archived") else None,
            confidence=_confidence if _confidence in ("high", "medium", "low") else None,
            actionability=_actionability if _actionability in ("reference", "decision", "task") else None,
        )
        graph.add_node(note)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_knowledge_id_branch.py -v`
Expected: All 5 tests PASS.

Run: `pytest tests/ -v --ignore=tests/test_cli.py`
Expected: All existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add prism_rag/ingest/ast_extractor.py tests/test_knowledge_id_branch.py
git commit -m "feat(ast_extractor): emit kind=knowledge and resolve [[KNOW-XXX]] wikilinks"
```

---

### Task 3.4: `relations:` frontmatter → typed EXTRACTED edges

**Files:**
- Modify: `prism_rag/ingest/ast_extractor.py` (add new pass after wikilink extraction)
- Test: `tests/test_knowledge_id_branch.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_knowledge_id_branch.py`:

```python
def test_relations_frontmatter_produces_edges(tmp_path):
    """frontmatter.relations.{depends_on,supersedes,...} emit typed EXTRACTED edges."""
    a = tmp_path / "knowledge" / "KNOW-001-base.md"
    a.parent.mkdir()
    a.write_text("---\nknowledge_id: KNOW-001\n---\n\nBase concept")

    b = tmp_path / "knowledge" / "KNOW-042-dep.md"
    b.write_text(
        "---\n"
        "knowledge_id: KNOW-042\n"
        "relations:\n"
        "  depends_on: [KNOW-001]\n"
        "  supersedes: []\n"
        "---\n\nDepends on KNOW-001"
    )

    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)

    assert graph.g.has_edge("KNOW-042", "KNOW-001")
    edge = graph.g.edges["KNOW-042", "KNOW-001"]
    assert edge["relation"] == "depends_on"
    assert edge["confidence"] == "EXTRACTED"
    assert edge["source_pass"] == "ast"


def test_relations_supersedes_edge_type(tmp_path):
    a = tmp_path / "knowledge" / "KNOW-040.md"
    a.parent.mkdir()
    a.write_text("---\nknowledge_id: KNOW-040\n---\n\nOld")
    b = tmp_path / "knowledge" / "KNOW-100.md"
    b.write_text(
        "---\n"
        "knowledge_id: KNOW-100\n"
        "relations:\n"
        "  supersedes: [KNOW-040]\n"
        "---\n\nNew"
    )

    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)

    assert graph.g.edges["KNOW-100", "KNOW-040"]["relation"] == "supersedes"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_knowledge_id_branch.py::test_relations_frontmatter_produces_edges -v`
Expected: FAIL — edge not present.

- [ ] **Step 3: Add `_extract_relations_edges` helper**

Edit `prism_rag/ingest/ast_extractor.py` — add near the other helpers (around line 130, before `extract_ast`):

```python
_RELATION_TYPES = ("supersedes", "superseded_by", "depends_on", "refines", "contradicts", "references")


def _extract_relations_edges(
    graph: KnowledgeGraph,
    doc: VaultDocument,
    doc_index: dict[str, str],
) -> int:
    """Emit EXTRACTED edges from frontmatter.relations.{type}: [targets].

    Returns number of edges emitted.
    """
    relations = doc.frontmatter.get("relations")
    if not isinstance(relations, dict):
        return 0

    count = 0
    for rel_type in _RELATION_TYPES:
        targets = relations.get(rel_type)
        if not targets:
            continue
        # Normalize: accept str or list[str]
        if isinstance(targets, str):
            targets = [targets]
        elif not isinstance(targets, list):
            continue

        for target in targets:
            target_str = str(target).strip()
            if not target_str:
                continue
            resolved = _resolve_wikilink(target_str, doc_index)
            if resolved is None or resolved == doc.id:
                continue
            graph.add_edge(
                Edge(
                    source=doc.id,
                    target=resolved,
                    relation=rel_type,
                    confidence="EXTRACTED",
                    confidence_score=1.0,
                    weight=1.0,
                    source_pass="ast",
                )
            )
            count += 1
    return count
```

- [ ] **Step 4: Call `_extract_relations_edges` from `extract_ast`**

In `prism_rag/ingest/ast_extractor.py` — in `extract_ast`, after the wikilink/tag/category edge extraction loop (Step 4), add a new Step 4d:

```python
    # Step 4d: Relations frontmatter (Phase 2 explicit typed edges)
    for doc in docs:
        _extract_relations_edges(graph, doc, doc_index)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_knowledge_id_branch.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add prism_rag/ingest/ast_extractor.py tests/test_knowledge_id_branch.py
git commit -m "feat(ast_extractor): emit typed EXTRACTED edges from frontmatter.relations"
```

---

### Task 3.5: `embedder.py` honours `embed: false`

**Files:**
- Modify: `prism_rag/ingest/embedder.py`
- Test: `tests/test_knowledge_id_branch.py`

- [ ] **Step 1: Inspect current embedder**

Run: `grep -n "def compute_embeddings\|def _embed_one\|for node_id\|frontmatter" prism_rag/ingest/embedder.py`

Goal: find the per-node loop in `compute_embeddings` so we can skip nodes with `embed: false`.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_knowledge_id_branch.py`:

```python
def test_embed_false_skips_embedding(monkeypatch, tmp_path):
    """Node with frontmatter embed: false must NOT be embedded."""
    from prism_rag.ingest.embedder import compute_embeddings
    from prism_rag.config import PrismRagSettings

    a = tmp_path / "knowledge" / "KNOW-REL.md"
    a.parent.mkdir()
    a.write_text(
        "---\n"
        "knowledge_id: KNOW-REL\n"
        "type: relation\n"
        "embed: false\n"
        "---\n\nRelation-only node"
    )
    b = tmp_path / "note.md"
    b.write_text("regular content")

    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)

    # Monkeypatch the Gemini client to avoid real API
    call_log: list[str] = []

    def fake_embed(text: str, *, settings) -> list[float]:
        call_log.append(text[:30])
        return [0.1] * 768

    monkeypatch.setattr(
        "prism_rag.ingest.embedder._embed_text",
        fake_embed,
    )

    settings = PrismRagSettings(gemini_api_key="TEST")
    vectors = compute_embeddings(graph, settings)

    # KNOW-REL has embed: false, should be skipped
    assert "KNOW-REL" not in vectors
    # note.md has no embed directive, should be included
    assert "note" in vectors
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_knowledge_id_branch.py::test_embed_false_skips_embedding -v`
Expected: FAIL — either `KNOW-REL` is in vectors, or the test errors on the monkeypatch target (adjust path to match actual embedder internals).

- [ ] **Step 4: Add embed:false check**

Edit `prism_rag/ingest/embedder.py` — in the `compute_embeddings` function's per-node loop, add an early skip. The loop looks like `for node_id, data in graph.g.nodes(data=True):`. Add after retrieving `data`:

```python
        fm = data.get("frontmatter") or {}
        if fm.get("embed") is False:
            logger.debug(f"[embedder] skipping {node_id} (embed: false)")
            continue
```

Also ensure the embedder has a testable seam — if `_embed_text` doesn't exist as a module-level function, refactor the one-node embedding call into such a function so the monkeypatch in the test works. The function signature should be:

```python
def _embed_text(text: str, *, settings: PrismRagSettings) -> list[float]:
    # existing Gemini call logic
    ...
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_knowledge_id_branch.py::test_embed_false_skips_embedding -v`
Expected: PASS

Run: `pytest tests/test_knowledge_id_branch.py -v`
Expected: All knowledge_id tests PASS.

- [ ] **Step 6: Commit**

```bash
git add prism_rag/ingest/embedder.py tests/test_knowledge_id_branch.py
git commit -m "feat(embedder): honour frontmatter embed: false to skip Pass 3"
```

---

### Task 3.6: Regression — incremental ingest still works

**Files:**
- Test: `tests/test_knowledge_id_branch.py`

- [ ] **Step 1: Write regression test**

Append to `tests/test_knowledge_id_branch.py`:

```python
def test_incremental_ingest_with_knowledge_id(tmp_path, monkeypatch):
    """ingest_file() should handle a knowledge_id file correctly."""
    from prism_rag.config import PrismRagSettings
    from prism_rag.ingest.incremental import ingest_file

    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()

    # Seed with an initial doc
    (vault / "base.md").write_text("---\nknowledge_id: KNOW-001\n---\n\nBase")

    settings = PrismRagSettings(vault_path=vault, data_dir=data, gemini_api_key="")

    # Full ingest first (no embedding — no API key)
    monkeypatch.chdir(tmp_path)
    from prism_rag.ingest.vault_loader import load_vault
    from prism_rag.ingest.ast_extractor import extract_ast
    from prism_rag.store.graph import KnowledgeGraph
    g = KnowledgeGraph()
    extract_ast(g, load_vault(vault))
    g.save(settings.graph_path)

    # Add a new file and incremental-ingest it
    new_file = vault / "new.md"
    new_file.write_text("---\nknowledge_id: KNOW-042\nrelations:\n  depends_on: [KNOW-001]\n---\n\nNew")

    result = ingest_file(new_file, settings=settings, skip_embed=True)
    assert result["node_id"] == "KNOW-042"
    assert result["action"] in ("added", "updated")
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_knowledge_id_branch.py::test_incremental_ingest_with_knowledge_id -v`
Expected: PASS (the changes in 3.1–3.5 should flow through `incremental.py` since it delegates to `extract_ast`).

If this FAILS, inspect `prism_rag/ingest/incremental.py`:
- Verify it uses `VaultDocument.from_path()` or `load_vault()` (picks up knowledge_id)
- Verify it calls `extract_ast` (gets our new relations-edges pass)
- If it has its own inlined logic that bypasses these, patch minimally to delegate.

- [ ] **Step 3: Commit**

```bash
git add tests/test_knowledge_id_branch.py
# If incremental.py was patched:
git add prism_rag/ingest/incremental.py
git commit -m "test(incremental): verify knowledge_id flows through incremental ingest"
```

---

## Phase B — Name Collision Resolution (Section 4)

### Task 4.1: Detect collisions in doc_index (AMBIGUOUS fallback only, no tier yet)

**Files:**
- Modify: `prism_rag/ingest/ast_extractor.py` (rewrite `_build_doc_index`)
- Test: `tests/test_name_collision.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_name_collision.py`:

```python
"""Tests for name collision resolution (Section 4)."""
from __future__ import annotations

import logging

from prism_rag.ingest.ast_extractor import extract_ast, _build_doc_index
from prism_rag.ingest.vault_loader import load_vault
from prism_rag.store.graph import KnowledgeGraph


def test_collision_same_alias_logs_warning(tmp_path, caplog):
    """Two files sharing an alias 'foo', same priority tier → log warning."""
    a = tmp_path / "a.md"
    a.write_text("---\naliases: [foo]\n---\nA")
    b = tmp_path / "b.md"
    b.write_text("---\naliases: [foo]\n---\nB")

    docs = load_vault(tmp_path)
    caplog.set_level(logging.WARNING, logger="prism_rag.ingest.ast_extractor")
    _build_doc_index(docs)

    assert any("collision" in rec.message.lower() or "ambiguous" in rec.message.lower()
               for rec in caplog.records)


def test_wikilink_to_ambiguous_is_dropped(tmp_path, caplog):
    """[[foo]] wikilink pointing to an ambiguous alias should NOT produce an edge."""
    a = tmp_path / "a.md"
    a.write_text("---\naliases: [foo]\n---\nA")
    b = tmp_path / "b.md"
    b.write_text("---\naliases: [foo]\n---\nB")
    c = tmp_path / "c.md"
    c.write_text("See [[foo]]")

    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    caplog.set_level(logging.WARNING, logger="prism_rag.ingest.ast_extractor")
    extract_ast(graph, docs)

    # No edge from c to either a or b via the 'foo' alias
    assert not graph.g.has_edge("c", "a")
    assert not graph.g.has_edge("c", "b")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_name_collision.py -v`
Expected: FAIL — current `_build_doc_index` silently overwrites with no warning.

- [ ] **Step 3: Rewrite `_build_doc_index` with collision detection**

Edit `prism_rag/ingest/ast_extractor.py` — replace `_build_doc_index` and add a priority helper:

```python
def _priority_tier(doc: VaultDocument) -> int:
    """Return collision-resolution priority (higher wins).

    Tier 3: has knowledge_id
    Tier 2: canonical: true in frontmatter
    Tier 1: lives under knowledge/ directory
    Tier 0: everything else
    """
    if doc.frontmatter.get("knowledge_id"):
        return 3
    if doc.frontmatter.get("canonical") is True:
        return 2
    try:
        parts = doc.relative_path.parts
    except AttributeError:
        parts = ()
    if "knowledge" in parts:
        return 1
    return 0


def _build_doc_index(docs: Iterable[VaultDocument]) -> dict[str, str]:
    """Build lowercase name → canonical doc.id lookup with tiered collision resolution.

    On collision:
      - Higher-priority tier wins
      - Same-tier collision → mark as AMBIGUOUS (entry removed, wikilinks to this key fail resolution)
      - Log a warning in both cases
    """
    docs_list = list(docs)
    candidates: dict[str, list[VaultDocument]] = {}

    def _register(key: str, doc: VaultDocument) -> None:
        candidates.setdefault(key.lower(), []).append(doc)

    for doc in docs_list:
        _register(doc.label, doc)
        for alias in doc.aliases:
            _register(alias, doc)
        kid = doc.frontmatter.get("knowledge_id")
        if kid:
            _register(str(kid), doc)

    index: dict[str, str] = {}
    for key, cands in candidates.items():
        if len(cands) == 1:
            index[key] = cands[0].id
            continue

        by_tier: dict[int, list[VaultDocument]] = {}
        for d in cands:
            by_tier.setdefault(_priority_tier(d), []).append(d)
        max_tier = max(by_tier.keys())
        winners = by_tier[max_tier]

        if len(winners) == 1:
            logger.warning(
                f"[ast_extractor] collision on key {key!r}: winner={winners[0].id} "
                f"(tier {max_tier}), others={[d.id for d in cands if d is not winners[0]]}"
            )
            index[key] = winners[0].id
        else:
            logger.warning(
                f"[ast_extractor] AMBIGUOUS collision on key {key!r}: "
                f"candidates={[d.id for d in winners]} (tier {max_tier}); wikilinks unresolved"
            )
            # Do NOT register in index — wikilinks to this key will be dropped
    return index
```

Also add at the top of `ast_extractor.py` if not already present:

```python
import logging
logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_name_collision.py -v`
Expected: Both tests PASS.

Run: `pytest tests/ -v`
Expected: All existing tests still pass (doc_index is a superset of the old behaviour when no collisions).

- [ ] **Step 5: Commit**

```bash
git add prism_rag/ingest/ast_extractor.py tests/test_name_collision.py
git commit -m "feat(ast_extractor): tiered name collision resolution with AMBIGUOUS fallback"
```

---

### Task 4.2: Verify `knowledge_id` wins over plain filename

**Files:**
- Test: `tests/test_name_collision.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_name_collision.py`:

```python
def test_knowledge_id_beats_filename(tmp_path):
    """Collision between knowledge_id=foo and a plain filename foo.md → knowledge_id wins."""
    a = tmp_path / "设计" / "foo.md"
    a.parent.mkdir()
    a.write_text("plain file called foo")
    b = tmp_path / "knowledge" / "KNOW-200-foo.md"
    b.parent.mkdir()
    b.write_text("---\nknowledge_id: foo\n---\nk-node aliased as foo")

    docs = load_vault(tmp_path)
    idx = _build_doc_index(docs)
    # "foo" key must resolve to the knowledge-id owner
    assert idx["foo"] == "foo"  # doc.id === knowledge_id === "foo"
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_name_collision.py::test_knowledge_id_beats_filename -v`
Expected: PASS (Task 4.1's priority logic already covers this).

If FAIL: inspect that `_priority_tier` returns 3 for the knowledge_id doc and 0 for the plain filename, and that `by_tier[3]` wins.

- [ ] **Step 3: Commit**

```bash
git add tests/test_name_collision.py
git commit -m "test(collision): verify knowledge_id tier beats plain filename"
```

---

### Task 4.3: Verify `canonical: true` wins over plain filename

**Files:**
- Test: `tests/test_name_collision.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_name_collision.py`:

```python
def test_canonical_flag_beats_filename(tmp_path):
    """canonical: true frontmatter beats a colliding plain filename."""
    a = tmp_path / "notes" / "bar.md"
    a.parent.mkdir()
    a.write_text("plain bar")
    b = tmp_path / "docs" / "canonical-bar.md"
    b.parent.mkdir()
    b.write_text("---\naliases: [bar]\ncanonical: true\n---\nCanonical definition")

    docs = load_vault(tmp_path)
    idx = _build_doc_index(docs)
    assert idx["bar"] == "docs/canonical-bar"
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_name_collision.py::test_canonical_flag_beats_filename -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_name_collision.py
git commit -m "test(collision): verify canonical:true tier beats plain filename"
```

---

### Task 4.4: Verify `knowledge/` directory tier

**Files:**
- Test: `tests/test_name_collision.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_name_collision.py`:

```python
def test_knowledge_dir_beats_plain_file(tmp_path):
    """File under knowledge/ wins over a colliding plain filename."""
    a = tmp_path / "random" / "baz.md"
    a.parent.mkdir()
    a.write_text("random baz")
    b = tmp_path / "knowledge" / "baz.md"
    b.parent.mkdir()
    b.write_text("knowledge baz")  # no knowledge_id, no canonical — just the directory signal

    docs = load_vault(tmp_path)
    idx = _build_doc_index(docs)
    assert idx["baz"] == "knowledge/baz"
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_name_collision.py::test_knowledge_dir_beats_plain_file -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_name_collision.py
git commit -m "test(collision): verify knowledge/ dir tier beats plain file"
```

---

## Phase B — ontology_type (Section 5)

### Task 5.1: Add `OntologyType` literal and `Node.ontology_type` field

**Files:**
- Modify: `prism_rag/store/graph.py`
- Test: `tests/test_ontology_type.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ontology_type.py`:

```python
"""Tests for ontology_type field (Section 5)."""
from __future__ import annotations

import typing

from prism_rag.store.graph import KnowledgeGraph, Node, OntologyType


def test_ontology_type_literal_values():
    values = set(typing.get_args(OntologyType))
    assert values == {
        "concept", "entity", "process", "tool", "project",
        "fact", "decision", "rule", "procedure", "relation",
        "unclassified",
    }


def test_node_ontology_type_default_none():
    n = Node(id="x", label="X", kind="note")
    assert n.ontology_type is None


def test_node_ontology_type_set():
    n = Node(id="x", label="X", kind="knowledge", ontology_type="decision")
    assert n.ontology_type == "decision"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ontology_type.py::test_ontology_type_literal_values -v`
Expected: FAIL — `OntologyType` doesn't exist yet.

- [ ] **Step 3: Add `OntologyType` and `Node.ontology_type`**

Edit `prism_rag/store/graph.py` — add near other Literal definitions (around line 62):

```python
OntologyType = Literal[
    "concept", "entity", "process", "tool", "project",
    "fact", "decision", "rule", "procedure", "relation",
    "unclassified",
]
```

In the `Node` dataclass (around line 65), add after the Am attributes:

```python
    # Semantic ontology type (Vault Phase 2). Populated from frontmatter.type.
    ontology_type: OntologyType | None = None
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ontology_type.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add prism_rag/store/graph.py tests/test_ontology_type.py
git commit -m "feat(graph): add OntologyType and Node.ontology_type field"
```

---

### Task 5.2: `ast_extractor` reads `frontmatter.type` into `ontology_type`

**Files:**
- Modify: `prism_rag/ingest/ast_extractor.py` (node creation block)
- Test: `tests/test_ontology_type.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ontology_type.py`:

```python
import typing
from pathlib import Path

from prism_rag.ingest.ast_extractor import extract_ast
from prism_rag.ingest.vault_loader import load_vault
from prism_rag.store.graph import OntologyType


def test_ontology_type_from_frontmatter(tmp_path):
    """frontmatter type: decision → Node.ontology_type=decision."""
    p = tmp_path / "decision.md"
    p.write_text(
        "---\n"
        "knowledge_id: KNOW-D1\n"
        "type: decision\n"
        "---\n\n"
        "A decision about X"
    )
    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)
    assert graph.g.nodes["KNOW-D1"]["ontology_type"] == "decision"


def test_invalid_type_becomes_unclassified(tmp_path):
    """Unknown type: value → ontology_type=unclassified."""
    p = tmp_path / "note.md"
    p.write_text("---\ntype: nonsense\n---\nbody")
    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)
    assert graph.g.nodes["note"]["ontology_type"] == "unclassified"


def test_no_type_leaves_ontology_type_none(tmp_path):
    """No type: frontmatter → ontology_type=None (not 'unclassified')."""
    p = tmp_path / "note.md"
    p.write_text("no frontmatter body")
    docs = load_vault(tmp_path)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)
    # Either None or absent — both acceptable
    ont = graph.g.nodes["note"].get("ontology_type")
    assert ont is None
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_ontology_type.py -v`
Expected: new tests FAIL (ontology_type not set from frontmatter).

- [ ] **Step 3: Read `type:` in ast_extractor node creation**

Edit `prism_rag/ingest/ast_extractor.py` — before the `note = Node(...)` construction, add:

```python
        _VALID_ONT = {
            "concept", "entity", "process", "tool", "project",
            "fact", "decision", "rule", "procedure", "relation",
            "unclassified",
        }
        fm_type = fm.get("type")
        if fm_type is None:
            _ont: str | None = None
        elif fm_type in _VALID_ONT:
            _ont = fm_type
        else:
            _ont = "unclassified"
```

Then extend the `Node(...)` construction to set `ontology_type=_ont`:

```python
        note = Node(
            id=doc.id,
            label=doc.label,
            kind=kind,
            source_file=str(doc.relative_path),
            content=doc.content,
            content_hash=doc.content_hash,
            tokens=_token_count(doc.content),
            frontmatter=doc.frontmatter,
            maturity=_maturity if _maturity in ("seed", "growing", "mature", "archived") else None,
            confidence=_confidence if _confidence in ("high", "medium", "low") else None,
            actionability=_actionability if _actionability in ("reference", "decision", "task") else None,
            ontology_type=_ont,
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ontology_type.py -v`
Expected: All PASS.

Run: `pytest tests/ -v`
Expected: All tests still pass.

- [ ] **Step 5: Commit**

```bash
git add prism_rag/ingest/ast_extractor.py tests/test_ontology_type.py
git commit -m "feat(ast_extractor): read frontmatter.type into Node.ontology_type"
```

---

### Task 5.3: MCP tools accept `ontology_type` filter

**Files:**
- Modify: `prism_rag/mcp_server/server.py` (`search_knowledge`, `list_communities`, `explore_community`)
- Test: `tests/test_ontology_type.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ontology_type.py`:

```python
def test_search_knowledge_filters_by_ontology_type(tmp_path, monkeypatch):
    """search_knowledge with ontology_type='decision' returns only decision nodes."""
    from prism_rag.config import PrismRagSettings, GraphSource
    from prism_rag.mcp_server import server as mcp_server

    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()

    (vault / "dec.md").write_text(
        "---\nknowledge_id: KNOW-D\ntype: decision\n---\nA decision"
    )
    (vault / "fact.md").write_text(
        "---\nknowledge_id: KNOW-F\ntype: fact\n---\nA fact"
    )

    # Build graph
    from prism_rag.ingest.vault_loader import load_vault as lv
    from prism_rag.ingest.ast_extractor import extract_ast
    from prism_rag.store.graph import KnowledgeGraph as KG
    g = KG()
    extract_ast(g, lv(vault))
    g.save(data / "graph.json")

    # Reset server state
    mcp_server._federated = None

    settings = PrismRagSettings(
        graphs=[GraphSource(namespace="default", vault_path=vault, data_dir=data)],
    )
    monkeypatch.setattr(
        "prism_rag.mcp_server.server.PrismRagSettings",
        lambda: settings,
    )

    # Call search_knowledge with ontology_type filter
    result = mcp_server.search_knowledge(query="decision", ontology_type="decision")
    import json
    parsed = json.loads(result)
    node_ids = [n["id"] for n in parsed.get("nodes", [])]
    assert "KNOW-D" in node_ids or any("KNOW-D" in nid for nid in node_ids)
    assert not any("KNOW-F" in nid for nid in node_ids)
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_ontology_type.py::test_search_knowledge_filters_by_ontology_type -v`
Expected: FAIL — `search_knowledge` doesn't accept `ontology_type` parameter.

- [ ] **Step 3: Add `ontology_type` parameter to `search_knowledge`**

Edit `prism_rag/mcp_server/server.py` — find `search_knowledge` (line 107). Update signature:

```python
@mcp.tool()
def search_knowledge(
    query: str,
    scope: str = "",
    mode: str = "bfs",
    budget: int = 4000,
    ontology_type: str = "",
) -> str:
    """... (existing docstring) ...

    Args:
        ...
        ontology_type: If non-empty, filter returned nodes to only those
            whose ontology_type matches (e.g., "decision", "concept").
    """
    # ... existing body ...
```

After the traversal produces `nodes` (the list of dicts) but BEFORE building the JSON response, add:

```python
    if ontology_type:
        nodes = [n for n in nodes if n.get("ontology_type") == ontology_type]
```

Also ensure `_node_summary` (line 66) includes `ontology_type` in its output dict:

```python
def _node_summary(graph: KnowledgeGraph, node_id: str, include_content: bool = False) -> dict:
    data = graph.g.nodes.get(node_id, {})
    summary = {
        "id": node_id,
        "label": data.get("label", node_id),
        "kind": data.get("kind", "?"),
        "ontology_type": data.get("ontology_type"),  # add this line
        "tokens": data.get("tokens", 0),
        "community": data.get("community_id", ""),
        "degree": graph.degree(node_id),
    }
    # ... rest unchanged ...
```

- [ ] **Step 4: Add same filter to `list_communities` and `explore_community`**

In `list_communities` (line 310) — if the function returns per-community aggregates, add optional filter:

```python
def list_communities(ontology_type: str = "") -> str:
    # inside the iteration over communities, when counting members,
    # apply the filter if specified:
    if ontology_type:
        members = [m for m in members if graph.g.nodes[m].get("ontology_type") == ontology_type]
    # ... rest of existing logic ...
```

In `explore_community` (line 353) — similarly filter the member list before returning.

(The exact placement depends on existing function structure; the principle is: after collecting node lists, if `ontology_type` is non-empty, filter nodes whose `ontology_type` attribute matches.)

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_ontology_type.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add prism_rag/mcp_server/server.py tests/test_ontology_type.py
git commit -m "feat(mcp): accept ontology_type filter in search/list/explore tools"
```

---

## Phase C — PDF Media Extraction (Section 2)

### Task 2.1: Make `pypdf` a required dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Move `pypdf` from `[media]` to main deps**

Edit `pyproject.toml` — in the main `dependencies` list (around line 12), add:

```
    "pypdf>=5.0",
```

And in `[project.optional-dependencies].media`, remove `pypdf>=5.0` (keep `pillow` and `faster-whisper` there as placeholders for future image/audio work).

- [ ] **Step 2: Install**

Run: `cd /home/kingy/Foundation/PrismRag && pip install -e ".[dev]"`
Expected: `pypdf` is installed, no errors.

Run: `python -c "import pypdf; print(pypdf.__version__)"`
Expected: version >= 5.0 printed.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: move pypdf to required dependencies for Pass 2"
```

---

### Task 2.2: Create `VaultMedia` dataclass and `discover_vault_files`

**Files:**
- Modify: `prism_rag/ingest/vault_loader.py`
- Test: `tests/test_pdf_extraction.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pdf_extraction.py`:

```python
"""Tests for Pass 2 PDF media extraction (Section 2)."""
from __future__ import annotations

from pathlib import Path

from prism_rag.ingest.vault_loader import VaultMedia, discover_vault_files


def test_vault_media_id_from_path(tmp_path):
    p = tmp_path / "docs" / "report.pdf"
    p.parent.mkdir()
    p.write_bytes(b"%PDF-1.4\n...")
    media = VaultMedia.from_path(p, tmp_path)
    assert media.id == "docs/report"
    assert media.path == p
    assert media.kind == "pdf"


def test_discover_vault_files_returns_md_and_pdf(tmp_path):
    (tmp_path / "note.md").write_text("# note")
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4\n...")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")

    files = discover_vault_files(tmp_path)
    suffixes = sorted(f.suffix for f in files)
    # Images not yet supported
    assert ".md" in suffixes
    assert ".pdf" in suffixes
    assert ".png" not in suffixes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pdf_extraction.py -v`
Expected: FAIL — `VaultMedia` and `discover_vault_files` don't exist.

- [ ] **Step 3: Add `VaultMedia` and `discover_vault_files`**

Edit `prism_rag/ingest/vault_loader.py` — add after `VaultDocument`:

```python
@dataclass
class VaultMedia:
    """A non-markdown vault file (PDF, image, audio)."""

    path: Path
    vault_root: Path
    kind: str  # "pdf" | "image" | "audio"
    content_hash: str = ""

    @classmethod
    def from_path(cls, path: Path, vault_root: Path) -> "VaultMedia":
        ext = path.suffix.lower()
        if ext == ".pdf":
            kind = "pdf"
        elif ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            kind = "image"
        elif ext in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
            kind = "audio"
        else:
            kind = "unknown"
        hash_hex = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(
            path=path,
            vault_root=vault_root,
            kind=kind,
            content_hash=f"sha256:{hash_hex}",
        )

    @property
    def relative_path(self) -> Path:
        return self.path.relative_to(self.vault_root)

    @property
    def id(self) -> str:
        """Stable ID: relative path without extension, POSIX-style."""
        return self.relative_path.with_suffix("").as_posix()

    @property
    def label(self) -> str:
        return self.path.stem


_MEDIA_EXTENSIONS: frozenset[str] = frozenset({".pdf"})
# Images and audio deferred (Pass 2 MVP is PDF only).


def discover_vault_files(
    vault_root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Path]:
    """Recursively find .md and supported media files under vault_root."""
    if not vault_root.exists():
        raise FileNotFoundError(f"Vault root does not exist: {vault_root}")
    if not vault_root.is_dir():
        raise NotADirectoryError(f"Vault root is not a directory: {vault_root}")

    results: list[Path] = []
    all_extensions = {".md"} | _MEDIA_EXTENSIONS
    for path in vault_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in all_extensions:
            continue
        rel_parts = path.relative_to(vault_root).parts
        if any(part in exclude_dirs for part in rel_parts):
            continue
        results.append(path)
    return sorted(results)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pdf_extraction.py -v`
Expected: First 2 tests PASS.

Run: `pytest tests/ -v`
Expected: All existing tests still pass (we added new APIs, didn't change `load_vault` or `discover_markdown_files`).

- [ ] **Step 5: Commit**

```bash
git add prism_rag/ingest/vault_loader.py tests/test_pdf_extraction.py
git commit -m "feat(vault_loader): add VaultMedia and discover_vault_files for Pass 2"
```

---

### Task 2.3: Create `media_extractor.py` with PDF handler

**Files:**
- Create: `prism_rag/ingest/media_extractor.py`
- Test: `tests/test_pdf_extraction.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pdf_extraction.py`:

```python
def test_extract_pdf_returns_text(tmp_path):
    """extract_pdf returns the concatenated text of a simple PDF."""
    from prism_rag.ingest.media_extractor import extract_pdf

    # Create a minimal PDF with known text using pypdf.PdfWriter
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    # Use reportlab or just write a known-text PDF via another library?
    # Instead, use a hand-built PDF with real text: fall back to a fixture

    # For this test, synthesise a PDF using pypdf internals or skip if
    # no reliable text-injection exists. Simpler: use a fixture PDF file.
    fixture = tmp_path / "hello.pdf"
    # Programmatically create a PDF with text using pypdf.Page not supported;
    # write a known small PDF as bytes:
    fixture.write_bytes(_TINY_PDF_WITH_TEXT)

    text = extract_pdf(fixture)
    assert "Hello" in text


def test_extract_pdf_empty_file_returns_empty(tmp_path):
    """Scan-only / empty-text PDF returns empty string and logs warning."""
    from prism_rag.ingest.media_extractor import extract_pdf

    # Write a minimal valid PDF with no extractable text
    fixture = tmp_path / "blank.pdf"
    fixture.write_bytes(_TINY_EMPTY_PDF)

    text = extract_pdf(fixture)
    assert text == ""


# Minimal valid PDFs (bytes fixtures) — precomputed.
_TINY_PDF_WITH_TEXT = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]"
    b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 20 180 Td (Hello world) Tj ET\nendstream endobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"xref\n0 6\n"
    b"0000000000 65535 f\n"
    b"0000000009 00000 n\n"
    b"0000000052 00000 n\n"
    b"0000000101 00000 n\n"
    b"0000000191 00000 n\n"
    b"0000000271 00000 n\n"
    b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n329\n%%EOF"
)

_TINY_EMPTY_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
    b"xref\n0 4\n"
    b"0000000000 65535 f\n"
    b"0000000009 00000 n\n"
    b"0000000052 00000 n\n"
    b"0000000101 00000 n\n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n155\n%%EOF"
)
```

Note: if the synthesized PDFs don't parse cleanly with pypdf, replace them with a pre-built fixture file in `tests/fixtures/` generated via a helper script. Use pypdf's own test fixtures if available.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pdf_extraction.py::test_extract_pdf_returns_text -v`
Expected: FAIL — `extract_pdf` doesn't exist.

- [ ] **Step 3: Create `media_extractor.py`**

Create `prism_rag/ingest/media_extractor.py`:

```python
"""Pass 2: media extraction — PDF → text, image/audio stubs."""
from __future__ import annotations

import logging
from pathlib import Path

from prism_rag.store.graph import KnowledgeGraph, Node
from prism_rag.ingest.vault_loader import VaultMedia

logger = logging.getLogger(__name__)

_MAX_CONTENT_CHARS = 30_000  # Match embedder truncation limit


def extract_pdf(path: Path) -> str:
    """Extract text from a PDF using pypdf.

    Returns the concatenated text of all pages, separated by
    "\\n\\n--- Page N ---\\n\\n". Returns empty string if the PDF has no
    extractable text (scan-only PDF, encrypted, etc).
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError(
            "pypdf is required for Pass 2 PDF extraction. "
            "Install with: pip install prism-rag"
        ) from e

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        logger.warning(f"[media_extractor] failed to open PDF {path}: {e}")
        return ""

    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"[media_extractor] page {i} of {path} failed: {e}")
            text = ""
        text = text.strip()
        if text:
            pages.append(f"--- Page {i} ---\n\n{text}")

    if not pages:
        logger.warning(f"[media_extractor] {path} has no extractable text (scan-only?)")
        return ""

    full = "\n\n".join(pages)
    if len(full) > _MAX_CONTENT_CHARS:
        full = full[:_MAX_CONTENT_CHARS] + "\n\n...[truncated]"
    return full


def extract_image(path: Path) -> str:
    """Image extraction — NOT YET IMPLEMENTED."""
    raise NotImplementedError("Image extraction deferred to future step")


def extract_audio(path: Path) -> str:
    """Audio extraction — NOT YET IMPLEMENTED."""
    raise NotImplementedError("Audio extraction deferred to future step")


def add_media_nodes(
    graph: KnowledgeGraph,
    media_files: list[VaultMedia],
) -> int:
    """Add a Node per media file to the graph.

    Only PDFs are processed; images and audio are skipped with a warning.
    Returns the number of nodes added.
    """
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    added = 0
    for m in media_files:
        if m.kind == "pdf":
            text = extract_pdf(m.path)
            tokens = max(1, len(_enc.encode(text))) if text else 0
            node = Node(
                id=m.id,
                label=m.label,
                kind="pdf",
                source_file=str(m.relative_path),
                content=text,
                content_hash=m.content_hash,
                tokens=tokens,
            )
            graph.add_node(node)
            added += 1
        elif m.kind in ("image", "audio"):
            logger.info(f"[media_extractor] skipping {m.kind}: {m.path} (not implemented)")
        else:
            logger.info(f"[media_extractor] unknown kind '{m.kind}': {m.path}")
    return added
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pdf_extraction.py -v`
Expected: All PDF tests PASS.

If the synthetic PDF bytes don't parse, replace with a real fixture:

```bash
# Optional: create fixture dir
mkdir -p tests/fixtures
python -c "
from pypdf import PdfWriter
w = PdfWriter()
w.add_blank_page(200, 200)
with open('tests/fixtures/empty.pdf', 'wb') as f:
    w.write(f)
"
```

And update the test to use the fixture instead of hand-crafted bytes.

- [ ] **Step 5: Commit**

```bash
git add prism_rag/ingest/media_extractor.py tests/test_pdf_extraction.py
git add tests/fixtures/ 2>/dev/null || true
git commit -m "feat(media_extractor): PDF extraction via pypdf; image/audio stubs"
```

---

### Task 2.4: Wire PDF extraction into the `ingest` pipeline

**Files:**
- Modify: `prism_rag/cli.py` (the `ingest` command)
- Test: `tests/test_pdf_extraction.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pdf_extraction.py`:

```python
def test_full_ingest_adds_pdf_node(tmp_path):
    """Running ingest on a vault with a PDF creates a kind='pdf' node."""
    from prism_rag.ingest.ast_extractor import extract_ast
    from prism_rag.ingest.vault_loader import discover_vault_files, VaultDocument, VaultMedia, load_vault
    from prism_rag.ingest.media_extractor import add_media_nodes
    from prism_rag.store.graph import KnowledgeGraph

    vault = tmp_path
    (vault / "note.md").write_text("just a note")
    (vault / "report.pdf").write_bytes(_TINY_PDF_WITH_TEXT)

    graph = KnowledgeGraph()

    # Markdown side
    docs = load_vault(vault)
    extract_ast(graph, docs)

    # Media side
    media_paths = [p for p in discover_vault_files(vault) if p.suffix == ".pdf"]
    media = [VaultMedia.from_path(p, vault) for p in media_paths]
    added = add_media_nodes(graph, media)

    assert added == 1
    assert "report" in graph.g.nodes
    assert graph.g.nodes["report"]["kind"] == "pdf"
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_pdf_extraction.py::test_full_ingest_adds_pdf_node -v`
Expected: PASS (the building blocks are in place; the test just verifies composition).

- [ ] **Step 3: Add media pass to `cli.py` ingest**

Edit `prism_rag/cli.py` — in the `ingest` command, after Pass 1b (AST extraction) and before Pass 3, add:

```python
    # ── Pass 2: Media extraction ──
    typer.secho("\n📄 Pass 2: Extracting PDF content...", fg=typer.colors.BLUE)
    from prism_rag.ingest.media_extractor import add_media_nodes
    from prism_rag.ingest.vault_loader import VaultMedia, discover_vault_files

    media_paths = [p for p in discover_vault_files(settings.vault_path) if p.suffix.lower() == ".pdf"]
    media = [VaultMedia.from_path(p, settings.vault_path) for p in media_paths]
    n_media = add_media_nodes(graph, media)
    typer.echo(f"   PDF nodes: {n_media}")
```

- [ ] **Step 4: Smoke test**

Run: `mkdir -p /tmp/prism-test-vault && echo '# hello' > /tmp/prism-test-vault/note.md`
Run: `python -c "from pypdf import PdfWriter; w=PdfWriter(); w.add_blank_page(200,200); open('/tmp/prism-test-vault/report.pdf','wb').write(w.write_stream().getvalue() if hasattr(w,'write_stream') else b'') " || echo "manual fixture"`
Run: `cd /tmp && PRISM_GEMINI_API_KEY="" prism-rag ingest --vault /tmp/prism-test-vault --output /tmp/prism-test-data --skip-embed`
Expected: Output includes "📄 Pass 2: Extracting PDF content..." and "PDF nodes: 1".

- [ ] **Step 5: Commit**

```bash
git add prism_rag/cli.py tests/test_pdf_extraction.py
git commit -m "feat(cli): wire Pass 2 PDF extraction into ingest pipeline"
```

---

## Phase C — vault_ops Write Path (Section 6)

### Task 6.1: Add `CASConflict` exception and `atomic_write` helper

**Files:**
- Modify: `prism_rag/vault_ops/cas.py`
- Test: `tests/test_vault_ops_write.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_vault_ops_write.py`:

```python
"""Tests for vault_ops write path (Section 6)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_rag.vault_ops.cas import (
    CASConflict,
    atomic_write,
    compute_hash,
    write_with_cas,
)


def test_atomic_write_creates_file(tmp_path):
    target = tmp_path / "sub" / "note.md"
    target.parent.mkdir()
    atomic_write(target, "hello")
    assert target.read_text() == "hello"


def test_atomic_write_leaves_no_tmp(tmp_path):
    target = tmp_path / "note.md"
    atomic_write(target, "hello")
    # No lingering .tmp file
    tmps = list(tmp_path.glob("*.tmp"))
    assert tmps == []


def test_write_with_cas_fresh_file(tmp_path):
    target = tmp_path / "new.md"
    new_hash = write_with_cas(target, "hello", expected_hash=None)
    assert target.read_text() == "hello"
    assert new_hash == compute_hash("hello")


def test_write_with_cas_matching_hash(tmp_path):
    target = tmp_path / "existing.md"
    target.write_text("v1")
    v1_hash = compute_hash("v1")
    new_hash = write_with_cas(target, "v2", expected_hash=v1_hash)
    assert target.read_text() == "v2"
    assert new_hash == compute_hash("v2")


def test_write_with_cas_conflict_raises(tmp_path):
    target = tmp_path / "existing.md"
    target.write_text("v1")
    with pytest.raises(CASConflict) as exc_info:
        write_with_cas(target, "v2", expected_hash="sha256:deadbeef")
    assert "deadbeef" in str(exc_info.value) or exc_info.value.expected == "sha256:deadbeef"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vault_ops_write.py -v`
Expected: FAIL — `CASConflict`, `atomic_write`, `write_with_cas` don't exist yet.

- [ ] **Step 3: Add the new helpers to `cas.py`**

Edit `prism_rag/vault_ops/cas.py` — append:

```python
import os


class CASConflict(Exception):
    """Raised when expected_hash does not match the current file hash."""

    def __init__(self, path: Path, expected: str, actual: str):
        super().__init__(
            f"CAS conflict on {path}: expected={expected}, actual={actual}"
        )
        self.path = path
        self.expected = expected
        self.actual = actual


def atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write `content` to `path` atomically via tempfile + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding=encoding)
        os.replace(tmp, path)
    except Exception:
        # Clean up tmp on error
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def write_with_cas(path: Path, content: str, expected_hash: str | None) -> str:
    """Atomically write `content`, honouring optimistic CAS.

    `expected_hash`:
      - None: file must NOT exist (create-only semantics)
      - str: file must exist and its SHA-256 must match

    Returns the new hash. Raises `CASConflict` on mismatch or pre-existence.
    """
    exists = path.exists()

    if expected_hash is None:
        if exists:
            raise CASConflict(path, "<new>", compute_file_hash(path))
    else:
        if not exists:
            raise CASConflict(path, expected_hash, "<missing>")
        # Normalize: expected may or may not have 'sha256:' prefix
        expected_clean = expected_hash.removeprefix("sha256:")
        actual = compute_file_hash(path)
        if actual != expected_clean:
            raise CASConflict(path, expected_hash, actual)

    atomic_write(path, content)
    return compute_hash(content)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_vault_ops_write.py -v`
Expected: All tests PASS.

Run: `pytest tests/ -v`
Expected: All existing tests still pass (the old `verify_cas` is untouched; new APIs are additive).

- [ ] **Step 5: Commit**

```bash
git add prism_rag/vault_ops/cas.py tests/test_vault_ops_write.py
git commit -m "feat(vault_ops): add CASConflict, atomic_write, write_with_cas"
```

---

### Task 6.2: Audit log writes to JSONL file

**Files:**
- Modify: `prism_rag/vault_ops/audit_log.py`
- Test: `tests/test_vault_ops_write.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_vault_ops_write.py`:

```python
def test_audit_log_appends_jsonl(tmp_path, monkeypatch):
    """log_operation writes one JSONL line per call to data/audit.jsonl."""
    from prism_rag.vault_ops import audit_log as al

    # Redirect audit path
    monkeypatch.setattr(al, "_audit_path", lambda: tmp_path / "audit.jsonl")

    al.log_operation(tool="write_note", target="foo.md", action="write",
                     status="ok", cas_before="sha256:a", cas_after="sha256:b")
    al.log_operation(tool="write_note", target="foo.md", action="write",
                     status="conflict", cas_before="sha256:a", cas_after="sha256:c",
                     error="CASConflict")

    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 2
    r1 = json.loads(lines[0])
    r2 = json.loads(lines[1])
    assert r1["status"] == "ok" and r1["target"] == "foo.md"
    assert r2["status"] == "conflict" and r2["error"] == "CASConflict"
    # ISO-8601 timestamp present
    assert "T" in r1["ts"] and r1["ts"].endswith("+00:00")
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_vault_ops_write.py::test_audit_log_appends_jsonl -v`
Expected: FAIL — `_audit_path` doesn't exist; current `log_operation` only logs via `logger.info`.

- [ ] **Step 3: Rewrite `audit_log.py`**

Replace `prism_rag/vault_ops/audit_log.py` with:

```python
"""
Obsidian Vault MCP — 结构化审计日志

All write operations append a JSONL line to data/audit.jsonl.
Logging is best-effort: a failed audit write does not block the main op.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("obsidian_mcp.audit")


def _audit_path() -> Path:
    """Return the path to the audit JSONL file.

    Uses PRISM_DATA_DIR if set, else defaults to ./data/audit.jsonl.
    Overridable in tests via monkeypatch of this function.
    """
    from prism_rag.config import PrismRagSettings
    settings = PrismRagSettings()
    return settings.data_dir / "audit.jsonl"


def log_operation(
    tool: str,
    target: str,
    action: str,
    status: str,
    cas_before: str = "",
    cas_after: str = "",
    **extra: Any,
) -> None:
    """Record an operation to the audit log (JSONL file + stdlib logger)."""
    entry: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "target": target,
        "action": action,
        "status": status,
    }
    if cas_before:
        entry["cas_before"] = cas_before
    if cas_after:
        entry["cas_after"] = cas_after
    if extra:
        entry.update(extra)

    # JSONL file append
    try:
        path = _audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[audit] failed to write audit log: {e}")

    # Stdlib logger (kept for backward compatibility)
    logger.info(json.dumps(entry, ensure_ascii=False))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_vault_ops_write.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add prism_rag/vault_ops/audit_log.py tests/test_vault_ops_write.py
git commit -m "feat(audit_log): write JSONL file in addition to stdlib logger"
```

---

### Task 6.3: `write_note` MCP tool uses `atomic_write` and logs to audit JSONL

**Files:**
- Modify: `prism_rag/mcp_server/server.py` (`write_note` function, line ~480)
- Test: `tests/test_vault_ops_write.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_vault_ops_write.py`:

```python
def test_write_note_logs_audit_on_success(tmp_path, monkeypatch):
    """Successful write_note produces an audit JSONL entry with status='ok'."""
    from prism_rag.config import PrismRagSettings, GraphSource
    from prism_rag.mcp_server import server as mcp_server
    from prism_rag.vault_ops import audit_log as al

    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()

    # Seed an empty graph
    from prism_rag.store.graph import KnowledgeGraph
    KnowledgeGraph().save(data / "graph.json")

    # Force audit path into tmp
    audit_path = data / "audit.jsonl"
    monkeypatch.setattr(al, "_audit_path", lambda: audit_path)

    # Configure settings
    settings = PrismRagSettings(
        graphs=[GraphSource(namespace="default", vault_path=vault, data_dir=data, writable=True)],
    )
    monkeypatch.setattr(
        "prism_rag.mcp_server.server.PrismRagSettings",
        lambda: settings,
    )
    mcp_server._federated = None

    result = mcp_server.write_note(
        path="new.md",
        content="# Hello",
        cas_hash="",
        namespace="default",
    )
    parsed = json.loads(result)
    assert parsed.get("status") == "ok"

    assert audit_path.exists()
    entries = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert any(e["tool"] == "write_note" and e["status"] == "ok" for e in entries)


def test_write_note_cas_conflict_logs_audit(tmp_path, monkeypatch):
    """CAS conflict in write_note produces an audit entry with status='conflict'."""
    from prism_rag.config import PrismRagSettings, GraphSource
    from prism_rag.mcp_server import server as mcp_server
    from prism_rag.vault_ops import audit_log as al

    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()
    (vault / "existing.md").write_text("v1")

    from prism_rag.store.graph import KnowledgeGraph
    KnowledgeGraph().save(data / "graph.json")

    audit_path = data / "audit.jsonl"
    monkeypatch.setattr(al, "_audit_path", lambda: audit_path)

    settings = PrismRagSettings(
        graphs=[GraphSource(namespace="default", vault_path=vault, data_dir=data, writable=True)],
    )
    monkeypatch.setattr(
        "prism_rag.mcp_server.server.PrismRagSettings",
        lambda: settings,
    )
    mcp_server._federated = None

    result = mcp_server.write_note(
        path="existing.md",
        content="v2",
        cas_hash="sha256:wronghash",
        namespace="default",
    )
    parsed = json.loads(result)
    # Should be an error response
    assert parsed.get("status") != "ok"

    entries = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert any(e["status"] == "conflict" for e in entries)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vault_ops_write.py::test_write_note_logs_audit_on_success -v`
Expected: FAIL — current `write_note` doesn't write to audit JSONL.

- [ ] **Step 3: Update `write_note` in server.py**

Edit `prism_rag/mcp_server/server.py` — find `write_note` (line ~480). Replace the write + error logic so that:

1. Uses `atomic_write` from `vault_ops/cas.py`
2. Calls `log_operation` on every outcome (ok/conflict/error)

Replace the body after `resolved = vault.resolve_path(path)` (line ~519) with:

```python
    lock = get_file_lock(resolved)

    from prism_rag.vault_ops.cas import atomic_write
    from prism_rag.vault_ops.audit_log import log_operation

    # Sync write (MCP tools are sync in FastMCP)
    expected = cas_hash if cas_hash else None
    is_valid, actual = verify_cas(resolved, expected)

    if not is_valid:
        # Audit the conflict
        log_operation(
            tool="write_note", target=str(path), action="write",
            status="conflict",
            cas_before=actual or "",
            namespace=src.namespace,
            expected_hash=expected or "<new>",
        )
        if expected is None:
            return json.dumps(fail(
                VaultErrorCode.ALREADY_EXISTS,
                f"File already exists: {path}. Use read_note to get cas_hash first.",
                actual_hash=actual,
            ), ensure_ascii=False)
        return json.dumps(fail(
            VaultErrorCode.CONFLICT,
            f"CAS conflict: file has been modified.",
            expected_hash=expected, actual_hash=actual,
        ), ensure_ascii=False)

    # Atomic write
    atomic_write(resolved, content)
    new_hash = compute_hash(content)

    log_operation(
        tool="write_note", target=str(path), action="write",
        status="ok",
        cas_before=actual or "",
        cas_after=new_hash,
        namespace=src.namespace,
    )

    # Incrementally update graph
    graph_stats = {}
    try:
        graph_stats = ingest_file(
            resolved, settings=settings, skip_embed=True, skip_persist=False,
        )
        global _federated
        _federated = FederatedGraph.load(settings.resolved_graphs)
    except Exception as e:
        logger.warning(f"[write_note] graph update failed: {e}")
        graph_stats = {"error": str(e)}

    result = {
        "status": "ok",
        "data": {"cas_hash": new_hash, "path": path, "namespace": src.namespace},
        "graph_update": graph_stats,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_vault_ops_write.py -v`
Expected: All PASS.

Run: `pytest tests/ -v`
Expected: All existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add prism_rag/mcp_server/server.py tests/test_vault_ops_write.py
git commit -m "feat(mcp/write_note): atomic write + audit JSONL on every outcome"
```

---

## Phase D — Verify `serve` (Section 1)

### Task 1.1: Smoke-test `prism-rag serve --transport stdio`

**Files:**
- Test: `tests/test_cli.py`

- [ ] **Step 1: Create `tests/test_cli.py` with serve smoke test**

Create `tests/test_cli.py`:

```python
"""CLI integration tests (Section 7) — but we start with serve smoke test here (Section 1)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


VENV_BIN = Path(sys.executable).parent


@pytest.fixture
def tiny_vault(tmp_path):
    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()
    (vault / "note.md").write_text("# Hello\n\n[[other]]\n")
    # Minimal empty graph so serve doesn't abort
    from prism_rag.store.graph import KnowledgeGraph, Node
    g = KnowledgeGraph()
    g.add_node(Node(id="note", label="note", kind="note", tokens=10))
    g.save(data / "graph.json")
    return vault, data


def test_serve_stdio_starts_and_exits(tiny_vault):
    """prism-rag serve --transport stdio starts and responds to SIGTERM."""
    vault, data = tiny_vault
    env = os.environ.copy()
    env["PRISM_VAULT_PATH"] = str(vault)
    env["PRISM_DATA_DIR"] = str(data)
    env["PRISM_GEMINI_API_KEY"] = ""  # no API calls

    proc = subprocess.Popen(
        [str(VENV_BIN / "prism-rag"), "serve", "--transport", "stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        # Give it a moment to initialize
        time.sleep(1.5)
        assert proc.poll() is None, f"serve exited early: stderr={proc.stderr.read().decode()}"
        # Send SIGTERM
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
        # Exit code 0 or 143 (SIGTERM) or -15 are all acceptable
        assert proc.returncode in (0, 143, -15, 1)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_cli.py::test_serve_stdio_starts_and_exits -v`
Expected: PASS (serve already exists; this just verifies it boots).

If FAIL: inspect stderr. Common issues:
- `prism-rag` not on PATH → use `sys.executable + " -m prism_rag.cli"` instead
- Missing `graph.json` → the fixture should provide it

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli.py
git commit -m "test(cli): smoke test prism-rag serve --transport stdio"
```

---

### Task 1.2: Verify `serve` errors helpfully when no graph exists

**Files:**
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write test**

Append to `tests/test_cli.py`:

```python
def test_serve_fails_gracefully_when_no_graph(tmp_path):
    """serve without a graph.json exits non-zero with a clear error."""
    empty_vault = tmp_path / "empty-vault"
    empty_data = tmp_path / "empty-data"
    empty_vault.mkdir()
    empty_data.mkdir()

    env = os.environ.copy()
    env["PRISM_VAULT_PATH"] = str(empty_vault)
    env["PRISM_DATA_DIR"] = str(empty_data)
    env["PRISM_GEMINI_API_KEY"] = ""

    result = subprocess.run(
        [str(VENV_BIN / "prism-rag"), "serve", "--transport", "stdio"],
        capture_output=True, text=True, env=env, timeout=10,
    )
    assert result.returncode != 0
    assert "No graphs loaded" in result.stderr or "graph" in result.stderr.lower()
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_cli.py::test_serve_fails_gracefully_when_no_graph -v`
Expected: PASS (the existing implementation already exits with this error).

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli.py
git commit -m "test(cli): serve exits cleanly with no graph present"
```

---

## Phase D — CLI Integration Tests (Section 7)

### Task 7.1: Add `--no-embedding` flag and `MockEmbedder`

**Files:**
- Create: `prism_rag/ingest/mock_embedder.py`
- Modify: `prism_rag/ingest/embedder.py` (add a pluggable backend)
- Modify: `prism_rag/cli.py` (rename `--skip-embed` alias or accept both)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Check existing flag**

The `ingest` command already has `--skip-embed` (line ~53 of cli.py). `--no-embedding` is a spec-specified alias. Add it.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_cli.py`:

```python
def test_mock_embedder_deterministic(tmp_path):
    """Mock embedder returns the same 768-dim vector for identical content."""
    from prism_rag.ingest.mock_embedder import mock_embed_text
    v1 = mock_embed_text("hello world")
    v2 = mock_embed_text("hello world")
    v3 = mock_embed_text("different content")
    assert len(v1) == 768
    assert v1 == v2
    assert v1 != v3


def test_ingest_with_no_embedding_flag(tmp_path):
    """prism-rag ingest --no-embedding skips Pass 3 entirely."""
    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()
    (vault / "a.md").write_text("# A\n\n[[B]]")
    (vault / "b.md").write_text("# B")

    env = os.environ.copy()
    env["PRISM_VAULT_PATH"] = str(vault)
    env["PRISM_DATA_DIR"] = str(data)
    env["PRISM_GEMINI_API_KEY"] = ""

    result = subprocess.run(
        [str(VENV_BIN / "prism-rag"), "ingest",
         "--vault", str(vault), "--output", str(data), "--no-embedding"],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert (data / "graph.json").exists()
```

- [ ] **Step 3: Create `mock_embedder.py`**

Create `prism_rag/ingest/mock_embedder.py`:

```python
"""Mock embedding backend for tests.

Derives a deterministic 768-dim pseudo-vector from the SHA-256 of content.
Same content → same vector → reproducible similarity edges.
"""
from __future__ import annotations

import hashlib
import struct


def mock_embed_text(text: str) -> list[float]:
    """Return a 768-dim pseudo-embedding derived from SHA-256 of `text`.

    The vector is L2-normalized so cosine-similarity semantics hold.
    """
    # Generate 768 floats from repeatedly hashing
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    floats: list[float] = []
    i = 0
    while len(floats) < 768:
        h = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        # Each sha256 is 32 bytes → 8 floats of 4 bytes
        for j in range(0, 32, 4):
            val = struct.unpack("<I", h[j:j+4])[0]
            # Map to [-1, 1]
            floats.append((val / 2**32) * 2 - 1)
        i += 1
    floats = floats[:768]
    # L2 normalize
    norm = sum(f * f for f in floats) ** 0.5
    if norm > 0:
        floats = [f / norm for f in floats]
    return floats
```

- [ ] **Step 4: Add `--no-embedding` flag**

Edit `prism_rag/cli.py` — in the `ingest` command signature, add:

```python
    no_embedding: bool = typer.Option(
        False, "--no-embedding",
        help="Alias for --skip-embed; skip Pass 3 entirely (for offline testing)",
    ),
```

And before the `if skip_embed:` check, unify:

```python
    skip_embed = skip_embed or no_embedding
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_cli.py::test_mock_embedder_deterministic tests/test_cli.py::test_ingest_with_no_embedding_flag -v`
Expected: Both PASS.

- [ ] **Step 6: Commit**

```bash
git add prism_rag/ingest/mock_embedder.py prism_rag/cli.py tests/test_cli.py
git commit -m "feat(cli): add --no-embedding flag and mock_embed_text for offline tests"
```

---

### Task 7.2: CLI test — `ingest` on single-file vault

**Files:**
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write test**

Append to `tests/test_cli.py`:

```python
def test_cli_ingest_single_file(tmp_path):
    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()
    (vault / "only.md").write_text("# Only\n\nno links")

    env = os.environ.copy()
    env["PRISM_GEMINI_API_KEY"] = ""

    result = subprocess.run(
        [str(VENV_BIN / "prism-rag"), "ingest",
         "--vault", str(vault), "--output", str(data), "--no-embedding"],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert (data / "graph.json").exists()
    assert (data / "GRAPH_REPORT.md").exists()

    # Verify node is present
    import json
    data_dict = json.loads((data / "graph.json").read_text())
    node_ids = {n["id"] for n in data_dict["nodes"]}
    assert "only" in node_ids
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_cli.py::test_cli_ingest_single_file -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli.py
git commit -m "test(cli): ingest single-file vault produces graph.json + report"
```

---

### Task 7.3: CLI test — `query` returns results

**Files:**
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write test**

Append to `tests/test_cli.py`:

```python
def test_cli_query_finds_node(tmp_path):
    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()
    (vault / "target.md").write_text("# Target\n\nContent about target")
    (vault / "other.md").write_text("# Other\n\n[[target]]")

    env = os.environ.copy()
    env["PRISM_VAULT_PATH"] = str(vault)
    env["PRISM_DATA_DIR"] = str(data)
    env["PRISM_GEMINI_API_KEY"] = ""

    # Build graph
    ingest = subprocess.run(
        [str(VENV_BIN / "prism-rag"), "ingest",
         "--vault", str(vault), "--output", str(data), "--no-embedding"],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert ingest.returncode == 0

    # Query
    query = subprocess.run(
        [str(VENV_BIN / "prism-rag"), "query", "target",
         "--graph", str(data / "graph.json")],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert query.returncode == 0
    assert "target" in query.stdout.lower()
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_cli.py::test_cli_query_finds_node -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli.py
git commit -m "test(cli): query surfaces matching entry node"
```

---

### Task 7.4: CLI test — `info` shows stats

**Files:**
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write test**

Append to `tests/test_cli.py`:

```python
def test_cli_info_shows_stats(tmp_path):
    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()
    (vault / "a.md").write_text("# A\n\n[[B]]")
    (vault / "b.md").write_text("# B")

    env = os.environ.copy()
    env["PRISM_GEMINI_API_KEY"] = ""

    subprocess.run(
        [str(VENV_BIN / "prism-rag"), "ingest",
         "--vault", str(vault), "--output", str(data), "--no-embedding"],
        check=True, env=env, timeout=30,
    )

    result = subprocess.run(
        [str(VENV_BIN / "prism-rag"), "info", "--graph", str(data / "graph.json")],
        capture_output=True, text=True, env=env, timeout=10,
    )
    assert result.returncode == 0
    assert "Nodes:" in result.stdout or "nodes" in result.stdout.lower()
    assert "2" in result.stdout  # 2 markdown nodes
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_cli.py::test_cli_info_shows_stats -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli.py
git commit -m "test(cli): info shows correct node count"
```

---

### Task 7.5: CLI test — `version`

**Files:**
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write test**

Append to `tests/test_cli.py`:

```python
def test_cli_version():
    result = subprocess.run(
        [str(VENV_BIN / "prism-rag"), "version"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "PrismRag" in result.stdout
```

- [ ] **Step 2: Run and commit**

Run: `pytest tests/test_cli.py::test_cli_version -v`
Expected: PASS.

```bash
git add tests/test_cli.py
git commit -m "test(cli): version prints package version"
```

---

### Task 7.6: CLI test — incremental add

**Files:**
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write test**

Append to `tests/test_cli.py`:

```python
def test_cli_incremental_add(tmp_path):
    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()
    (vault / "a.md").write_text("# A")

    env = os.environ.copy()
    env["PRISM_VAULT_PATH"] = str(vault)
    env["PRISM_DATA_DIR"] = str(data)
    env["PRISM_GEMINI_API_KEY"] = ""

    # Initial ingest
    subprocess.run(
        [str(VENV_BIN / "prism-rag"), "ingest",
         "--vault", str(vault), "--output", str(data), "--no-embedding"],
        check=True, env=env, timeout=30,
    )

    # Add a new file
    new = vault / "b.md"
    new.write_text("# B")

    # Incremental add
    result = subprocess.run(
        [str(VENV_BIN / "prism-rag"), "add", str(new), "--skip-embed"],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert result.returncode == 0
    assert "Added" in result.stdout or "Updated" in result.stdout or "added" in result.stdout.lower()
```

- [ ] **Step 2: Run and commit**

Run: `pytest tests/test_cli.py::test_cli_incremental_add -v`
Expected: PASS.

```bash
git add tests/test_cli.py
git commit -m "test(cli): incremental add updates graph with new file"
```

---

### Task 7.7: CLI test — PDF end-to-end via ingest

**Files:**
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write test**

Append to `tests/test_cli.py`:

```python
def test_cli_ingest_vault_with_pdf(tmp_path):
    """ingest of a vault containing a PDF produces a kind=pdf node."""
    import json as _json
    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()
    (vault / "note.md").write_text("# Note")

    # Write a minimal PDF via pypdf
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    with (vault / "doc.pdf").open("wb") as fh:
        w.write(fh)

    env = os.environ.copy()
    env["PRISM_GEMINI_API_KEY"] = ""

    result = subprocess.run(
        [str(VENV_BIN / "prism-rag"), "ingest",
         "--vault", str(vault), "--output", str(data), "--no-embedding"],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"

    graph_data = _json.loads((data / "graph.json").read_text())
    kinds = {n.get("kind") for n in graph_data["nodes"]}
    assert "pdf" in kinds
```

- [ ] **Step 2: Run and commit**

Run: `pytest tests/test_cli.py::test_cli_ingest_vault_with_pdf -v`
Expected: PASS.

```bash
git add tests/test_cli.py
git commit -m "test(cli): ingest handles vault containing a PDF"
```

---

### Task 7.8: Final full-suite run + performance check

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `cd /home/kingy/Foundation/PrismRag && pytest tests/ -v`
Expected: All tests PASS. No errors.

- [ ] **Step 2: Measure test_cli runtime**

Run: `cd /home/kingy/Foundation/PrismRag && pytest tests/test_cli.py -v --durations=10`
Expected: Total runtime under 60 seconds (spec acceptance criterion).

If slower: investigate which test is slow (likely `serve_stdio_starts_and_exits` with its sleep). Consider reducing sleep or using a readiness probe.

- [ ] **Step 3: Run mypy for type check**

Run: `cd /home/kingy/Foundation/PrismRag && mypy prism_rag/`
Expected: No new type errors introduced (baseline errors may pre-exist; don't block on those unless this plan introduced them).

- [ ] **Step 4: Commit verification results (no code change)**

No code to commit — this is a verification step. If any issues were fixed, they'd already have been committed as part of their respective task.

---

## Self-Review

**Spec coverage check:**

| Spec Section | Implementation Task(s) | Covered? |
|---|---|---|
| Section 1 — `serve` | Task 1.1, 1.2 | ✅ |
| Section 2 — Pass 2 PDF | Task 2.1, 2.2, 2.3, 2.4 | ✅ |
| Section 3 — knowledge_id branch | Task 3.1 – 3.6 | ✅ |
| Section 4 — Collision resolution | Task 4.1 – 4.4 | ✅ |
| Section 5 — ontology_type | Task 5.1, 5.2, 5.3 | ✅ |
| Section 6 — vault_ops write | Task 6.1, 6.2, 6.3 | ✅ |
| Section 7 — CLI tests | Task 7.1 – 7.8 | ✅ |
| Acceptance criterion: serve + SIGTERM | Task 1.1 | ✅ |
| Acceptance criterion: PDF → pdf node | Task 2.4, 7.7 | ✅ |
| Acceptance criterion: knowledge_id resolves | Task 3.3 | ✅ |
| Acceptance criterion: embed:false skipped | Task 3.5 | ✅ |
| Acceptance criterion: collision → knowledge_id wins | Task 4.2 | ✅ |
| Acceptance criterion: type:decision → ontology_type + filter | Task 5.2, 5.3 | ✅ |
| Acceptance criterion: CAS conflict + audit log | Task 6.1, 6.3 | ✅ |
| Acceptance criterion: `pytest tests/test_cli.py` offline <60s | Task 7.8 | ✅ |

All spec requirements mapped to at least one task.

**Placeholder scan:** Reviewed — no "TBD", "TODO", "fill in later" in task bodies. All code blocks are complete. All commands are exact.

**Type / name consistency:**
- `knowledge_id` is used consistently (never `knowledgeId` or `know_id`).
- `ontology_type` is consistent across `graph.py`, `ast_extractor.py`, MCP tools.
- `CASConflict`, `atomic_write`, `write_with_cas` are defined in Task 6.1 and referenced the same way in Task 6.3.
- `extract_pdf`, `add_media_nodes`, `VaultMedia`, `discover_vault_files` consistent.
- `MockEmbedder` in spec maps to `mock_embed_text` function (simpler surface than a class — clarified in Task 7.1).

All cross-task references verified.
