"""Comprehensive tests for the 3-phase atomize workflow.

Covers edge cases for scan / propose / apply, end-to-end flows, and
graph integration with mocked embeddings.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from prism_rag.ingest.atomize import (
    atomize_scan_impl,
    atomize_propose_impl,
    atomize_apply_impl,
    ScanExpiredError,
    StaleDocError,
)
from prism_rag.store.graph import KnowledgeGraph, Node


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _write_doc(tmp_path: Path, content: str, name: str = "design.md") -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _full_scan(vault: Path, doc: Path, scan_dir: Path) -> dict[str, Any]:
    return atomize_scan_impl(doc, vault_root=vault, scan_dir=scan_dir)


def _full_propose(
    scan: dict[str, Any],
    scan_dir: Path,
    pending_dir: Path,
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    return atomize_propose_impl(
        scan_id=scan["scan_id"],
        claims=claims,
        scan_dir=scan_dir,
        proposal_dir=pending_dir,
    )


def _make_claim(
    section_id: str,
    knowledge_id: str,
    title: str = "Test Concept",
    body: str = "Body text.",
    ontology_type: str = "concept",
) -> dict[str, Any]:
    return {
        "section_id": section_id,
        "knowledge_id": knowledge_id,
        "title": title,
        "body": body,
        "ontology_type": ontology_type,
    }


def _do_apply(
    proposal: dict[str, Any],
    vault: Path,
    pending_dir: Path,
    applied_dir: Path,
    graph_path: Path | None = None,
) -> dict[str, Any]:
    return atomize_apply_impl(
        proposal_id=proposal["proposal_id"],
        vault_root=vault,
        pending_dir=pending_dir,
        applied_dir=applied_dir,
        graph_path=graph_path,
    )


# ---------------------------------------------------------------------------
# Group 1: Scan edge cases
# ---------------------------------------------------------------------------

def test_scan_raises_on_nonexistent_file(tmp_path):
    """atomize_scan_impl must raise FileNotFoundError for a missing doc."""
    missing = tmp_path / "no_such_file.md"
    with pytest.raises(FileNotFoundError):
        atomize_scan_impl(missing, vault_root=tmp_path, scan_dir=tmp_path / "scan_cache")


def test_scan_hint_field_present(tmp_path):
    """Result dict must contain a 'hint' key."""
    doc = _write_doc(tmp_path, "# Title\n\nContent")
    result = atomize_scan_impl(doc, vault_root=tmp_path, scan_dir=tmp_path / "scan_cache")
    assert "hint" in result


def test_scan_hint_mentions_read_note(tmp_path):
    """Hint text must mention 'read_note' so the LLM knows to call it first."""
    doc = _write_doc(tmp_path, "# Title\n\nContent")
    result = atomize_scan_impl(doc, vault_root=tmp_path, scan_dir=tmp_path / "scan_cache")
    assert "read_note" in result["hint"]


def test_scan_already_atomized_doc(tmp_path):
    """A doc that already has atomized_nodes frontmatter should scan fine."""
    content = textwrap.dedent("""\
        ---
        title: Already Atomized
        atomized_nodes:
          - KNOW-000001
        ---
        # Already Atomized

        Some content that was previously atomized.

        ## Section Two

        More content here.
    """)
    doc = _write_doc(tmp_path, content)
    result = atomize_scan_impl(doc, vault_root=tmp_path, scan_dir=tmp_path / "scan_cache")
    assert len(result["sections"]) >= 1


def test_scan_doc_with_no_headings(tmp_path):
    """A doc with no headings should still return at least one section (full doc)."""
    content = "Just some plain text.\nNo headings anywhere.\nMore text."
    doc = _write_doc(tmp_path, content)
    result = atomize_scan_impl(doc, vault_root=tmp_path, scan_dir=tmp_path / "scan_cache")
    assert len(result["sections"]) >= 1


# ---------------------------------------------------------------------------
# Group 2: Propose edge cases
# ---------------------------------------------------------------------------

def _setup_scan(tmp_path: Path, content: str | None = None) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Return (vault, scan_dir, pending_dir, scan_result)."""
    vault = tmp_path / "vault"
    vault.mkdir()
    doc = vault / "source.md"
    doc.write_text(
        content or "# Source\n\nIntro text.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B.",
        encoding="utf-8",
    )
    scan_dir = tmp_path / "scan_cache"
    pending_dir = tmp_path / "pending"
    scan = atomize_scan_impl(doc, vault_root=vault, scan_dir=scan_dir)
    return vault, scan_dir, pending_dir, scan


def test_propose_three_claims(tmp_path):
    """Proposing 3 distinct claims results in claim_count=3."""
    vault, scan_dir, pending_dir, scan = _setup_scan(tmp_path)
    sid = scan["sections"][0]["section_id"]
    claims = [
        _make_claim(sid, f"KNOW-00000{i}", title=f"Claim {i}")
        for i in range(1, 4)
    ]
    result = _full_propose(scan, scan_dir, pending_dir, claims)
    assert result["claim_count"] == 3


def test_propose_empty_claims(tmp_path):
    """Proposing with claims=[] should succeed with claim_count=0."""
    vault, scan_dir, pending_dir, scan = _setup_scan(tmp_path)
    result = _full_propose(scan, scan_dir, pending_dir, [])
    assert result["claim_count"] == 0
    # Proposal file should exist
    proposal_file = pending_dir / f"{result['proposal_id']}.json"
    assert proposal_file.exists()


def test_propose_section_id_not_in_scan(tmp_path):
    """A claim with a section_id that doesn't exist gets empty content_snapshot but no error."""
    vault, scan_dir, pending_dir, scan = _setup_scan(tmp_path)
    claims = [_make_claim("nonexistent-section-999", "KNOW-000001")]
    # Should not raise
    result = _full_propose(scan, scan_dir, pending_dir, claims)
    # Check that the proposal was written with empty snapshot
    proposal_file = pending_dir / f"{result['proposal_id']}.json"
    proposal = json.loads(proposal_file.read_text())
    assert proposal["claims"][0]["content_snapshot"] == ""


def test_propose_claim_body_preserved(tmp_path):
    """The claim body text must appear intact in the written proposal JSON."""
    vault, scan_dir, pending_dir, scan = _setup_scan(tmp_path)
    body_text = "Unique body content that must be preserved verbatim."
    sid = scan["sections"][0]["section_id"]
    claims = [_make_claim(sid, "KNOW-000001", body=body_text)]
    result = _full_propose(scan, scan_dir, pending_dir, claims)
    proposal_file = pending_dir / f"{result['proposal_id']}.json"
    proposal = json.loads(proposal_file.read_text())
    assert proposal["claims"][0]["body"] == body_text


# ---------------------------------------------------------------------------
# Group 3: Apply edge cases
# ---------------------------------------------------------------------------

def _setup_proposal_n(
    tmp_path: Path,
    n: int,
    content: str | None = None,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    """Full scan + propose with n claims. Returns (vault, scan_dir, pending_dir, applied_dir, proposal)."""
    vault = tmp_path / "vault"
    vault.mkdir()
    doc = vault / "source.md"
    doc.write_text(
        content or "# Source\n\nContent here.",
        encoding="utf-8",
    )
    scan_dir = tmp_path / "scan_cache"
    pending_dir = tmp_path / "pending"
    applied_dir = tmp_path / "applied"

    scan = atomize_scan_impl(doc, vault_root=vault, scan_dir=scan_dir)
    sid = scan["sections"][0]["section_id"]
    claims = [
        _make_claim(sid, f"KNOW-{i:06d}", title=f"Concept {i}", body=f"Body for concept {i}.")
        for i in range(1, n + 1)
    ]
    proposal = atomize_propose_impl(
        scan_id=scan["scan_id"],
        claims=claims,
        scan_dir=scan_dir,
        proposal_dir=pending_dir,
    )
    return vault, scan_dir, pending_dir, applied_dir, proposal


def test_apply_three_claims_creates_three_files(tmp_path):
    """Applying a 3-claim proposal creates exactly 3 KNOW-*.md files."""
    vault, scan_dir, pending_dir, applied_dir, proposal = _setup_proposal_n(tmp_path, 3)
    _do_apply(proposal, vault, pending_dir, applied_dir)
    know_dir = vault / "knowledge"
    know_files = list(know_dir.glob("KNOW-*.md"))
    assert len(know_files) == 3


def test_apply_three_claims_all_patch_source_doc(tmp_path):
    """All 3 KNOW-IDs must appear in the source doc's atomized_nodes frontmatter."""
    vault, scan_dir, pending_dir, applied_dir, proposal = _setup_proposal_n(tmp_path, 3)
    doc = vault / "source.md"
    _do_apply(proposal, vault, pending_dir, applied_dir)
    updated = doc.read_text()
    for i in range(1, 4):
        assert f"KNOW-{i:06d}" in updated


def test_apply_applied_count_in_result(tmp_path):
    """result['applied_count'] must equal the number of claims in the proposal."""
    vault, scan_dir, pending_dir, applied_dir, proposal = _setup_proposal_n(tmp_path, 3)
    result = _do_apply(proposal, vault, pending_dir, applied_dir)
    assert result["applied_count"] == 3


def test_apply_raises_on_unknown_proposal_id(tmp_path):
    """atomize_apply_impl raises FileNotFoundError for a non-existent proposal_id."""
    vault = tmp_path / "vault"
    vault.mkdir()
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    applied_dir = tmp_path / "applied"
    with pytest.raises(FileNotFoundError):
        atomize_apply_impl(
            proposal_id="nonexistent-proposal-id",
            vault_root=vault,
            pending_dir=pending_dir,
            applied_dir=applied_dir,
        )


def test_apply_body_content_in_knowledge_file(tmp_path):
    """The claim body text must appear in the written KNOW file."""
    vault, scan_dir, pending_dir, applied_dir, proposal = _setup_proposal_n(tmp_path, 1)
    _do_apply(proposal, vault, pending_dir, applied_dir)
    know_file = vault / "knowledge" / "KNOW-000001.md"
    assert know_file.exists()
    text = know_file.read_text()
    assert "Body for concept 1." in text


def test_apply_result_has_knowledge_files_list(tmp_path):
    """result['knowledge_files'] must list all created file paths."""
    vault, scan_dir, pending_dir, applied_dir, proposal = _setup_proposal_n(tmp_path, 3)
    result = _do_apply(proposal, vault, pending_dir, applied_dir)
    assert "knowledge_files" in result
    assert len(result["knowledge_files"]) == 3
    for entry in result["knowledge_files"]:
        assert entry.startswith("knowledge/KNOW-")


# ---------------------------------------------------------------------------
# Group 4: Full end-to-end
# ---------------------------------------------------------------------------

def test_full_workflow_single_claim(tmp_path):
    """Complete scan → propose → apply flow for a single claim."""
    vault = tmp_path / "vault"
    vault.mkdir()
    doc = vault / "design.md"
    doc.write_text("# Design\n\nThis is the design document.", encoding="utf-8")

    scan_dir = tmp_path / "scan_cache"
    pending_dir = tmp_path / "pending"
    applied_dir = tmp_path / "applied"

    scan = _full_scan(vault, doc, scan_dir)
    sid = scan["sections"][0]["section_id"]
    proposal = _full_propose(
        scan, scan_dir, pending_dir,
        [_make_claim(sid, "KNOW-000001", title="Design Principle", body="Body of design principle.")],
    )
    result = _do_apply(proposal, vault, pending_dir, applied_dir)

    # Knowledge file created
    know_file = vault / "knowledge" / "KNOW-000001.md"
    assert know_file.exists()
    assert "Design Principle" in know_file.read_text()

    # Source doc patched
    assert "KNOW-000001" in doc.read_text()

    # Proposal moved to applied
    assert (applied_dir / f"{proposal['proposal_id']}.json").exists()


def test_full_workflow_three_claims(tmp_path):
    """Complete scan → propose 3 → apply verifies 3 KNOW files and source patch."""
    vault = tmp_path / "vault"
    vault.mkdir()
    doc = vault / "arch.md"
    doc.write_text(
        "# Architecture\n\nOverview.\n\n## Module A\n\nA detail.\n\n## Module B\n\nB detail.",
        encoding="utf-8",
    )

    scan_dir = tmp_path / "scan_cache"
    pending_dir = tmp_path / "pending"
    applied_dir = tmp_path / "applied"

    scan = _full_scan(vault, doc, scan_dir)
    sid = scan["sections"][0]["section_id"]
    claims = [
        _make_claim(sid, f"KNOW-{i:06d}", title=f"Arch Concept {i}", body=f"Arch body {i}.")
        for i in range(1, 4)
    ]
    proposal = _full_propose(scan, scan_dir, pending_dir, claims)
    result = _do_apply(proposal, vault, pending_dir, applied_dir)

    assert result["applied_count"] == 3
    know_dir = vault / "knowledge"
    assert len(list(know_dir.glob("KNOW-*.md"))) == 3
    patched = doc.read_text()
    for i in range(1, 4):
        assert f"KNOW-{i:06d}" in patched


def test_full_workflow_re_atomize(tmp_path):
    """Re-atomize a doc that already has atomized_nodes — new KIDs are additive."""
    vault = tmp_path / "vault"
    vault.mkdir()
    doc = vault / "notes.md"
    doc.write_text("# Notes\n\nContent to atomize.", encoding="utf-8")

    scan_dir = tmp_path / "scan_cache"
    pending_dir = tmp_path / "pending"
    applied_dir = tmp_path / "applied"

    # First round
    scan1 = _full_scan(vault, doc, scan_dir)
    sid1 = scan1["sections"][0]["section_id"]
    proposal1 = _full_propose(scan1, scan_dir, pending_dir,
                               [_make_claim(sid1, "KNOW-000001", title="First")])
    _do_apply(proposal1, vault, pending_dir, applied_dir)

    # Verify first atomization
    assert "KNOW-000001" in doc.read_text()

    # Second round — doc now has atomized_nodes in frontmatter
    scan2 = _full_scan(vault, doc, scan_dir)
    sid2 = scan2["sections"][0]["section_id"]
    # Use a fresh pending dir to avoid collision
    pending_dir2 = tmp_path / "pending2"
    proposal2 = _full_propose(scan2, scan_dir, pending_dir2,
                               [_make_claim(sid2, "KNOW-000002", title="Second")])
    _do_apply(proposal2, vault, pending_dir2, applied_dir)

    # Both KIDs must appear in source doc
    final_text = doc.read_text()
    assert "KNOW-000001" in final_text
    assert "KNOW-000002" in final_text


# ---------------------------------------------------------------------------
# Group 5: Post-apply graph integration (mock embedding)
# ---------------------------------------------------------------------------

def _build_minimal_graph(graph_path: Path) -> None:
    """Save a minimal graph with one existing note node."""
    graph = KnowledgeGraph()
    graph.add_node(Node(
        id="note/existing",
        label="Existing Note",
        kind="note",
        content="Some existing content about embeddings and knowledge.",
        tokens=10,
    ))
    graph.save(graph_path)


def test_apply_with_graph_adds_know_node(tmp_path, monkeypatch):
    """atomize_apply_impl with graph_path ingests KNOW node into the graph.

    Monkeypatches compute_embeddings to return a deterministic vector so no
    Ollama/Gemini service is required.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    graph_path = data_dir / "graph.json"

    doc = vault / "source.md"
    doc.write_text("# Source\n\nContent for graph test.", encoding="utf-8")

    scan_dir = tmp_path / "scan_cache"
    pending_dir = tmp_path / "pending"
    applied_dir = tmp_path / "applied"

    _build_minimal_graph(graph_path)

    scan = atomize_scan_impl(doc, vault_root=vault, scan_dir=scan_dir)
    sid = scan["sections"][0]["section_id"]
    proposal = atomize_propose_impl(
        scan_id=scan["scan_id"],
        claims=[_make_claim(sid, "KNOW-000001", title="Graph Node Concept", body="Body for graph node.")],
        scan_dir=scan_dir,
        proposal_dir=pending_dir,
    )

    # Patch compute_embeddings to return a fake 4-dim vector
    fake_vector = [1.0, 0.0, 0.0, 0.0]

    def fake_compute_embeddings(graph, settings, **kwargs):
        return {nid: fake_vector for nid in graph.g.nodes()}

    monkeypatch.setattr("prism_rag.ingest.embedder.compute_embeddings", fake_compute_embeddings)

    # Also patch EmbeddingStore to avoid LanceDB dimension issues in tests
    from prism_rag.store import embedding_store as es_mod

    class FakeEmbeddingStore:
        def __init__(self, *a, **kw):
            self._data: dict[str, list[float]] = {}

        def upsert(self, node_id, vec):
            self._data[node_id] = vec

        def get(self, node_id):
            return self._data.get(node_id)

        def all_embeddings(self):
            return dict(self._data)

    monkeypatch.setattr(es_mod, "EmbeddingStore", FakeEmbeddingStore)

    atomize_apply_impl(
        proposal_id=proposal["proposal_id"],
        vault_root=vault,
        pending_dir=pending_dir,
        applied_dir=applied_dir,
        graph_path=graph_path,
    )

    # Reload graph and check for KNOW node
    reloaded = KnowledgeGraph.load(graph_path)
    node_ids = list(reloaded.g.nodes())
    know_nodes = [nid for nid in node_ids if "KNOW-000001" in nid]
    assert len(know_nodes) >= 1, f"Expected KNOW-000001 node in graph; got nodes: {node_ids}"


def test_apply_with_graph_creates_similarity_edges(tmp_path, monkeypatch):
    """Applying with graph_path and similar fake vectors creates similarity edges.

    Two KNOW nodes + one existing note all share the same fake vector [1,0,0,0],
    so cosine similarity = 1.0 which is well above the default threshold (0.65).
    Similarity edges should be created between them.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    graph_path = data_dir / "graph.json"

    doc = vault / "source.md"
    doc.write_text("# Source\n\nContent for similarity test.", encoding="utf-8")

    scan_dir = tmp_path / "scan_cache"
    pending_dir = tmp_path / "pending"
    applied_dir = tmp_path / "applied"

    _build_minimal_graph(graph_path)

    scan = atomize_scan_impl(doc, vault_root=vault, scan_dir=scan_dir)
    sid = scan["sections"][0]["section_id"]
    proposal = atomize_propose_impl(
        scan_id=scan["scan_id"],
        claims=[
            _make_claim(sid, "KNOW-000001", title="Concept Alpha", body="Alpha body."),
            _make_claim(sid, "KNOW-000002", title="Concept Beta", body="Beta body."),
        ],
        scan_dir=scan_dir,
        proposal_dir=pending_dir,
    )

    fake_vector = [1.0, 0.0, 0.0, 0.0]

    def fake_compute_embeddings(graph, settings, **kwargs):
        return {nid: fake_vector for nid in graph.g.nodes()}

    monkeypatch.setattr("prism_rag.ingest.embedder.compute_embeddings", fake_compute_embeddings)

    # Fake embedding store that pre-seeds existing note with same vector
    from prism_rag.store import embedding_store as es_mod

    class FakeEmbeddingStore:
        def __init__(self, *a, **kw):
            # Pre-seed existing node with the same vector so similarity = 1.0
            self._data: dict[str, list[float]] = {
                "note/existing": fake_vector,
            }

        def upsert(self, node_id, vec):
            self._data[node_id] = vec

        def get(self, node_id):
            return self._data.get(node_id)

        def all_embeddings(self):
            return dict(self._data)

    monkeypatch.setattr(es_mod, "EmbeddingStore", FakeEmbeddingStore)

    atomize_apply_impl(
        proposal_id=proposal["proposal_id"],
        vault_root=vault,
        pending_dir=pending_dir,
        applied_dir=applied_dir,
        graph_path=graph_path,
    )

    # Reload graph and check similarity edges exist
    reloaded = KnowledgeGraph.load(graph_path)
    sim_edges = [
        (u, v)
        for u, v, d in reloaded.g.edges(data=True)
        if d.get("source_pass") == "embedding"
    ]
    # With 3 nodes all at cosine distance 0 from each other, we expect at least 1 similarity edge
    assert len(sim_edges) >= 1, (
        f"Expected similarity edges in graph but found none. "
        f"Edges: {list(reloaded.g.edges(data=True))}"
    )
