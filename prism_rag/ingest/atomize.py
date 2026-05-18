"""atomize_document — three-phase vault document atomization.

Phase 1: atomize_scan    — read structure, cache content snapshots server-side
Phase 2: atomize_propose — LLM submits claims; server writes proposal file
Phase 3: atomize_apply   — create knowledge/*.md, patch source doc, ingest

Content snapshots are stored server-side (scan_cache/<scan_id>.json) to avoid
returning 10k+ tokens to the LLM. Proposals live at
atomize-proposals/pending/<id>.json and move to applied/ on completion.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ScanExpiredError(Exception):
    """Raised when a scan_id is not found or is older than the TTL."""


class StaleDocError(Exception):
    """Raised when source document content has changed since the proposal was created."""


_SCAN_TTL_HOURS = 24
_HEADING_RE = re.compile(r"^(#{1,2})\s+(.*)", re.MULTILINE)


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_estimate(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(0, len(text) // 4)


def _parse_sections(content: str) -> list[dict[str, Any]]:
    """Split markdown content into sections at H1/H2 headings.

    Returns list of dicts with: section_id, heading, level, start_line, content_snapshot.
    Intro text before first heading is included as section_id=0.
    """
    lines = content.splitlines(keepends=True)
    sections: list[dict[str, Any]] = []
    current_heading = "(intro)"
    current_level = 0
    current_start = 0
    current_lines: list[str] = []

    def _flush(heading: str, level: int, start: int, body_lines: list[str], idx: int) -> None:
        text = "".join(body_lines).strip()
        if text or heading != "(intro)":
            sections.append({
                "section_id": str(idx),
                "heading": heading,
                "level": level,
                "start_line": start,
                "content_snapshot": text,
            })

    section_idx = 0
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            _flush(current_heading, current_level, current_start, current_lines, section_idx)
            section_idx += 1
            current_heading = m.group(2).strip()
            current_level = len(m.group(1))
            current_start = i
            current_lines = []
        else:
            current_lines.append(line)

    _flush(current_heading, current_level, current_start, current_lines, section_idx)
    return sections


def atomize_scan_impl(
    doc_path: Path,
    vault_root: Path,
    scan_dir: Path,
) -> dict[str, Any]:
    """Phase 1: read document structure, cache snapshots, return section metadata.

    Does NOT return content_snapshot to caller — only section headers and estimates.
    """
    doc_path = Path(doc_path).expanduser().resolve()
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    raw = doc_path.read_text(encoding="utf-8")
    doc_sha = _sha256(raw)
    scan_id = str(uuid.uuid4())
    sections_full = _parse_sections(raw)

    # Build what gets cached (includes content_snapshot)
    cached_sections = [
        {
            "section_id": s["section_id"],
            "heading": s["heading"],
            "level": s["level"],
            "start_line": s["start_line"],
            "content_snapshot": s["content_snapshot"],
            "token_estimate": _token_estimate(s["content_snapshot"]),
        }
        for s in sections_full
    ]

    # Persist to scan cache
    scan_dir = Path(scan_dir)
    scan_dir.mkdir(parents=True, exist_ok=True)
    cache_entry = {
        "scan_id": scan_id,
        "doc_path": str(doc_path),
        "doc_sha": doc_sha,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "sections": cached_sections,
    }
    (scan_dir / f"{scan_id}.json").write_text(
        json.dumps(cache_entry, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Return to caller: no content_snapshot
    public_sections = [
        {
            "section_id": s["section_id"],
            "heading": s["heading"],
            "level": s["level"],
            "start_line": s["start_line"],
            "token_estimate": s["token_estimate"],
        }
        for s in cached_sections
    ]

    return {
        "scan_id": scan_id,
        "doc_path": str(doc_path.relative_to(vault_root) if doc_path.is_relative_to(vault_root) else doc_path),
        "doc_sha": doc_sha,
        "section_count": len(public_sections),
        "sections": public_sections,
        "hint": (
            "Section content is NOT included here. "
            "Call read_note(doc_path) first to read the full document, "
            "then use this scan result to propose claims with atomize_propose."
        ),
    }


def atomize_propose_impl(
    scan_id: str,
    claims: list[dict[str, Any]],
    scan_dir: Path,
    proposal_dir: Path,
    *,
    embedding_store: Any = None,
    embedder: Any = None,
    knowledge_node_ids: "set[str] | None" = None,
    knowledge_node_labels: "dict[str, str] | None" = None,
    dedup_threshold: float = 0.90,
    min_nodes_for_dedup: int = 100,
) -> dict[str, Any]:
    """Phase 2: validate claims against scan cache, write proposal file.

    Args:
        scan_id: from atomize_scan result
        claims: list of claim dicts with keys: section_id, knowledge_id, title, body, ontology_type
        scan_dir: where scan cache lives
        proposal_dir: where to write pending proposals
        embedding_store: optional EmbeddingStore for ANN similarity search.
            When provided (and embedder is not None), claims are batch-embedded
            and compared against existing KNOW nodes. Claims with a similar existing
            node get ``similar_existing`` and ``claim_status='needs_review'`` set.
        embedder: optional OllamaEmbedder (or compatible) for embedding claim bodies.
        knowledge_node_ids: pre-computed set of node IDs with kind='knowledge' for
            post-filtering ANN results. Callers should extract this from the graph.
            When None, all ANN results are used (may include non-knowledge nodes).
        knowledge_node_labels: node_id → label lookup for populating similar_existing.
        dedup_threshold: cosine similarity above which a claim is flagged (0.90 default).
        min_nodes_for_dedup: skip dedup if embedding_store has fewer nodes than this
            (cold-start protection).

    Returns:
        dict with proposal_id and claim summary
    """
    from datetime import timedelta

    scan_dir = Path(scan_dir)
    cache_file = scan_dir / f"{scan_id}.json"
    if not cache_file.exists():
        raise ScanExpiredError(f"Scan {scan_id!r} not found in cache")

    cached = json.loads(cache_file.read_text(encoding="utf-8"))

    # Check TTL
    scanned_at = datetime.fromisoformat(cached["scanned_at"])
    age = datetime.now(timezone.utc) - scanned_at
    if age.total_seconds() > _SCAN_TTL_HOURS * 3600:
        raise ScanExpiredError(f"Scan {scan_id!r} expired ({age.total_seconds()/3600:.1f}h old)")

    # Build valid section_id set and snapshot lookup
    snapshot_map: dict[str, str] = {}
    for s in cached["sections"]:
        snapshot_map[s["section_id"]] = s.get("content_snapshot", "")

    # Deduplicate by knowledge_id (first wins)
    seen_kids: set[str] = set()
    validated_claims: list[dict[str, Any]] = []
    for claim in claims:
        kid = claim.get("knowledge_id", "")
        if kid in seen_kids:
            continue
        seen_kids.add(kid)
        sid = str(claim.get("section_id", ""))
        snapshot = snapshot_map.get(sid, "")
        validated_claims.append({
            "section_id": sid,
            "knowledge_id": kid,
            "title": claim.get("title", ""),
            "body": claim.get("body", ""),
            "ontology_type": claim.get("ontology_type", "concept"),
            "content_snapshot": snapshot,
            "claim_status": "pending",
        })

    # ── Semantic deduplication (optional) ────────────────────────────────────
    # Requires both embedding_store and embedder; skips gracefully if either is absent.
    # Cold-start protection: skip if fewer than min_nodes_for_dedup nodes are indexed.
    _dedup_active = (
        embedding_store is not None
        and embedder is not None
        and embedding_store.count() >= min_nodes_for_dedup
        and len(validated_claims) > 0
    )
    if _dedup_active:
        logger.info(
            f"[atomize/propose] running semantic dedup on {len(validated_claims)} claim(s) "
            f"(threshold={dedup_threshold})"
        )
        try:
            claim_bodies = [c["body"] for c in validated_claims]
            vectors: list[list[float]] = embedder.embed_batch(claim_bodies)
            for claim, vec in zip(validated_claims, vectors):
                # ANN search returns (node_id, distance); convert distance → similarity.
                raw_hits = embedding_store.search(vec, top_k=5)
                similar_existing: list[dict[str, Any]] = []
                for node_id, distance in raw_hits:
                    similarity = max(0.0, 1.0 - float(distance))
                    if similarity < dedup_threshold:
                        continue
                    # Post-filter: only knowledge nodes.
                    if knowledge_node_ids is not None and node_id not in knowledge_node_ids:
                        continue
                    # Skip self-match (same knowledge_id being reproposed).
                    if node_id == claim.get("knowledge_id"):
                        continue
                    label = (knowledge_node_labels or {}).get(node_id, node_id)
                    similar_existing.append({
                        "id": node_id,
                        "score": round(similarity, 4),
                        "title": label,
                    })
                if similar_existing:
                    claim["similar_existing"] = similar_existing
                    claim["claim_status"] = "needs_review"
                    logger.debug(
                        f"[atomize/propose] claim {claim['knowledge_id']!r} flagged "
                        f"({len(similar_existing)} similar node(s))"
                    )
                else:
                    claim["similar_existing"] = []
        except Exception as exc:
            logger.warning(f"[atomize/propose] dedup pass failed (skipped): {exc}")
    # ─────────────────────────────────────────────────────────────────────────

    proposal_id = str(uuid.uuid4())
    proposal_dir = Path(proposal_dir)
    proposal_dir.mkdir(parents=True, exist_ok=True)

    proposal = {
        "proposal_id": proposal_id,
        "scan_id": scan_id,
        "doc_path": cached["doc_path"],
        "doc_sha": cached["doc_sha"],
        "proposed_at": datetime.now(timezone.utc).isoformat(),
        "claims": validated_claims,
    }

    (proposal_dir / f"{proposal_id}.json").write_text(
        json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    claim_summary = []
    for c in validated_claims:
        entry: dict[str, Any] = {
            "knowledge_id": c["knowledge_id"],
            "title": c["title"],
            "claim_status": c["claim_status"],
        }
        if c.get("similar_existing"):
            entry["similar_existing"] = c["similar_existing"]
        claim_summary.append(entry)

    return {
        "proposal_id": proposal_id,
        "doc_path": cached["doc_path"],
        "doc_sha": cached["doc_sha"],
        "claim_count": len(validated_claims),
        "dedup_active": _dedup_active,
        "claims": claim_summary,
    }


# ---------------------------------------------------------------------------
# Phase 3: atomize_apply
# ---------------------------------------------------------------------------


def _write_knowledge_file(
    path: Path,
    knowledge_id: str,
    title: str,
    ontology_type: str,
    atomized_from: str,
    body: str,
) -> None:
    """Write a knowledge markdown file atomically (tmp + os.rename)."""
    import os
    import tempfile

    frontmatter_lines = [
        "---",
        f"knowledge_id: {knowledge_id}",
        f"title: {title}",
        f"ontology_type: {ontology_type}",
        f"atomized_from: {atomized_from}",
        "status: active",
        "---",
        body,
    ]
    content = "\n".join(frontmatter_lines) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: tmp file in same directory, then rename
    fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.rename(tmp_path_str, path)
    except Exception:
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise


def _patch_source_doc_atomized_nodes(doc_path: Path, knowledge_ids: list[str]) -> None:
    """Add/update atomized_nodes list in source document frontmatter using PyYAML."""
    from prism_rag.vault_ops.markdown_ops import parse_frontmatter, serialize_frontmatter

    content = doc_path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(content)
    fm["atomized_nodes"] = knowledge_ids
    new_content = serialize_frontmatter(fm, body)
    doc_path.write_text(new_content, encoding="utf-8")


def _read_existing_atomized_nodes(doc_path: Path) -> set[str]:
    """Return the set of KNOW-IDs already in the source doc's atomized_nodes frontmatter."""
    from prism_rag.vault_ops.markdown_ops import parse_frontmatter

    content = doc_path.read_text(encoding="utf-8")
    fm, _ = parse_frontmatter(content)
    existing = fm.get("atomized_nodes", [])
    if not isinstance(existing, list):
        return set()
    return {str(kid) for kid in existing if kid}


def atomize_apply_impl(
    proposal_id: str,
    vault_root: Path,
    pending_dir: Path,
    applied_dir: Path,
    graph_path: Path | None = None,
    dedup_log_path: Path | None = None,
) -> dict[str, Any]:
    """Phase 3: create knowledge files, patch source doc, move proposal to applied.

    Idempotent: if a knowledge_id is already in source doc's atomized_nodes, skip
    creating that file (crash recovery).
    Raises StaleDocError if source doc has changed since proposal was created.

    Args:
        proposal_id: UUID of the pending proposal.
        vault_root: Root of the knowledge vault.
        pending_dir: Directory holding pending proposal JSON files.
        applied_dir: Directory to move applied proposal JSON files into.
        graph_path: Optional path to graph.json; if provided, each new knowledge
                    file is incrementally ingested into the graph immediately after
                    creation (skip_embed=True, skip_leiden=True for speed).
        dedup_log_path: Optional path to dedup_log.jsonl. When provided, each
                        'reuse' claim writes a DedupSnapshot for auditability and
                        rollback support.
    """
    pending_dir = Path(pending_dir)
    applied_dir = Path(applied_dir)
    vault_root = Path(vault_root)

    proposal_file = pending_dir / f"{proposal_id}.json"
    if not proposal_file.exists():
        raise FileNotFoundError(f"Proposal {proposal_id!r} not found in pending")

    proposal = json.loads(proposal_file.read_text(encoding="utf-8"))

    # Resolve doc_path (may be relative to vault_root or absolute)
    doc_path_str = proposal["doc_path"]
    doc_path = Path(doc_path_str)
    if not doc_path.is_absolute():
        doc_path = vault_root / doc_path_str

    # Read existing atomized_nodes for idempotency (crash recovery)
    # Do this BEFORE stale-doc check so we can distinguish a true stale doc
    # from a doc that was already patched by a previous (possibly crashed) apply run.
    existing_atomized = _read_existing_atomized_nodes(doc_path)

    proposal_kids = {c["knowledge_id"] for c in proposal["claims"]}

    # Stale-doc detection: only raise if it's not a crash-recovery resume.
    # If all proposal KNOW-IDs are already in atomized_nodes, the doc was patched
    # by a prior apply run — this is idempotent re-apply, not a true stale doc.
    current_sha = _sha256(doc_path.read_text(encoding="utf-8"))
    if current_sha != proposal["doc_sha"]:
        all_already_applied = proposal_kids.issubset(existing_atomized)
        if not all_already_applied:
            raise StaleDocError(
                f"Source doc {doc_path_str!r} has changed since proposal was created. "
                "Re-run atomize_scan and atomize_propose."
            )

    # Determine atomized_from relative path for frontmatter
    try:
        atomized_from = str(doc_path.resolve().relative_to(vault_root.resolve()))
    except ValueError:
        atomized_from = str(doc_path)

    # Apply claims: write knowledge files, track all KNOW-IDs (existing + new)
    all_kids: list[str] = list(existing_atomized)
    reuse_claims: list[dict[str, Any]] = []  # collected for graph-level ops below

    for claim in proposal["claims"]:
        kid = claim["knowledge_id"]

        # ── Reuse path: existing node referenced, no new file needed ──────────
        if claim.get("action") == "reuse":
            reuse_id = claim.get("reuse_id", "")
            if not reuse_id:
                logger.warning(f"[atomize/apply] claim {kid!r} has action='reuse' but missing reuse_id; treating as create")
            else:
                claim["claim_status"] = "reused"
                reuse_claims.append(claim)
                logger.info(f"[atomize/apply] claim {kid!r} → reuse {reuse_id!r}")
                continue
        # ─────────────────────────────────────────────────────────────────────

        if kid in existing_atomized:
            # Already applied in a previous run — mark status and skip file write
            claim["claim_status"] = "applied"
            if kid not in all_kids:
                all_kids.append(kid)
            continue

        knowledge_file = vault_root / "knowledge" / f"{kid}.md"
        _write_knowledge_file(
            path=knowledge_file,
            knowledge_id=kid,
            title=claim.get("title", kid),
            ontology_type=claim.get("ontology_type", "concept"),
            atomized_from=atomized_from,
            body=claim.get("body", ""),
        )
        claim["claim_status"] = "applied"
        if kid not in all_kids:
            all_kids.append(kid)

    # Patch source doc atomized_nodes (idempotent — overwrites with full list)
    _patch_source_doc_atomized_nodes(doc_path, all_kids)

    # ── Write dedup snapshots for reuse decisions ─────────────────────────────
    if reuse_claims and dedup_log_path is not None:
        from datetime import timezone as _tz
        from prism_rag.ingest.dedup_log import DedupSnapshot, write_snapshot
        now_iso = datetime.now(_tz.utc).isoformat()
        for rc in reuse_claims:
            snap = DedupSnapshot(
                decision_id=str(uuid.uuid4()),
                timestamp=now_iso,
                action="reuse",
                claim_title=rc.get("title", rc.get("knowledge_id", "")),
                reused_id=rc.get("reuse_id"),
                source_doc=atomized_from,
                similarity_score=float(rc.get("similarity_score", 0.90)),
                pre_state={"mentions_edges": []},
            )
            try:
                write_snapshot(dedup_log_path, snap)
            except Exception as exc:
                logger.warning(f"[atomize/apply] failed to write dedup snapshot: {exc}")
    # ─────────────────────────────────────────────────────────────────────────

    # Incremental graph update for each new knowledge file
    if graph_path is not None:
        from prism_rag.config import PrismRagSettings
        from prism_rag.ingest.incremental import ingest_file
        _ingest_settings = PrismRagSettings()
        _ingest_settings.vault_path = vault_root
        _ingest_settings.data_dir = graph_path.parent
        for claim in proposal["claims"]:
            kid = claim["knowledge_id"]
            knowledge_file = vault_root / "knowledge" / f"{kid}.md"
            if knowledge_file.exists():
                try:
                    ingest_file(
                        knowledge_file,
                        settings=_ingest_settings,
                        skip_embed=True,
                        skip_leiden=True,
                    )
                except Exception as exc:
                    logger.warning(f"[atomize/apply] incremental ingest failed for {kid}: {exc}")

        # Embed new KNOW nodes in one batch, relink similarity edges, recluster.
        # Also handles reuse claims: MENTIONS edges + CONTEXT_REF nodes.
        try:
            from prism_rag.store.graph import KnowledgeGraph, Node, Edge
            from prism_rag.store.embedding_store import EmbeddingStore
            from prism_rag.ingest.embedder import compute_embeddings
            from prism_rag.ingest.similarity_linker import link_similar_nodes
            from prism_rag.cluster.leiden import run_leiden
            from prism_rag.report.graph_report import generate_report

            graph = KnowledgeGraph.load(_ingest_settings.graph_path)
            store = EmbeddingStore(_ingest_settings.embedding_cache_path, dim=_ingest_settings.embedding_dim)

            # ── Reuse claims: add MENTIONS edges + optional CONTEXT_REF nodes ──
            for rc in reuse_claims:
                reuse_id = rc.get("reuse_id", "")
                if not reuse_id:
                    continue
                if not graph.g.has_node(reuse_id):
                    logger.warning(
                        f"[atomize/apply] reuse target {reuse_id!r} not in graph — "
                        "MENTIONS edge not created (node may not be ingested yet)"
                    )
                    continue
                # MENTIONS edge: source_doc → reused KNOW node.
                graph.add_edge(Edge(
                    source=atomized_from,
                    target=reuse_id,
                    relation="mentions",
                    confidence="INFERRED",
                    confidence_score=float(rc.get("similarity_score", 0.90)),
                    source_pass="dedup",
                ))
                logger.info(
                    f"[atomize/apply] added MENTIONS edge {atomized_from!r} → {reuse_id!r}"
                )
                # Optional CONTEXT_REF node for additional context annotation.
                context_note = rc.get("context_note", "")
                if context_note:
                    ctx_id = f"CONTEXT_REF_{reuse_id}_{uuid.uuid4().hex[:8]}"
                    graph.add_node(Node(
                        id=ctx_id,
                        label=f"Context: {rc.get('title', reuse_id)}",
                        kind="context_ref",
                        source_file=atomized_from,
                        content=context_note,
                    ))
                    logger.info(f"[atomize/apply] added CONTEXT_REF node {ctx_id!r}")
            # ────────────────────────────────────────────────────────────────────

            new_kids = [c["knowledge_id"] for c in proposal["claims"] if c.get("claim_status") == "applied"]
            temp_graph = KnowledgeGraph()
            for kid in new_kids:
                if store.get(kid) is None and kid in graph.g:
                    node_data = graph.g.nodes[kid]
                    temp_graph.add_node(Node(
                        id=kid,
                        label=node_data.get("label", kid),
                        kind="knowledge",
                        content=node_data.get("content", ""),
                        tokens=node_data.get("tokens", 0),
                    ))

            if temp_graph.node_count > 0:
                vectors = compute_embeddings(temp_graph, _ingest_settings)
                for nid, vec in vectors.items():
                    store.upsert(nid, vec)
                logger.info(f"[atomize/apply] embedded {len(vectors)} new KNOW nodes")

            all_vectors = store.all_embeddings()
            stale_sim = [(u, v) for u, v, d in graph.g.edges(data=True) if d.get("source_pass") == "embedding"]
            for u, v in stale_sim:
                if graph.g.has_edge(u, v):
                    graph.g.remove_edge(u, v)
            link_similar_nodes(graph, all_vectors, _ingest_settings)

            graph.communities.clear()
            run_leiden(graph, resolution=_ingest_settings.leiden_resolution, seed=_ingest_settings.leiden_seed)
            graph.save(_ingest_settings.graph_path)
            try:
                generate_report(graph, _ingest_settings.report_path, vault_root=vault_root)
            except Exception:
                pass
        except Exception as exc:
            logger.warning(f"[atomize/apply] post-apply embedding pass failed: {exc}")

    # Move proposal from pending/ to applied/
    applied_dir.mkdir(parents=True, exist_ok=True)
    proposal_file.rename(applied_dir / f"{proposal_id}.json")

    applied_count = len([c for c in proposal["claims"] if c.get("claim_status") == "applied"])
    reused_count = len([c for c in proposal["claims"] if c.get("claim_status") == "reused"])
    return {
        "proposal_id": proposal_id,
        "applied_count": applied_count,
        "reused_count": reused_count,
        "knowledge_files": [
            f"knowledge/{c['knowledge_id']}.md"
            for c in proposal["claims"]
            if c.get("claim_status") == "applied"
        ],
        "reused_nodes": [
            c.get("reuse_id") for c in proposal["claims"]
            if c.get("claim_status") == "reused"
        ],
        "doc_patched": doc_path_str,
    }
