"""Incremental ingest: add or update a single file in the existing graph.

Usage:
    from prism_rag.ingest.incremental import ingest_file
    ingest_file(Path("设计细节/new_doc.md"))

What it does:
1. Parse the file (frontmatter + AST extraction)
2. Embed the file (Gemini Embedding 2)
3. Generate similarity edges (top-K cosine vs existing nodes)
4. Re-run Leiden on the full graph
5. Persist graph.json + GRAPH_REPORT.md

Typical latency: ~2s (dominated by embedding API call)
"""

from __future__ import annotations

import logging
from pathlib import Path

from prism_rag.config import PrismRagSettings
from prism_rag.ingest.ast_extractor import extract_ast
from prism_rag.ingest.vault_loader import VaultDocument
from prism_rag.report.graph_report import generate_report
from prism_rag.store.graph import Edge, KnowledgeGraph, Node

logger = logging.getLogger(__name__)


def _remove_node_and_edges(graph: KnowledgeGraph, node_id: str) -> None:
    """Remove a node and all its edges from the graph (for re-indexing)."""
    if node_id in graph.g:
        graph.g.remove_node(node_id)


def _add_single_doc_ast(
    graph: KnowledgeGraph,
    doc: VaultDocument,
    all_docs: list[VaultDocument] | None = None,
) -> None:
    """Add a single document's AST-extracted nodes and edges to the graph.

    If all_docs is provided, wikilinks can resolve against the full vault.
    Otherwise, wikilinks resolve against existing graph nodes.
    """
    from prism_rag.ingest.ast_extractor import (
        _extract_inline_tags,
        _extract_relations_edges,
        _extract_wikilinks,
        _tag_node_id,
        _category_node_id,
        _token_count,
    )

    # Build doc_index from existing graph nodes + all_docs if provided
    doc_index: dict[str, str] = {}
    for nid, data in graph.g.nodes(data=True):
        if data.get("kind") in ("note", "knowledge"):
            label = data.get("label", "")
            if label:
                doc_index[label.lower()] = nid
            # Also index aliases and knowledge_id
            fm = data.get("frontmatter", {})
            for alias in fm.get("aliases", []):
                doc_index[str(alias).lower()] = nid
            kid = fm.get("knowledge_id")
            if kid:
                doc_index[str(kid).lower()] = nid

    if all_docs:
        for d in all_docs:
            doc_index[d.label.lower()] = d.id
            for alias in d.aliases:
                doc_index[alias.lower()] = d.id
            kid = d.frontmatter.get("knowledge_id")
            if kid:
                doc_index[str(kid).lower()] = d.id

    # Determine node kind: 'knowledge' if knowledge_id frontmatter present, else 'note'
    kind = "knowledge" if doc.frontmatter.get("knowledge_id") else "note"

    # Also index this doc itself so relations can resolve to it
    doc_index[doc.label.lower()] = doc.id
    doc_index[doc.id.lower()] = doc.id

    # Extract Am attributes and ontology_type from frontmatter (mirrors extract_ast)
    fm = doc.frontmatter
    _maturity = fm.get("maturity")
    _confidence = fm.get("confidence")
    _actionability = fm.get("actionability")
    fm_type = fm.get("type")

    _VALID_ONT = {
        "concept", "entity", "process", "tool", "project",
        "fact", "decision", "rule", "procedure", "relation",
        "unclassified",
    }
    if fm_type is None:
        _ont = None
    elif fm_type in _VALID_ONT:
        _ont = fm_type
    else:
        _ont = "unclassified"

    # Add note/knowledge node
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
    graph.add_node(note)

    # Wikilinks
    for target, relation in _extract_wikilinks(doc.content):
        resolved = doc_index.get(target.lower())
        if resolved and resolved != doc.id:
            graph.add_edge(Edge(
                source=doc.id, target=resolved, relation=relation,
                confidence="EXTRACTED", confidence_score=1.0, weight=1.0, source_pass="ast",
            ))

    # Tags
    inline_tags = _extract_inline_tags(doc.content)
    all_tags = set(doc.frontmatter_tags) | inline_tags
    for tag in all_tags:
        tid = _tag_node_id(tag)
        if tid not in graph.g:
            graph.add_node(Node(id=tid, label=f"#{tag}", kind="tag"))
        graph.add_edge(Edge(
            source=doc.id, target=tid, relation="tagged_as",
            confidence="EXTRACTED", confidence_score=1.0, weight=1.0, source_pass="ast",
        ))

    # Category
    if doc.category:
        cid = _category_node_id(doc.category)
        if cid not in graph.g:
            graph.add_node(Node(id=cid, label=doc.category, kind="category"))
        graph.add_edge(Edge(
            source=doc.id, target=cid, relation="categorized_as",
            confidence="EXTRACTED", confidence_score=1.0, weight=1.0, source_pass="ast",
        ))

    # Relations frontmatter (Phase 2 explicit typed edges)
    _extract_relations_edges(graph, doc, doc_index)


def ingest_file(
    file_path: Path,
    settings: PrismRagSettings | None = None,
    skip_embed: bool = False,
    skip_leiden: bool = False,
    skip_persist: bool = False,
) -> dict:
    """Incrementally add or update a single file in the knowledge graph.

    Args:
        file_path: Absolute or vault-relative path to the .md file.
        settings: PrismRag settings (loads from env if None).
        skip_embed: Skip embedding (faster, but no similarity edges for new file).
        skip_leiden: Skip Leiden re-clustering.
        skip_persist: Don't save graph.json / report (useful for batch).

    Returns:
        dict with stats: {node_id, new_edges, total_nodes, total_edges, communities}
    """
    settings = settings or PrismRagSettings()
    vault_root = settings.vault_path.expanduser().resolve()
    graph_path = settings.graph_path

    # Resolve file path
    fp = Path(file_path).expanduser()
    if not fp.is_absolute():
        fp = vault_root / fp
    fp = fp.resolve()

    if not fp.exists():
        raise FileNotFoundError(f"File not found: {fp}")
    if not fp.suffix == ".md":
        raise ValueError(f"Not a markdown file: {fp}")
    if not str(fp).startswith(str(vault_root)):
        raise ValueError(f"File is outside vault: {fp} (vault={vault_root})")

    # Load existing graph (or create empty)
    if graph_path.exists():
        graph = KnowledgeGraph.load(graph_path)
        logger.info(f"[incremental] loaded existing graph: {graph.node_count} nodes, {graph.edge_count} edges")
    else:
        graph = KnowledgeGraph()
        logger.info("[incremental] no existing graph, starting fresh")

    # Parse the file
    doc = VaultDocument.from_path(fp, vault_root)
    logger.info(f"[incremental] processing: {doc.id} ({doc.label})")

    # Remove old version of this node (if updating)
    old_existed = doc.id in graph.g
    if old_existed:
        _remove_node_and_edges(graph, doc.id)
        logger.info(f"[incremental] removed old version of {doc.id}")

        # Remove old embedding from LanceDB
        try:
            from prism_rag.store.embedding_store import EmbeddingStore
            store = EmbeddingStore(settings.embedding_cache_path)
            store.delete(doc.id)
        except Exception:
            pass  # LanceDB may not exist yet

    # Add AST-extracted nodes and edges
    edges_before = graph.edge_count
    _add_single_doc_ast(graph, doc)
    ast_edges = graph.edge_count - edges_before

    # Embedding + similarity edges
    embed_edges = 0
    if not skip_embed and settings.gemini_api_key:
        from prism_rag.ingest.embedder import compute_embeddings
        from prism_rag.ingest.similarity_linker import link_similar_nodes

        # Only embed the new file
        temp_graph = KnowledgeGraph()
        temp_graph.add_node(Node(
            id=doc.id, label=doc.label, kind="note",
            content=doc.content, tokens=0,
        ))
        vectors = compute_embeddings(temp_graph, settings)

        if vectors:
            new_vec = vectors.get(doc.id)
            if new_vec:
                # Persist new embedding to LanceDB
                from prism_rag.store.embedding_store import EmbeddingStore
                store = EmbeddingStore(settings.embedding_cache_path)
                store.upsert(doc.id, new_vec)

                # Load existing embeddings from LanceDB (cached, no API calls)
                all_vectors = store.all_embeddings()

                edges_before_sim = graph.edge_count
                link_similar_nodes(graph, all_vectors, settings)
                embed_edges = graph.edge_count - edges_before_sim

    # Leiden re-clustering
    if not skip_leiden:
        from prism_rag.cluster.leiden import run_leiden
        graph.communities.clear()
        run_leiden(graph, resolution=settings.leiden_resolution, seed=settings.leiden_seed)

    # Persist
    if not skip_persist:
        graph.save(graph_path)
        generate_report(graph, settings.report_path, vault_root=vault_root)

        try:
            from prism_rag.report.visualize import generate_html
            generate_html(graph, settings.data_dir / "graph.html")
        except ImportError:
            pass

    result = {
        "node_id": doc.id,
        "action": "updated" if old_existed else "added",
        "ast_edges": ast_edges,
        "similarity_edges": embed_edges,
        "total_nodes": graph.node_count,
        "total_edges": graph.edge_count,
        "communities": len(graph.communities),
    }
    logger.info(f"[incremental] done: {result}")
    return result
