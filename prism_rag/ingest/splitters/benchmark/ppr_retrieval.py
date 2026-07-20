"""Personalized PageRank retrieval on Atom-Entity bipartite graph.

Implements the AtomicRAG-style PPR retrieval where atoms (propositions) and
entities form a bipartite graph. Query entities seed the PPR walk, which
propagates relevance through shared entities to discover multi-hop connections.

Usage::

    from prism_rag.ingest.splitters.benchmark.ppr_retrieval import (
        AtomEntityGraph, build_atom_entity_graph, retrieve_ppr,
        extract_query_entities,
    )
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field

import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AtomEntityGraph:
    """Atom-Entity bipartite graph for PPR retrieval."""

    graph: nx.Graph  # undirected bipartite graph
    atom_nodes: list[str]  # atom node IDs (a_0, a_1, ...)
    entity_nodes: list[str]  # entity node IDs (e_<name>)
    atom_texts: dict[str, str]  # atom_id -> text
    atom_vectors: dict[str, list[float]]  # atom_id -> embedding
    entity_to_atoms: dict[str, list[str]]  # entity -> list of atom_ids


# ---------------------------------------------------------------------------
# Cosine similarity (local copy)
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_atom_entity_graph(
    l0_texts: list[str],
    l0_entities: list[list[str]],
    l0_vectors: list[list[float]],
    synonym_threshold: float = 0.8,
    embedder_fn=None,
) -> AtomEntityGraph:
    """Build Atom-Entity bipartite graph (AtomicRAG style).

    Nodes: atoms (a_0, a_1, ...) + entities (e_<name>)
    Edges:
      - Containment: atom <-> entity it mentions (weight=1.0)
      - Synonym: entity <-> entity if embedding cosine > threshold (weight=cosine)

    Entity names are normalized to lowercase.

    Args:
        l0_texts: Atom proposition texts.
        l0_entities: Entities per atom (already lowercase-normalized).
        l0_vectors: Embedding vectors per atom.
        synonym_threshold: Cosine threshold for entity synonym edges.
        embedder_fn: Optional embedder for entity synonym detection.

    Returns:
        AtomEntityGraph with all nodes and edges.
    """
    G = nx.Graph()

    atom_nodes: list[str] = []
    atom_texts: dict[str, str] = {}
    atom_vectors: dict[str, list[float]] = {}
    entity_to_atoms: dict[str, list[str]] = {}

    # Collect all unique entities
    all_entities: set[str] = set()
    for ents in l0_entities:
        all_entities.update(ents)

    # Add atom nodes
    for i, text in enumerate(l0_texts):
        aid = f"a_{i}"
        atom_nodes.append(aid)
        atom_texts[aid] = text
        if i < len(l0_vectors):
            atom_vectors[aid] = l0_vectors[i]
        G.add_node(aid, bipartite=0, kind="atom")

    # Add entity nodes
    entity_nodes: list[str] = []
    entity_id_map: dict[str, str] = {}  # normalized name -> node id
    for ent in sorted(all_entities):
        eid = f"e_{ent}"
        entity_nodes.append(eid)
        entity_id_map[ent] = eid
        G.add_node(eid, bipartite=1, kind="entity")

    # Add containment edges (atom <-> entity)
    for i, ents in enumerate(l0_entities):
        aid = f"a_{i}"
        for ent in ents:
            eid = entity_id_map.get(ent)
            if eid:
                G.add_edge(aid, eid, weight=1.0, edge_type="containment")
                entity_to_atoms.setdefault(ent, []).append(aid)

    # Add synonym edges (entity <-> entity) if embedder available
    if embedder_fn is not None and len(all_entities) > 1:
        ent_list = sorted(all_entities)
        try:
            ent_vecs = embedder_fn(ent_list)
            for i in range(len(ent_list)):
                for j in range(i + 1, len(ent_list)):
                    sim = _cosine(ent_vecs[i], ent_vecs[j])
                    if sim > synonym_threshold:
                        eid_i = entity_id_map[ent_list[i]]
                        eid_j = entity_id_map[ent_list[j]]
                        G.add_edge(eid_i, eid_j, weight=sim, edge_type="synonym")
        except Exception as exc:
            logger.warning("Entity synonym embedding failed: %s", exc)

    logger.info(
        "AEG: %d atoms, %d entities, %d edges",
        len(atom_nodes), len(entity_nodes), G.number_of_edges(),
    )

    return AtomEntityGraph(
        graph=G,
        atom_nodes=atom_nodes,
        entity_nodes=entity_nodes,
        atom_texts=atom_texts,
        atom_vectors=atom_vectors,
        entity_to_atoms=entity_to_atoms,
    )


# ---------------------------------------------------------------------------
# PPR retrieval
# ---------------------------------------------------------------------------


def retrieve_ppr(
    query: str,
    query_entities: list[str],
    query_vector: list[float],
    aeg: AtomEntityGraph,
    damping: float = 0.3,
    atom_seed_weight: float = 0.1,
    top_k: int = 25,
) -> list[str]:
    """Personalized PageRank retrieval on Atom-Entity graph.

    1. Build seed dict:
       - Entity seeds: matching entity nodes -> weight 1.0
       - Atom seeds: top-5 atoms by embedding cosine to query -> weight atom_seed_weight
    2. Run nx.pagerank(G, alpha=1-damping, personalization=seed_dict)
       (NetworkX alpha = 1 - AtomicRAG rho, so alpha=0.7 when damping=0.3)
    3. Rank atoms by PPR score, return top-k atom texts.

    Entity matching: case-insensitive exact match on normalized names.

    Args:
        query: Query text (for logging).
        query_entities: Entities extracted from query.
        query_vector: Query embedding vector.
        aeg: The Atom-Entity graph.
        damping: PPR damping factor (AtomicRAG rho). Higher = more exploration.
        atom_seed_weight: Weight for embedding-based atom seeds.
        top_k: Number of atoms to return.

    Returns:
        List of atom texts ranked by PPR score.
    """
    if not aeg.atom_nodes:
        return []

    # Build seed (personalization) dict
    seed: dict[str, float] = {}

    # Entity seeds: match query entities to graph entity nodes
    normalized_query_ents = [e.strip().lower() for e in query_entities]
    for qe in normalized_query_ents:
        eid = f"e_{qe}"
        if eid in aeg.graph:
            seed[eid] = 1.0

    # Atom seeds: top-5 atoms by embedding cosine to query
    if query_vector and aeg.atom_vectors:
        atom_sims: list[tuple[str, float]] = []
        for aid, vec in aeg.atom_vectors.items():
            sim = _cosine(query_vector, vec)
            atom_sims.append((aid, sim))
        atom_sims.sort(key=lambda x: x[1], reverse=True)
        for aid, sim in atom_sims[:5]:
            seed[aid] = atom_seed_weight

    if not seed:
        # Fallback: uniform over all atoms
        for aid in aeg.atom_nodes:
            seed[aid] = 1.0 / len(aeg.atom_nodes)

    # Ensure all graph nodes have a personalization value (0 for non-seeds)
    full_personalization: dict[str, float] = {}
    for node in aeg.graph.nodes():
        full_personalization[node] = seed.get(node, 0.0)

    # Normalize personalization to sum to 1
    total = sum(full_personalization.values())
    if total > 0:
        for k in full_personalization:
            full_personalization[k] /= total

    # Run PPR: NetworkX alpha = probability of following edges = 1 - damping
    alpha = 1.0 - damping
    try:
        ppr_scores = nx.pagerank(
            aeg.graph,
            alpha=alpha,
            personalization=full_personalization,
            weight="weight",
            max_iter=100,
            tol=1e-06,
        )
    except nx.PowerIterationFailedConvergence:
        logger.warning("PPR did not converge; falling back to cosine")
        # Fallback: return top-k by cosine
        atom_sims = []
        for aid, vec in aeg.atom_vectors.items():
            sim = _cosine(query_vector, vec)
            atom_sims.append((aid, sim))
        atom_sims.sort(key=lambda x: x[1], reverse=True)
        return [aeg.atom_texts[aid] for aid, _ in atom_sims[:top_k]]

    # Rank atoms only (not entities) by PPR score
    atom_scores = [
        (aid, ppr_scores.get(aid, 0.0))
        for aid in aeg.atom_nodes
    ]
    atom_scores.sort(key=lambda x: x[1], reverse=True)

    return [aeg.atom_texts[aid] for aid, _ in atom_scores[:top_k]]


# ---------------------------------------------------------------------------
# PPR retrieval returning L1 groups
# ---------------------------------------------------------------------------


def retrieve_ppr_l1(
    query: str,
    query_entities: list[str],
    query_vector: list[float],
    aeg: AtomEntityGraph,
    l1_members: list[list[int]],
    l1_texts: list[str],
    damping: float = 0.3,
    atom_seed_weight: float = 0.1,
    top_k: int = 10,
) -> list[str]:
    """PPR retrieval returning L1 group texts (hybrid: PPR atoms -> L1 parent).

    Runs PPR to rank atoms, then maps top atoms back to their L1 groups,
    deduplicates, and returns top-k L1 group texts.

    Args:
        l1_members: L1 group membership (list of L0 index lists).
        l1_texts: L1 group texts.
        Other args same as retrieve_ppr.

    Returns:
        List of L1 group texts ranked by best atom PPR score in the group.
    """
    if not aeg.atom_nodes:
        return []

    # Get more atoms than needed to fill L1 groups
    ppr_atoms = retrieve_ppr(
        query, query_entities, query_vector, aeg,
        damping=damping, atom_seed_weight=atom_seed_weight,
        top_k=top_k * 3,
    )

    # Build L0 index -> L1 index mapping
    l0_to_l1: dict[int, int] = {}
    for l1_idx, members in enumerate(l1_members):
        for l0_idx in members:
            l0_to_l1[l0_idx] = l1_idx

    # Map PPR-ranked atom texts -> L0 indices -> L1 groups
    # Maintain order from PPR ranking
    seen_l1: set[int] = set()
    results: list[str] = []

    # Build reverse lookup: text -> L0 index
    text_to_l0: dict[str, int] = {}
    for aid, text in aeg.atom_texts.items():
        l0_idx = int(aid.split("_")[1])
        text_to_l0[text] = l0_idx

    for atom_text in ppr_atoms:
        l0_idx = text_to_l0.get(atom_text)
        if l0_idx is None:
            continue
        l1_idx = l0_to_l1.get(l0_idx)
        if l1_idx is not None and l1_idx not in seen_l1:
            seen_l1.add(l1_idx)
            results.append(l1_texts[l1_idx])
            if len(results) >= top_k:
                break

    return results


# ---------------------------------------------------------------------------
# Query entity extraction
# ---------------------------------------------------------------------------

_ENTITY_EXTRACTION_PROMPT = """\
Extract key entities (people, places, concepts, tools, technical terms) from this question. Return ONLY a JSON array of strings, nothing else.

Question: {query}"""


def extract_query_entities(query: str, llm_fn) -> list[str]:
    """Use LLM to extract entities from query.

    Args:
        query: The question text.
        llm_fn: Function(str) -> str for LLM calls.

    Returns:
        List of entity strings (lowercase normalized).
    """
    prompt = _ENTITY_EXTRACTION_PROMPT.format(query=query)
    try:
        raw = llm_fn(prompt)
        # Strip thinking tags
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Find JSON array
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            entities = json.loads(match.group())
            if isinstance(entities, list):
                return [e.strip().lower() for e in entities if isinstance(e, str) and e.strip()]
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Entity extraction failed: %s", exc)

    # Fallback: extract capitalized words
    words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
    return [w.lower() for w in words]
