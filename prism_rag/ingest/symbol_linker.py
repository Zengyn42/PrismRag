"""SymbolLinker — build mentions_symbol cross-namespace edges.

Scans vault note content for code symbol names and writes
mentions_symbol edges back into the vault graph.json.

Matching levels (v5.1a — deterministic only):
  L1a  wikilink [[Symbol]]            → EXTRACTED / 1.0
  L1b  word-boundary regex (unique)   → INFERRED  / 0.9
  L1b' word-boundary regex (ambiguous)→ AMBIGUOUS / 0.2  (all candidates)

L2 (BM25 + graph expansion + LLM confirmation) is deferred to v5.1b.

Design doc: NimbusVault/design_details/PrismRag v5.1 — mentions_symbol cross-namespace link design.md
GitHub: https://github.com/Zengyn42/PrismRag/issues/10
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from prism_rag.store.graph import Edge, KnowledgeGraph, LifecycleClass

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_MIN_SYMBOL_LEN = 5

_BUILTIN_BLACKLIST: frozenset[str] = frozenset({
    # Python builtins and keywords ≥5 chars that are too generic
    "False", "None", "True", "class", "async", "yield", "raise",
    "while", "break", "import", "return", "lambda", "assert",
    "global", "except", "finally", "continue", "delete",
    "open", "type", "list", "dict", "tuple", "set", "str",
    "int", "float", "bool", "bytes", "iter", "next", "sorted",
    "range", "print", "super", "input", "format", "round",
    "enumerate", "filter", "hasattr", "getattr", "setattr", "delattr",
    "isinstance", "issubclass", "property", "staticmethod", "classmethod",
    "NotImplemented", "Ellipsis",
    # Common short identifiers that appear everywhere
    "value", "error", "index", "count", "items", "keys", "values",
    "result", "output", "logger", "logging", "config", "settings",
    "data", "text", "path", "name", "label", "model",
})

_CODE_KINDS: frozenset[str] = frozenset({"function", "class", "module"})

_WIKILINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]")
_RELATION_MENTIONS = "mentions_symbol"


# ── Core helpers ───────────────────────────────────────────────────────────────

def _build_symbol_dict(code_graph: KnowledgeGraph) -> dict[str, list[str]]:
    """Return {short_label → [qualified_node_id, ...]} for all code symbols."""
    sym: dict[str, list[str]] = {}
    for node_id, data in code_graph.g.nodes(data=True):
        if data.get("kind") not in _CODE_KINDS:
            continue
        if data.get("namespace", "") != "code":
            continue
        label = data.get("label", "")
        if not label:
            continue
        if len(label) < _MIN_SYMBOL_LEN:
            continue
        if label in _BUILTIN_BLACKLIST:
            continue
        sym.setdefault(label, []).append(node_id)
    return sym


def _scan_wikilinks(content: str) -> set[str]:
    """Extract symbol names referenced as [[wikilinks]] in content."""
    return {m.group(1).strip() for m in _WIKILINK_RE.finditer(content)}


def _clear_existing_mentions(graph) -> None:
    """Remove all non-ANCHORED mentions_symbol edges.

    SymbolLinker owns DETERMINISTIC mentions_symbol edges (full-overwrite on each run).
    ANCHORED edges (created by EdgeClassifier Tier 1 or human approval) are preserved.
    """
    for u, v, d in list(graph.edges(data=True)):
        if d.get("relation") != "mentions_symbol":
            continue
        if d.get("lifecycle_class") == LifecycleClass.ANCHORED:
            continue
        graph.remove_edge(u, v)


# ── Main linker ────────────────────────────────────────────────────────────────

def link_symbols(
    vault_graph: KnowledgeGraph,
    code_graph: KnowledgeGraph,
) -> tuple[int, int, int]:
    """Build mentions_symbol edges from vault notes to code nodes.

    Mutates vault_graph in place (removes stale edges, adds new ones).

    Returns:
        (n_extracted, n_inferred, n_ambiguous) edge counts.
    """
    # Idempotency: remove all non-ANCHORED mentions_symbol edges before rewriting.
    # ANCHORED edges (human-approved / Tier-1 promoted) are preserved.
    before = vault_graph.g.number_of_edges()
    _clear_existing_mentions(vault_graph.g)
    removed = before - vault_graph.g.number_of_edges()
    logger.debug(f"[symbol_linker] removed {removed} stale mentions_symbol edges")

    sym_dict = _build_symbol_dict(code_graph)
    if not sym_dict:
        logger.info("[symbol_linker] no code symbols found (code graph empty?)")
        return 0, 0, 0

    # Sort symbols by length descending: longer names match before shorter ones
    # to avoid a shorter alias shadowing a longer match in the same content.
    all_symbols = sorted(sym_dict.keys(), key=len, reverse=True)

    n_extracted = n_inferred = n_ambiguous = 0

    # Snapshot to avoid "dict changed size during iteration" when add_edge creates stubs
    vault_nodes = list(vault_graph.g.nodes(data=True))

    for node_id, data in vault_nodes:
        content = data.get("content", "")
        if not content:
            continue

        # Track targets already linked from this source to avoid duplicates
        seen: set[str] = set()

        # ── L1a: wikilinks → EXTRACTED ────────────────────────────────────
        for wl_name in _scan_wikilinks(content):
            candidates = sym_dict.get(wl_name, [])
            if not candidates:
                continue
            tier, score = ("EXTRACTED", 1.0) if len(candidates) == 1 else ("AMBIGUOUS", 0.2)
            for target in candidates:
                if target not in seen:
                    seen.add(target)
                    vault_graph.add_edge(Edge(
                        source=node_id, target=target,
                        relation=_RELATION_MENTIONS,
                        confidence=tier,
                        confidence_score=score,
                        weight=1.0, source_pass="ast",
                        lifecycle_class=LifecycleClass.DETERMINISTIC,
                    ))
                    if tier == "EXTRACTED":
                        n_extracted += 1
                    else:
                        n_ambiguous += 1

        # ── L1b: word-boundary regex ───────────────────────────────────────
        for sym_name in all_symbols:
            candidates = sym_dict[sym_name]
            pattern = r"\b" + re.escape(sym_name) + r"\b"
            if not re.search(pattern, content):
                continue
            occurrences = len(re.findall(pattern, content))
            tier, score = ("INFERRED", 0.9) if len(candidates) == 1 else ("AMBIGUOUS", 0.2)
            for target in candidates:
                if target not in seen:
                    seen.add(target)
                    vault_graph.add_edge(Edge(
                        source=node_id, target=target,
                        relation=_RELATION_MENTIONS,
                        confidence=tier,
                        confidence_score=score,
                        weight=min(1.0, 0.5 + occurrences * 0.1),
                        source_pass="ast",
                        lifecycle_class=LifecycleClass.DETERMINISTIC,
                    ))
                    if tier == "INFERRED":
                        n_inferred += 1
                    else:
                        n_ambiguous += 1

    logger.info(
        f"[symbol_linker] done — EXTRACTED={n_extracted}, "
        f"INFERRED={n_inferred}, AMBIGUOUS={n_ambiguous}"
    )
    return n_extracted, n_inferred, n_ambiguous


# ── Stale ref detection ────────────────────────────────────────────────────────

def mark_stale_refs(
    vault_graph: KnowledgeGraph,
    changed_code_ids: set[str],
    changed_at: str,
) -> int:
    """Mark vault nodes that mention changed code symbols as stale.

    For each vault node with a mentions_symbol edge pointing to a changed
    code node, appends a stale_refs entry into the node's frontmatter dict
    in the graph (does NOT write to the .md file on disk).

    Args:
        vault_graph: Vault graph, already containing mentions_symbol edges.
        changed_code_ids: Set of code node IDs whose content_hash changed.
        changed_at: ISO date string for the change timestamp.

    Returns:
        Number of vault nodes marked as stale.
    """
    marked = 0
    for source, target, data in vault_graph.g.edges(data=True):
        if data.get("relation") != _RELATION_MENTIONS:
            continue
        if target not in changed_code_ids:
            continue
        node_data = vault_graph.g.nodes.get(source)
        if not node_data:
            continue
        fm = node_data.get("frontmatter") or {}
        stale_refs: list[dict] = fm.get("stale_refs", [])
        # Avoid duplicate entries for the same target
        existing = {r.get("code_node") for r in stale_refs}
        if target not in existing:
            stale_refs.append({
                "symbol": target.split("::")[-1],
                "changed_at": changed_at,
                "code_node": target,
            })
            fm["stale_refs"] = stale_refs
            vault_graph.g.nodes[source]["frontmatter"] = fm
            marked += 1
    if marked:
        logger.info(f"[symbol_linker] marked {marked} vault nodes as stale")
    return marked


# ── Public entry point ─────────────────────────────────────────────────────────

def run_link_symbols(
    vault_graph_path: Path,
    code_graph_path: Path,
) -> tuple[int, int, int]:
    """Load graphs, run linker, write back to vault graph.json.

    Args:
        vault_graph_path: Path to vault graph.json.
        code_graph_path:  Path to code graph.json.

    Returns:
        (n_extracted, n_inferred, n_ambiguous)
    """
    vault_graph = KnowledgeGraph.load(vault_graph_path)
    code_graph = KnowledgeGraph.load(code_graph_path)
    counts = link_symbols(vault_graph, code_graph)
    vault_graph.save(vault_graph_path)
    logger.info(f"[symbol_linker] saved → {vault_graph_path}")
    return counts
