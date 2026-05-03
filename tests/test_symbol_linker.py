"""Tests for SymbolLinker — mentions_symbol cross-namespace edge builder."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from prism_rag.ingest.symbol_linker import (
    _build_symbol_dict,
    _scan_wikilinks,
    link_symbols,
    mark_stale_refs,
    run_link_symbols,
)
from prism_rag.store.graph import Edge, KnowledgeGraph, Node


# ── Helpers ───────────────────────────────────────────────────────────────────

def _code_graph(*symbols: tuple[str, str, str]) -> KnowledgeGraph:
    """Build a code graph with (node_id, label, kind) triples."""
    kg = KnowledgeGraph()
    for node_id, label, kind in symbols:
        kg.add_node(Node(id=node_id, label=label, kind=kind, namespace="code",
                         source_file="framework/x.py"))
    return kg


def _vault_graph(*notes: tuple[str, str, str]) -> KnowledgeGraph:
    """Build a vault graph with (node_id, label, content) triples."""
    kg = KnowledgeGraph()
    for node_id, label, content in notes:
        kg.add_node(Node(id=node_id, label=label, kind="note",
                         namespace="nimbus", content=content))
    return kg


# ── _build_symbol_dict ────────────────────────────────────────────────────────

def test_symbol_dict_basic():
    cg = _code_graph(
        ("code::m.py::build_graph", "build_graph", "function"),
        ("code::m.py::MyClass",     "MyClass",     "class"),
    )
    sym = _build_symbol_dict(cg)
    assert "build_graph" in sym
    assert "MyClass" in sym
    assert sym["build_graph"] == ["code::m.py::build_graph"]


def test_symbol_dict_skips_short():
    cg = _code_graph(("code::m.py::run", "run", "function"))
    sym = _build_symbol_dict(cg)
    assert "run" not in sym


def test_symbol_dict_skips_blacklisted():
    cg = _code_graph(("code::m.py::logger", "logger", "function"))
    sym = _build_symbol_dict(cg)
    assert "logger" not in sym


def test_symbol_dict_ambiguous():
    cg = _code_graph(
        ("code::a.py::parse_node", "parse_node", "function"),
        ("code::b.py::parse_node", "parse_node", "function"),
    )
    sym = _build_symbol_dict(cg)
    assert len(sym["parse_node"]) == 2


# ── _scan_wikilinks ───────────────────────────────────────────────────────────

def test_scan_wikilinks_basic():
    links = _scan_wikilinks("See [[build_graph]] for details.")
    assert "build_graph" in links


def test_scan_wikilinks_with_alias():
    links = _scan_wikilinks("See [[build_graph|the builder]].")
    assert "build_graph" in links


def test_scan_wikilinks_empty():
    assert _scan_wikilinks("no links here") == set()


# ── link_symbols ──────────────────────────────────────────────────────────────

def test_link_symbols_wikilink_extracted():
    cg = _code_graph(("code::m.py::build_graph", "build_graph", "function"))
    vg = _vault_graph(("nimbus::design.md", "design", "See [[build_graph]] impl."))

    n_ext, n_inf, n_amb = link_symbols(vg, cg)

    assert n_ext == 1
    assert n_inf == 0
    edges = list(vg.g.edges(data=True))
    assert any(
        d.get("relation") == "mentions_symbol" and d.get("confidence") == "EXTRACTED"
        for _, _, d in edges
    )


def test_link_symbols_word_boundary_inferred():
    cg = _code_graph(("code::m.py::build_graph", "build_graph", "function"))
    vg = _vault_graph(("nimbus::design.md", "design",
                        "The build_graph function assembles the graph."))

    n_ext, n_inf, n_amb = link_symbols(vg, cg)

    assert n_inf == 1
    edges = list(vg.g.edges(data=True))
    assert any(
        d.get("relation") == "mentions_symbol" and d.get("confidence") == "INFERRED"
        for _, _, d in edges
    )


def test_link_symbols_ambiguous():
    cg = _code_graph(
        ("code::a.py::parse_node", "parse_node", "function"),
        ("code::b.py::parse_node", "parse_node", "function"),
    )
    vg = _vault_graph(("nimbus::note.md", "note", "We use parse_node here."))

    n_ext, n_inf, n_amb = link_symbols(vg, cg)

    assert n_amb == 2
    assert n_inf == 0
    edges = [(s, t, d) for s, t, d in vg.g.edges(data=True)
             if d.get("relation") == "mentions_symbol"]
    assert all(d.get("confidence") == "AMBIGUOUS" for _, _, d in edges)


def test_link_symbols_short_name_ignored():
    cg = _code_graph(("code::m.py::run", "run", "function"))
    vg = _vault_graph(("nimbus::note.md", "note", "We run the system."))

    n_ext, n_inf, n_amb = link_symbols(vg, cg)
    assert n_ext + n_inf + n_amb == 0


def test_link_symbols_idempotent():
    cg = _code_graph(("code::m.py::build_graph", "build_graph", "function"))
    vg = _vault_graph(("nimbus::note.md", "note", "build_graph is called here."))

    link_symbols(vg, cg)
    link_symbols(vg, cg)  # second run — should not double-add

    edges = [d for _, _, d in vg.g.edges(data=True)
             if d.get("relation") == "mentions_symbol"]
    assert len(edges) == 1


def test_link_symbols_no_duplicate_per_source_target():
    """Wikilink AND word match for same symbol → only one edge."""
    cg = _code_graph(("code::m.py::build_graph", "build_graph", "function"))
    vg = _vault_graph(("nimbus::note.md", "note",
                        "[[build_graph]] calls build_graph internally."))

    link_symbols(vg, cg)

    edges = [d for _, _, d in vg.g.edges(data=True)
             if d.get("relation") == "mentions_symbol"]
    assert len(edges) == 1
    assert edges[0]["confidence"] == "EXTRACTED"  # wikilink wins (processed first)


def test_link_symbols_weight_scales_with_occurrences():
    cg = _code_graph(("code::m.py::build_graph", "build_graph", "function"))
    vg = _vault_graph(("nimbus::note.md", "note",
                        "build_graph build_graph build_graph build_graph build_graph"))

    link_symbols(vg, cg)

    edges = [d for _, _, d in vg.g.edges(data=True)
             if d.get("relation") == "mentions_symbol"]
    assert edges[0]["weight"] > 0.5


def test_link_symbols_empty_code_graph():
    vg = _vault_graph(("nimbus::note.md", "note", "build_graph is interesting."))
    cg = KnowledgeGraph()

    n_ext, n_inf, n_amb = link_symbols(vg, cg)
    assert (n_ext, n_inf, n_amb) == (0, 0, 0)


# ── mark_stale_refs ───────────────────────────────────────────────────────────

def test_mark_stale_refs_basic():
    cg = _code_graph(("code::m.py::build_graph", "build_graph", "function"))
    vg = _vault_graph(("nimbus::note.md", "note", "build_graph is key."))
    link_symbols(vg, cg)

    n = mark_stale_refs(vg, {"code::m.py::build_graph"}, "2026-05-02")
    assert n == 1

    node_data = vg.g.nodes["nimbus::note.md"]
    stale = node_data["frontmatter"]["stale_refs"]
    assert len(stale) == 1
    assert stale[0]["symbol"] == "build_graph"
    assert stale[0]["changed_at"] == "2026-05-02"


def test_mark_stale_refs_no_duplicate():
    cg = _code_graph(("code::m.py::build_graph", "build_graph", "function"))
    vg = _vault_graph(("nimbus::note.md", "note", "build_graph is key."))
    link_symbols(vg, cg)

    mark_stale_refs(vg, {"code::m.py::build_graph"}, "2026-05-02")
    n2 = mark_stale_refs(vg, {"code::m.py::build_graph"}, "2026-05-03")
    assert n2 == 0  # already marked

    stale = vg.g.nodes["nimbus::note.md"]["frontmatter"]["stale_refs"]
    assert len(stale) == 1


def test_mark_stale_refs_unrelated_code_not_marked():
    cg = _code_graph(
        ("code::m.py::build_graph", "build_graph", "function"),
        ("code::m.py::load_config", "load_config", "function"),
    )
    vg = _vault_graph(("nimbus::note.md", "note", "build_graph is key."))
    link_symbols(vg, cg)

    n = mark_stale_refs(vg, {"code::m.py::load_config"}, "2026-05-02")
    assert n == 0


# ── run_link_symbols (integration) ───────────────────────────────────────────

def test_run_link_symbols_roundtrip(tmp_path):
    cg = _code_graph(("code::m.py::build_graph", "build_graph", "function"))
    vg = _vault_graph(("nimbus::note.md", "note", "build_graph is central."))

    vault_path = tmp_path / "vault_graph.json"
    code_path = tmp_path / "code_graph.json"
    vg.save(vault_path)
    cg.save(code_path)

    n_ext, n_inf, n_amb = run_link_symbols(vault_path, code_path)
    assert n_inf == 1

    # Reload and verify edges persisted
    reloaded = KnowledgeGraph.load(vault_path)
    edges = [d for _, _, d in reloaded.g.edges(data=True)
             if d.get("relation") == "mentions_symbol"]
    assert len(edges) == 1
    assert edges[0]["confidence"] == "INFERRED"
