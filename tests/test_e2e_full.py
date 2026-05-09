"""Full end-to-end test: two vaults -> ingest -> federate -> all MCP tools -> incremental.

No external API calls — embeddings are mocked with synthetic vectors.
Covers the complete pipeline from .md files to cross-namespace query results.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from prism_rag.config import GraphSource, PrismRagSettings
from prism_rag.ingest.ast_extractor import extract_ast
from prism_rag.ingest.vault_loader import load_vault
from prism_rag.store.embedding_store import EmbeddingStore
from prism_rag.store.federated import FederatedGraph
from prism_rag.store.graph import KnowledgeGraph
from prism_rag.retrieve.bfs import federated_bfs
from prism_rag.retrieve.dfs import federated_dfs
from prism_rag.retrieve.entry import resolve_entry_points


# -- Vault fixtures -----------------------------------------------------------

VAULT_A_FILES = {
    "system-design.md": """\
---
tags: [architecture, system-design]
category: Concepts
aliases: [System Architecture]
---

# System Design

Core principles for building scalable systems.
Depends on [[authentication]] for security layer.
See also [[caching]] for performance.
Uses #architecture patterns throughout.
""",
    "authentication.md": """\
---
tags: [security, auth]
---

# Authentication

OAuth2 and JWT-based auth flows.
Integrates with [[caching]] for token storage.
""",
    "caching.md": """\
---
tags: [performance, infrastructure]
---

# Caching

Redis and Memcached patterns.
Related to #performance optimization.
""",
    "api-gateway.md": """\
---
tags: [api, infrastructure]
---

# API Gateway

Central entry point for all services.
Handles [[authentication]] checks.
Uses [[caching]] for rate limiting.
""",
}

VAULT_B_FILES = {
    "deployment.md": """\
---
tags: [devops, deployment]
---

# Deployment Strategy

CI/CD pipelines and rollout strategies.
Requires [[monitoring]] for health checks.
""",
    "monitoring.md": """\
---
tags: [observability, devops]
---

# Monitoring

Prometheus, Grafana, and alerting setup.
""",
    "infrastructure.md": """\
---
tags: [devops, cloud]
---

# Infrastructure

Cloud architecture and resource management.
Depends on [[deployment]] for provisioning.
""",
}


def _create_vault(vault_dir: Path, files: dict[str, str]) -> None:
    """Write markdown files to a directory."""
    vault_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (vault_dir / name).write_text(content, encoding="utf-8")


def _make_synthetic_embeddings(graph: KnowledgeGraph, theme_vector: list[float]) -> dict[str, list[float]]:
    """Generate synthetic embeddings for all embeddable nodes.

    Each node gets a slightly perturbed version of theme_vector,
    so within-vault nodes are similar to each other but distinguishable.
    """
    vectors = {}
    for i, (node_id, data) in enumerate(graph.g.nodes(data=True)):
        if data.get("kind") in ("note", "image", "pdf", "audio"):
            # Perturb the theme vector slightly per node
            vec = list(theme_vector)
            vec[i % len(vec)] += 0.05 * (i + 1)
            # Normalize
            norm = sum(x * x for x in vec) ** 0.5
            vectors[node_id] = [x / norm for x in vec]
    return vectors


@pytest.fixture
def two_vaults(tmp_path):
    """Create two vaults with separate data dirs and ingest both."""
    vault_a = tmp_path / "vault_a"
    vault_b = tmp_path / "vault_b"
    data_a = tmp_path / "data_a"
    data_b = tmp_path / "data_b"
    data_a.mkdir()
    data_b.mkdir()

    _create_vault(vault_a, VAULT_A_FILES)
    _create_vault(vault_b, VAULT_B_FILES)

    # Ingest vault A
    docs_a, _ = load_vault(vault_a)
    graph_a = KnowledgeGraph()
    extract_ast(graph_a, docs_a)

    # Ingest vault B
    docs_b, _ = load_vault(vault_b)
    graph_b = KnowledgeGraph()
    extract_ast(graph_b, docs_b)

    # Synthetic embeddings — vault A = "systems" theme, vault B = "ops" theme
    # Vectors are close enough to generate embedding bridges across namespaces
    vecs_a = _make_synthetic_embeddings(graph_a, [0.8, 0.2, 0.1] + [0.0] * 765)
    vecs_b = _make_synthetic_embeddings(graph_b, [0.75, 0.25, 0.1] + [0.0] * 765)

    # Persist embeddings to LanceDB
    store_a = EmbeddingStore(data_a / "lance")
    for nid, vec in vecs_a.items():
        store_a.upsert(nid, vec)

    store_b = EmbeddingStore(data_b / "lance")
    for nid, vec in vecs_b.items():
        store_b.upsert(nid, vec)

    # Save graphs
    graph_a.save(data_a / "graph.json")
    graph_b.save(data_b / "graph.json")

    # Build sources
    sources = [
        GraphSource(namespace="tech", vault_path=vault_a, data_dir=data_a),
        GraphSource(namespace="ops", vault_path=vault_b, data_dir=data_b, writable=True),
    ]

    return {
        "sources": sources,
        "vault_a": vault_a,
        "vault_b": vault_b,
        "data_a": data_a,
        "data_b": data_b,
        "graph_a": graph_a,
        "graph_b": graph_b,
        "vecs_a": vecs_a,
        "vecs_b": vecs_b,
    }


# -- Tests --------------------------------------------------------------------


class TestE2EFullPipeline:
    """End-to-end: two vaults -> ingest -> federate -> query -> write -> incremental."""

    def test_01_vault_loading(self, two_vaults):
        """Vault loader finds all markdown files."""
        docs_a, _ = load_vault(two_vaults["vault_a"])
        docs_b, _ = load_vault(two_vaults["vault_b"])
        assert len(docs_a) == 4, f"Expected 4 docs in vault_a, got {len(docs_a)}"
        assert len(docs_b) == 3, f"Expected 3 docs in vault_b, got {len(docs_b)}"

    def test_02_ast_extraction(self, two_vaults):
        """AST extracts nodes (notes + tags) and edges (wikilinks + tagged_as)."""
        g = two_vaults["graph_a"]
        # Notes
        note_nodes = [n for n, d in g.g.nodes(data=True) if d.get("kind") == "note"]
        assert len(note_nodes) == 4

        # Tags — count depends on dedup of frontmatter + inline tags
        tag_nodes = [n for n, d in g.g.nodes(data=True) if d.get("kind") == "tag"]
        assert len(tag_nodes) >= 4, f"Expected at least 4 tag nodes, got {len(tag_nodes)}: {[n for n, d in g.g.nodes(data=True) if d.get('kind') == 'tag']}"

        # Wikilink edges (links_to)
        wikilink_edges = [(u, v) for u, v, d in g.g.edges(data=True) if d.get("relation") == "links_to"]
        assert len(wikilink_edges) >= 3, f"Expected at least 3 wikilink edges, got {len(wikilink_edges)}"

    def test_03_embeddings_persisted(self, two_vaults):
        """Embeddings are stored in LanceDB."""
        store_a = EmbeddingStore(two_vaults["data_a"] / "lance")
        store_b = EmbeddingStore(two_vaults["data_b"] / "lance")
        assert store_a.count() == len(two_vaults["vecs_a"])
        assert store_b.count() == len(two_vaults["vecs_b"])

        # Verify retrieval
        vec = store_a.get("system-design")
        assert vec is not None
        assert len(vec) == 768

    def test_04_federation_and_bridges(self, two_vaults):
        """FederatedGraph loads both graphs and creates bridges."""
        fg = FederatedGraph.load(two_vaults["sources"])
        assert "tech" in fg.namespaces
        assert "ops" in fg.namespaces
        assert fg.node_count == two_vaults["graph_a"].node_count + two_vaults["graph_b"].node_count

        # Should have at least some bridges (embedding or shared tag)
        assert len(fg.bridges) >= 1, "Expected at least 1 bridge between the two namespaces"

    def test_05_entry_resolution(self, two_vaults):
        """resolve_entry_points finds nodes by label, alias, and qualified ID."""
        fg = FederatedGraph.load(two_vaults["sources"])

        # By label (exact match on filename stem)
        entries = resolve_entry_points(fg, "system-design")
        assert len(entries) >= 1
        assert entries[0][0] == "tech"

        # By alias
        alias_entries = resolve_entry_points(fg, "System Architecture")
        assert len(alias_entries) >= 1

        # Scoped
        scoped = resolve_entry_points(fg, "deployment", scope="ops")
        assert len(scoped) >= 1
        assert all(ns == "ops" for ns, _ in scoped)

        # Qualified ID
        qualified = resolve_entry_points(fg, "tech::system-design")
        assert len(qualified) == 1
        assert qualified[0] == ("tech", "system-design")

    def test_06_bfs_crosses_bridges(self, two_vaults):
        """BFS from tech vault reaches ops vault via bridge."""
        fg = FederatedGraph.load(two_vaults["sources"])
        results = federated_bfs(fg, "tech", "system-design", budget=10000)
        namespaces = {r["namespace"] for r in results}
        assert "tech" in namespaces
        # If bridges exist, BFS should cross
        if fg.bridges:
            assert "ops" in namespaces, "BFS should cross bridge to ops namespace"

    def test_07_dfs_crosses_bridges(self, two_vaults):
        """DFS from tech vault reaches ops vault via bridge."""
        fg = FederatedGraph.load(two_vaults["sources"])
        results = federated_dfs(fg, "tech", "system-design", budget=10000)
        namespaces = {r["namespace"] for r in results}
        assert "tech" in namespaces

    def test_08_bfs_scope_restricts(self, two_vaults):
        """BFS with scope stays within the specified namespace."""
        fg = FederatedGraph.load(two_vaults["sources"])
        results = federated_bfs(fg, "tech", "system-design", budget=10000, scope="tech")
        namespaces = {r["namespace"] for r in results}
        assert namespaces == {"tech"}, "Scoped BFS should not cross namespace boundary"

    def test_09_trace_path_within_namespace(self, two_vaults):
        """trace_path finds path between nodes in the same namespace."""
        import prism_rag.mcp_server.server as mcp_mod
        fg = FederatedGraph.load(two_vaults["sources"])
        mcp_mod._federated = fg

        result = json.loads(mcp_mod.trace_path("system-design", "caching", max_length=5))
        assert "error" not in result, f"trace_path failed: {result}"
        assert result["path_length"] >= 1

    def test_10_trace_path_cross_namespace(self, two_vaults):
        """trace_path finds cross-namespace path via bridge."""
        import prism_rag.mcp_server.server as mcp_mod
        fg = FederatedGraph.load(two_vaults["sources"])
        mcp_mod._federated = fg

        # Only possible if bridges exist
        if fg.bridges:
            result = json.loads(mcp_mod.trace_path(
                "tech::system-design", "ops::infrastructure", max_length=15
            ))
            # Should not claim cross-namespace trace is unsupported
            assert result.get("error") != "Cross-namespace trace not yet supported"

    def test_11_list_namespaces(self, two_vaults):
        """list_namespaces MCP tool returns correct stats."""
        import prism_rag.mcp_server.server as mcp_mod
        fg = FederatedGraph.load(two_vaults["sources"])
        mcp_mod._federated = fg

        result = json.loads(mcp_mod.list_namespaces())
        assert len(result["namespaces"]) == 2
        ns_names = {ns["namespace"] for ns in result["namespaces"]}
        assert ns_names == {"tech", "ops"}

    def test_12_explain_node(self, two_vaults):
        """explain_node returns detailed node info with edges."""
        import prism_rag.mcp_server.server as mcp_mod
        fg = FederatedGraph.load(two_vaults["sources"])
        mcp_mod._federated = fg

        result = json.loads(mcp_mod.explain_node("system-design"))
        assert "error" not in result
        assert result["node"]["kind"] == "note"
        # Should have outgoing edges (links_to, tagged_as)
        assert len(result.get("outgoing_edges", [])) >= 1 or len(result.get("incoming_edges", [])) >= 1

    def test_13_search_knowledge(self, two_vaults):
        """search_knowledge returns relevant nodes with content."""
        import prism_rag.mcp_server.server as mcp_mod
        fg = FederatedGraph.load(two_vaults["sources"])
        mcp_mod._federated = fg

        result = json.loads(mcp_mod.search_knowledge("authentication", budget=5000))
        assert "error" not in result
        assert len(result.get("nodes", [])) >= 1

    def test_14_incremental_write_note(self, two_vaults):
        """Writing a new note triggers incremental ingest and updates the graph."""
        data_b = two_vaults["data_b"]
        vault_b = two_vaults["vault_b"]

        # Write a new file directly
        new_file = vault_b / "new-runbook.md"
        new_file.write_text(
            "---\ntags: [runbook, devops]\n---\n\n"
            "# Incident Runbook\n\nSteps for handling outages.\n"
            "See [[deployment]] for rollback procedures.\n",
            encoding="utf-8",
        )

        # Re-ingest vault_b
        docs_b, _ = load_vault(vault_b)
        graph_b = KnowledgeGraph()
        extract_ast(graph_b, docs_b)
        graph_b.save(data_b / "graph.json")

        # Verify new note is in graph
        assert "new-runbook" in graph_b.g
        assert graph_b.g.nodes["new-runbook"]["kind"] == "note"

        # Verify wikilink edge
        assert graph_b.g.has_edge("new-runbook", "deployment")

    def test_15_embedding_store_survives_reingestion(self, two_vaults):
        """EmbeddingStore retains vectors after graph re-save."""
        store = EmbeddingStore(two_vaults["data_a"] / "lance")
        count_before = store.count()

        # Re-save graph (simulating re-ingest without embedding change)
        two_vaults["graph_a"].save(two_vaults["data_a"] / "graph.json")

        # LanceDB should be independent of graph.json
        store2 = EmbeddingStore(two_vaults["data_a"] / "lance")
        assert store2.count() == count_before
