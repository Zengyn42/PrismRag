"""PrismRag v5.4 smoke tests — require live Jei session (skip in CI).

S1  P1 path: vault root hint appears in MCP instructions.
S2  P2 path: search_knowledge KNOW-ID → hint; explain_node KNOW-ID → node.
S3  P3/P4 path: list_knowledge_nodes body_preview + KNOW node has clean label.

All tests are skipped by default. Remove the skip marker and set
PRISM_RAG_GRAPH_PATH / VAULT_PATH env vars to run manually.
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Smoke test — requires live vault + graph; run manually")
def test_s1_p1_vault_hint_in_instructions():
    """S1 — MCP server instructions must contain the vault path hint at startup."""
    import prism_rag.mcp_server.server as srv

    # The vault hint is injected at import time via _startup_settings.
    instructions = srv.mcp._instructions  # type: ignore[attr-defined]
    assert instructions is not None, "MCP server has no instructions"
    assert "Vault root:" in instructions, (
        "Vault root hint not found in MCP instructions — P1 may be broken"
    )


@pytest.mark.skip(reason="Smoke test — requires live vault + graph; run manually")
def test_s2_p2_know_id_routing():
    """S2 — P2 two-layer routing: soft hint in search_knowledge, node in explain_node."""
    import json
    import prism_rag.mcp_server.server as srv

    # Layer 1: search_knowledge pure KNOW-ID must return soft hint.
    raw = srv.search_knowledge("KNOW-000001")
    result = json.loads(raw)
    assert "hint" in result, "search_knowledge did not return soft hint for pure KNOW-ID"
    assert result["results"] == [], "search_knowledge must return empty results with hint"
    assert "explain_node" in result["hint"], "hint must reference explain_node"

    # Layer 2: explain_node must resolve the same ID to a real node.
    raw2 = srv.explain_node("KNOW-000001")
    result2 = json.loads(raw2)
    assert "error" not in result2, f"explain_node returned error: {result2.get('error')}"
    assert "node" in result2, "explain_node must return a node dict"


@pytest.mark.skip(reason="Smoke test — requires live vault + graph; run manually")
def test_s3_p3_p4_list_nodes_preview_and_label():
    """S3 — P3 body_preview present; P4 KNOW node label is human-readable (not raw stem)."""
    import json
    import prism_rag.mcp_server.server as srv

    raw = srv.list_knowledge_nodes()
    result = json.loads(raw)

    nodes = result.get("nodes", [])
    assert len(nodes) > 0, "list_knowledge_nodes returned no nodes (graph empty?)"

    for node in nodes:
        # P3: every node must have body_preview ≤ 50 chars.
        assert "body_preview" in node, f"Node {node.get('id')} missing body_preview"
        assert len(node["body_preview"]) <= 50, (
            f"body_preview exceeds 50 chars on {node.get('id')}"
        )
        # P4: KNOW node label must not look like a raw stem (e.g. 'KNOW-000043-...')
        label: str = node.get("label", "")
        node_id: str = node.get("id", "")
        if node_id.startswith("KNOW-"):
            assert not label.startswith("KNOW-"), (
                f"KNOW node {node_id} has raw stem label '{label}' — P4 label resolver may be broken"
            )
