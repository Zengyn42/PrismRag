from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_rag.config import GraphSource, PrismRagSettings
from prism_rag.inbox.store import InboxEntry, InboxStore
from prism_rag.inbox.tui import InboxReviewApp
from prism_rag.store.graph import KnowledgeGraph, Node


@pytest.fixture
def env(tmp_path):
    src = GraphSource(namespace="nimbus", vault_path=tmp_path, data_dir=tmp_path)
    g = KnowledgeGraph()
    g.add_node(Node(id="doc", label="doc"))
    g.save(src.graph_path)
    inbox = InboxStore(tmp_path / "inbox.jsonl")
    inbox.append(InboxEntry(
        id="e1", source="nimbus::doc", target="code::a.py::Foo",
        edge_kind="mentions_symbol", confidence=0.74, confidence_tier="INFERRED",
        model_id="bge-m3", probe_signals=[], top_k_rank=1, status="pending",
        created_at="x",
    ))
    inbox.save_atomic()
    return tmp_path, src


@pytest.mark.asyncio
async def test_tui_approve_flow(env):
    tmp_path, src = env
    settings = PrismRagSettings(graphs=[src])
    inbox_path = tmp_path / "inbox.jsonl"
    app = InboxReviewApp(inbox_path, settings)
    async with app.run_test() as pilot:
        await pilot.press("a")    # approve current
        await pilot.press("q")    # save and quit
    inbox2 = InboxStore(inbox_path)
    assert inbox2.get("e1")["status"] == "approved"


@pytest.mark.asyncio
async def test_tui_reject_flow(env):
    tmp_path, src = env
    settings = PrismRagSettings(graphs=[src])
    inbox_path = tmp_path / "inbox.jsonl"
    app = InboxReviewApp(inbox_path, settings)
    async with app.run_test() as pilot:
        await pilot.press("r")
        await pilot.press("q")
    inbox2 = InboxStore(inbox_path)
    assert inbox2.get("e1")["status"] == "rejected"


@pytest.mark.asyncio
async def test_tui_skip_keeps_pending(env):
    tmp_path, src = env
    settings = PrismRagSettings(graphs=[src])
    inbox_path = tmp_path / "inbox.jsonl"
    app = InboxReviewApp(inbox_path, settings)
    async with app.run_test() as pilot:
        await pilot.press("s")
        await pilot.press("q")
    inbox2 = InboxStore(inbox_path)
    assert inbox2.get("e1")["status"] == "pending"
