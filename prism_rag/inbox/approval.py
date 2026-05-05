"""Shared approval logic — used by TUI, MCP tool, and CLI.

apply_decision(edge_id, decision, note, inbox, fg, src, decided_by)
mutates the inbox entry in memory and, on approve, writes an ANCHORED
mentions_symbol edge into the vault graph (also in memory). The caller
is responsible for `inbox.save_atomic()` and `KnowledgeGraph.save(...)`
to persist (TUI batches; MCP/CLI saves immediately).
"""

from __future__ import annotations

from prism_rag.config import GraphSource
from prism_rag.inbox.store import InboxStore, StatusTransitionError
from prism_rag.store.federated import FederatedGraph
from prism_rag.store.graph import Edge, LifecycleClass

_VALID_DECISIONS = {"approve", "reject"}


def apply_decision(
    edge_id: str,
    decision: str,
    note: str,
    *,
    inbox: InboxStore,
    fg: FederatedGraph,
    src: GraphSource,
    decided_by: str,
) -> None:
    if decision not in _VALID_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(_VALID_DECISIONS)}; got {decision!r}")
    entry = inbox.get(edge_id)
    if entry is None:
        raise KeyError(f"unknown edge_id: {edge_id}")

    if decision == "approve":
        sem_src = entry["source"]            # "nimbus::doc"
        sem_tgt = entry["target"]            # "code::a.py::Symbol"
        nimbus = fg.get_graph("nimbus")
        if nimbus is None:
            raise RuntimeError("nimbus namespace not loaded; cannot write mentions_symbol edge")
        bare_src = sem_src.split("::", 1)[1] if "::" in sem_src else sem_src
        edge = Edge(
            source=bare_src,
            target=sem_tgt,
            relation="mentions_symbol",
            confidence="INFERRED",
            confidence_score=float(entry["confidence"]),
            source_pass="conv",
            lifecycle_class=LifecycleClass.ANCHORED,
        )
        nimbus.add_edge(edge)
        inbox.set_status(edge_id, "approved", decided_by=decided_by, decision_note=note)
    else:
        inbox.set_status(edge_id, "rejected", decided_by=decided_by, decision_note=note)
