"""Inbox review TUI — primary v5.2 review interface (textual app).

Keys: a=approve, r=reject, s=skip, j/↓=next, k/↑=prev, q=save+quit
"""
from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Static

from prism_rag.config import PrismRagSettings
from prism_rag.inbox.approval import apply_decision
from prism_rag.inbox.store import InboxStore, StatusTransitionError
from prism_rag.store.federated import FederatedGraph


class InboxReviewApp(App):  # type: ignore[type-arg]
    BINDINGS = [
        Binding("a", "approve", "Approve"),
        Binding("r", "reject", "Reject"),
        Binding("s", "skip", "Skip"),
        Binding("j,down", "next_entry", "Next"),
        Binding("k,up", "prev_entry", "Prev"),
        Binding("q", "save_and_quit", "Quit"),
    ]

    def __init__(self, inbox_path: Path, settings: PrismRagSettings) -> None:
        super().__init__()
        self._inbox_path = Path(inbox_path)
        self._settings = settings
        self._inbox = InboxStore(self._inbox_path)
        self._pending = self._inbox.list_pending(top_n=999)
        self._idx = 0
        self._decisions: dict[str, tuple[str, str]] = {}

    def compose(self) -> ComposeResult:
        self._summary = Static("(loading)")
        yield Header()
        yield Vertical(self._summary, id="main")
        yield Footer()

    def on_mount(self) -> None:
        self._render()

    def _render(self) -> None:
        if not self._pending:
            self._summary.update("No pending entries.")
            return
        e = self._pending[self._idx]
        marker = self._decisions.get(e["id"], (None, ""))[0] or "pending"
        self._summary.update(
            f"[{self._idx + 1}/{len(self._pending)}]  conf={e['confidence']:.2f}  "
            f"top_k={e['top_k_rank']}  status_pending_action={marker}\n"
            f"\nSource (vault): {e['source']}\n"
            f"Target (code):  {e['target']}\n"
            f"\nKeys: a=approve r=reject s=skip j/k=nav q=save+quit"
        )

    def action_next_entry(self) -> None:
        if self._pending:
            self._idx = (self._idx + 1) % len(self._pending)
            self._render()

    def action_prev_entry(self) -> None:
        if self._pending:
            self._idx = (self._idx - 1) % len(self._pending)
            self._render()

    def action_approve(self) -> None:
        if not self._pending:
            return
        eid = self._pending[self._idx]["id"]
        self._decisions[eid] = ("approve", "")
        self._render()

    def action_reject(self) -> None:
        if not self._pending:
            return
        eid = self._pending[self._idx]["id"]
        self._decisions[eid] = ("reject", "")
        self._render()

    def action_skip(self) -> None:
        if not self._pending:
            return
        eid = self._pending[self._idx]["id"]
        self._decisions.pop(eid, None)
        self._render()

    def action_save_and_quit(self) -> None:
        self._save()
        self.exit()

    def _save(self) -> None:
        if not self._decisions:
            return
        # Reload fresh to avoid stale state
        self._inbox = InboxStore(self._inbox_path)
        fg = FederatedGraph.load(self._settings.resolved_graphs)
        nimbus_src = next(
            (s for s in self._settings.resolved_graphs if s.namespace == "nimbus"), None
        )
        if nimbus_src is None:
            return
        for eid, (decision, note) in self._decisions.items():
            try:
                apply_decision(
                    eid, decision, note,
                    inbox=self._inbox, fg=fg, src=nimbus_src,
                    decided_by="user_via_tui",
                )
            except (StatusTransitionError, KeyError):
                continue
        self._inbox.save_atomic()
        nimbus = fg.get_graph("nimbus")
        if nimbus is not None:
            nimbus.save(nimbus_src.graph_path)
