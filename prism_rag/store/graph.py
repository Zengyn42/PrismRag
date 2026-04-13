"""Knowledge graph schema and JSON persistence.

The graph is stored in memory as a NetworkX DiGraph (wikilinks are directional).
Nodes and edges are created via the Node / Edge dataclasses for type safety,
then stored as raw dicts inside NetworkX's attribute system.

JSON format (persisted to graph.json):

    {
      "metadata": {"version", "generated_at", "node_count", "edge_count"},
      "nodes": [
        {"id", "label", "kind", "source_file", "content", "content_hash",
         "tokens", "frontmatter", "community_id",
         "maturity", "confidence", "actionability"}
      ],
      "edges": [
        {"source", "target", "relation", "confidence", "confidence_score",
         "weight", "source_pass"}
      ],
      "communities": [
        {"id", "label", "god_nodes", "member_count", "internal_density"}
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import networkx as nx


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for non-primitive types commonly found in Obsidian frontmatter.

    Handles date/datetime (isoformat), Path (str), and sets (list).
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

Confidence = Literal["EXTRACTED", "INFERRED", "AMBIGUOUS"]
NodeKind = Literal["note", "tag", "category", "image", "pdf", "audio", "section", "block"]
SourcePass = Literal["ast", "media", "embedding", "llm"]

# ── Six-space Am attributes (K-space attribute dimension) ─────────────────
# Derived from Wang Yanzhang's Six-Space Theory, K-space (knowledge element
# space) Am (attribute) dimension. These three attributes describe knowledge
# element metadata, populated by Agents at write time, defined
# and persisted by PrismRag at schema level.
Maturity = Literal["seed", "growing", "mature", "archived"]
ConfidenceLevel = Literal["high", "medium", "low"]
Actionability = Literal["reference", "decision", "task"]


@dataclass
class Node:
    """A single graph node."""

    id: str
    label: str
    kind: NodeKind = "note"
    source_file: str = ""
    content: str = ""
    content_hash: str = ""
    tokens: int = 0
    frontmatter: dict[str, Any] = field(default_factory=dict)
    community_id: str | None = None

    # Am attributes (populated by upstream Agent, persisted by PrismRag)
    maturity: Maturity | None = None          # knowledge maturity: seed → growing → mature → archived
    confidence: ConfidenceLevel | None = None  # source reliability: high / medium / low
    actionability: Actionability | None = None # actionability type: reference / decision / task

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Edge:
    """A directed edge between two nodes."""

    source: str
    target: str
    relation: str
    confidence: Confidence = "EXTRACTED"
    confidence_score: float = 1.0
    weight: float = 1.0
    source_pass: SourcePass = "ast"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Community:
    """A detected community (cluster) in the graph."""

    id: str
    label: str
    god_nodes: list[str] = field(default_factory=list)
    member_count: int = 0
    internal_density: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KnowledgeGraph:
    """NetworkX-backed knowledge graph with JSON persistence."""

    def __init__(self) -> None:
        self.g: nx.DiGraph = nx.DiGraph()
        self.communities: dict[str, Community] = {}

    # ── Mutation ─────────────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        """Add or overwrite a node."""
        self.g.add_nodes_from([(node.id, node.to_dict())])

    def add_edge(self, edge: Edge) -> None:
        """Add or overwrite an edge. Endpoints are auto-created as stub nodes if missing."""
        # Ensure endpoints exist — NetworkX will add them without attributes,
        # so we pre-add empty stubs with the right default schema for dangling targets.
        for endpoint in (edge.source, edge.target):
            if endpoint not in self.g:
                stub = Node(id=endpoint, label=endpoint, kind="note")
                self.add_node(stub)
        self.g.add_edges_from([(edge.source, edge.target, edge.to_dict())])

    # ── Queries ──────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return int(self.g.number_of_nodes())

    @property
    def edge_count(self) -> int:
        return int(self.g.number_of_edges())

    def degree(self, node_id: str) -> int:
        return int(self.g.degree(node_id))

    # ── Persistence ──────────────────────────────────────────────────

    def to_json(self) -> dict[str, Any]:
        return {
            "metadata": {
                "version": "v4.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "node_count": self.node_count,
                "edge_count": self.edge_count,
                "community_count": len(self.communities),
            },
            "nodes": [dict(data) for _, data in self.g.nodes(data=True)],
            "edges": [
                {**data}
                for _, _, data in self.g.edges(data=True)
            ],
            "communities": [c.to_dict() for c in self.communities.values()],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        data = json.loads(path.read_text(encoding="utf-8"))
        kg = cls()
        for n in data.get("nodes", []):
            node_id = n["id"]
            kg.g.add_nodes_from([(node_id, n)])
        for e in data.get("edges", []):
            kg.g.add_edges_from([(e["source"], e["target"], e)])
        for c in data.get("communities", []):
            community = Community(**c)
            kg.communities[community.id] = community
        return kg
