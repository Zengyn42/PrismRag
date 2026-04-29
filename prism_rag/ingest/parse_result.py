"""ParseResult — validated Parser output contract (Pydantic).

Pipeline:
    Parser.parse(source)
        → ParseTree                         (internal tree structure)
        → ParseResult.from_tree(tree, ...)  (Pydantic validation)
        → StorageBackend.write_result()     (storage)

Confidence two-axis model:
    confidence_tier  — epistemic source (categorical, PRIMARY filter)
    confidence       — strength (continuous, SECONDARY filter)

Tier ranges (non-overlapping):
    EXTRACTED   0.95 – 1.00   deterministic parsers only (AST / Tree-sitter / wikilink)
    INFERRED    0.30 – 0.94   reasoning, LLM, embedding similarity
    AMBIGUOUS   0.00 – 0.29   marked for review, not traversed by default

Iron law (D6):
    ConvParser edges are ALWAYS INFERRED, regardless of how many times extracted.
    EXTRACTED is only legal from deterministic parsers.
    Violation raises ValueError at parse time — not at query time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from prism_rag.ingest.base_tree import FILE_LEVEL_KINDS, ParseTree

ConfidenceTier = Literal["EXTRACTED", "INFERRED", "AMBIGUOUS"]

# Non-overlapping tier ranges — enforced by validators.
_TIER_RANGES: dict[str, tuple[float, float]] = {
    "EXTRACTED":  (0.95, 1.00),
    "INFERRED":   (0.30, 0.94),
    "AMBIGUOUS":  (0.00, 0.29),
}

# tier_decay for cumulative_decay path scoring (see impact_bfs)
TIER_DECAY: dict[str, float] = {
    "EXTRACTED":  1.0,
    "INFERRED":   0.6,
    "AMBIGUOUS":  0.2,
}


# ── Node record ───────────────────────────────────────────────────────

class NodeRecord(BaseModel):
    id: str
    namespace: Literal["nimbus", "code", "conv"]
    kind: str
    label: str
    content: str
    source_file: str

    tokens: int = 0
    content_hash: str = ""
    confidence_tier: ConfidenceTier = "EXTRACTED"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_tier_range(self) -> Self:
        lo, hi = _TIER_RANGES[self.confidence_tier]
        if not (lo <= self.confidence <= hi):
            raise ValueError(
                f"Node '{self.id}': confidence_tier={self.confidence_tier} "
                f"requires confidence in [{lo}, {hi}], got {self.confidence:.3f}"
            )
        return self


# ── Edge record ───────────────────────────────────────────────────────

class EdgeRecord(BaseModel):
    source_id: str
    target_id: str
    kind: str                              # "contains"|"wikilink"|"calls"|"imports"|...

    confidence_tier: ConfidenceTier = "EXTRACTED"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_tier_and_iron_law(self) -> Self:
        # Tier ↔ float consistency
        lo, hi = _TIER_RANGES[self.confidence_tier]
        if not (lo <= self.confidence <= hi):
            raise ValueError(
                f"Edge '{self.source_id}'→'{self.target_id}': "
                f"confidence_tier={self.confidence_tier} "
                f"requires confidence in [{lo}, {hi}], got {self.confidence:.3f}"
            )
        # Iron law (D6): ConvParser edges are always INFERRED
        if self.confidence_tier == "EXTRACTED":
            for ev in self.evidence:
                if "conv::" in ev or "ConvParser" in ev:
                    raise ValueError(
                        f"Edge '{self.source_id}'→'{self.target_id}': "
                        "ConvParser output cannot be EXTRACTED. "
                        "Epistemic source cannot be overwritten by frequency (D6)."
                    )
        return self


# ── Parse result ──────────────────────────────────────────────────────

class ParseResult(BaseModel):
    nodes: list[NodeRecord]
    edges: list[EdgeRecord]
    parser_id: str
    namespace: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_tree(
        cls,
        tree: ParseTree,
        parser_id: str,
        extra_edges: list[EdgeRecord] | None = None,
    ) -> "ParseResult":
        """Convert a ParseTree to a validated ParseResult.

        Generates EdgeRecord(kind="contains") for every parent→child pair.
        Lateral edges (wikilinks, calls, imports) are passed via extra_edges.
        """
        nodes: list[NodeRecord] = []
        edges: list[EdgeRecord] = []

        # conv:: namespace is always INFERRED (iron law)
        is_conv = tree.namespace == "conv"

        def _tier(ns: str) -> ConfidenceTier:
            return "INFERRED" if ns == "conv" else "EXTRACTED"

        def _confidence(tier: ConfidenceTier) -> float:
            return 1.0 if tier == "EXTRACTED" else 0.80

        def walk(node, parent_id: str | None = None) -> None:
            tier = _tier(node.namespace)
            conf = _confidence(tier)

            nodes.append(NodeRecord(
                id=node.id,
                namespace=node.namespace,
                kind=node.kind,
                label=node.label,
                content=node.content,
                source_file=node.source_file,
                tokens=node.tokens,
                content_hash=node.content_hash,
                confidence_tier=tier,
                confidence=conf,
                metadata=node.metadata,
            ))

            if parent_id is not None:
                edges.append(EdgeRecord(
                    source_id=parent_id,
                    target_id=node.id,
                    kind="contains",
                    confidence_tier=tier,
                    confidence=conf,
                    weight=1.0,
                    evidence=["tree-structure"],
                ))

            for child in node.children:
                walk(child, parent_id=node.id)

        walk(tree.root)

        if extra_edges:
            edges.extend(extra_edges)

        return cls(
            nodes=nodes,
            edges=edges,
            parser_id=parser_id,
            namespace=tree.namespace,
        )
