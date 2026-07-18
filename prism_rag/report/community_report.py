"""Phase 4 (v6.0): LLM-written community reports.

Each Leiden community gets a CommunityReport — a structured summary produced by
an LLM that reads all node labels and content within the community.  Reports are
cacheable (JSON + optional Markdown) and consumed by ``global_ask`` for
map-reduce question answering.

Backend resolution follows the same pattern as ``LlmSplitter``:
  1. ``llm_fn`` argument — any ``Callable[[str], str]``.
  2. Default: local Ollama ``/api/generate``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

_OLLAMA_DEFAULT_HOST = "http://localhost:11434"
_OLLAMA_DEFAULT_MODEL = "qwen3.5:9b"

# Maximum characters of node content to feed into the prompt per community.
# Prevents prompt blow-up on large communities.
_MAX_CONTENT_CHARS = 12_000


# ── Dataclass ────────────────────────────────────────────────────────────────


@dataclass
class CommunityReport:
    """Structured LLM-generated summary of a single community."""

    community_id: str
    title: str
    summary: str
    rating: int  # 1–10 importance
    key_findings: list[str] = field(default_factory=list)
    cited_nodes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── LLM helpers ──────────────────────────────────────────────────────────────


def _default_ollama_generate(prompt: str, *, model: str, host: str, timeout: int) -> str:
    """Minimal stdlib Ollama /api/generate call (non-streaming)."""
    import urllib.request

    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "")


def _make_default_llm_fn() -> Callable[[str], str]:
    model = os.environ.get("PRISM_OLLAMA_LLM_MODEL", _OLLAMA_DEFAULT_MODEL)
    host = os.environ.get("PRISM_OLLAMA_HOST", _OLLAMA_DEFAULT_HOST)
    timeout = 300
    return lambda prompt: _default_ollama_generate(
        prompt, model=model, host=host, timeout=timeout
    )


def _extract_json_object(text: str) -> dict:
    """Parse the first JSON object found in *text*.

    Tolerates markdown fences and leading/trailing prose.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    # Fenced block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # First brace-balanced object
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError(f"No parseable JSON object in LLM output (head: {text[:200]!r})")


# ── Prompt ───────────────────────────────────────────────────────────────────

_COMMUNITY_REPORT_PROMPT = """\
You are a knowledge graph analyst. Summarize the following community of \
related knowledge nodes into a structured report.

## Community "{community_label}" (ID: {community_id})

Member count: {member_count}
God nodes (highest-degree hubs): {god_nodes}

### Member nodes

{node_listing}

## Instructions

Produce a JSON object (no markdown fences, no commentary) with these fields:
{{
  "title": "<concise title for this community, <=12 words>",
  "summary": "<2-3 sentence summary of what this community covers>",
  "rating": <integer 1-10, importance/centrality of this community>,
  "key_findings": ["<finding 1>", "<finding 2>", ...],
  "cited_nodes": ["<node_id_1>", "<node_id_2>", ...]
}}

- key_findings: 3-6 bullet points of the most important knowledge in this community.
- cited_nodes: IDs of the most important nodes you referenced.
- rating: 10 = critical infrastructure / core concept; 1 = peripheral / trivial.
"""


# ── Core functions ───────────────────────────────────────────────────────────


def generate_community_report(
    graph: KnowledgeGraph,
    community_id: str,
    llm_fn: Callable[[str], str] | None = None,
) -> CommunityReport:
    """Generate an LLM-written report for a single community.

    Args:
        graph: The knowledge graph containing the community.
        community_id: ID of the community to report on.
        llm_fn: Optional LLM callable (prompt → text). Defaults to Ollama.

    Returns:
        A populated CommunityReport.
    """
    if community_id not in graph.communities:
        raise KeyError(f"Community {community_id!r} not found in graph")

    community = graph.communities[community_id]
    if llm_fn is None:
        llm_fn = _make_default_llm_fn()

    # Collect member nodes
    member_nodes = [
        (nid, data)
        for nid, data in graph.g.nodes(data=True)
        if data.get("community_id") == community_id
    ]

    # Build node listing (truncate to _MAX_CONTENT_CHARS total)
    lines: list[str] = []
    char_budget = _MAX_CONTENT_CHARS
    for nid, data in member_nodes:
        label = data.get("label", nid)
        kind = data.get("kind", "?")
        content = (data.get("content") or "")[:500]
        line = f"- [{nid}] {label} (kind={kind}): {content}"
        if char_budget - len(line) < 0:
            lines.append(f"... ({len(member_nodes) - len(lines)} more nodes truncated)")
            break
        lines.append(line)
        char_budget -= len(line)

    prompt = _COMMUNITY_REPORT_PROMPT.format(
        community_id=community_id,
        community_label=community.label,
        member_count=community.member_count,
        god_nodes=", ".join(community.god_nodes),
        node_listing="\n".join(lines),
    )

    raw = llm_fn(prompt)
    try:
        parsed = _extract_json_object(raw)
    except ValueError:
        logger.warning(
            "[community_report] failed to parse LLM output for %s, using fallback",
            community_id,
        )
        parsed = {}

    return CommunityReport(
        community_id=community_id,
        title=str(parsed.get("title", community.label)),
        summary=str(parsed.get("summary", "")),
        rating=int(parsed.get("rating", 5)),
        key_findings=[str(f) for f in parsed.get("key_findings", [])],
        cited_nodes=[str(n) for n in parsed.get("cited_nodes", [])],
    )


def generate_all_community_reports(
    graph: KnowledgeGraph,
    llm_fn: Callable[[str], str] | None = None,
    min_members: int = 3,
) -> list[CommunityReport]:
    """Generate reports for all communities with at least *min_members* members.

    Args:
        graph: The knowledge graph.
        llm_fn: Optional LLM callable. Defaults to Ollama.
        min_members: Skip communities smaller than this.

    Returns:
        List of CommunityReport objects.
    """
    reports: list[CommunityReport] = []
    for comm in sorted(graph.communities.values(), key=lambda c: -c.member_count):
        if comm.member_count < min_members:
            continue
        try:
            report = generate_community_report(graph, comm.id, llm_fn=llm_fn)
            reports.append(report)
        except Exception as exc:
            logger.warning("[community_report] skipping %s: %s", comm.id, exc)
    return reports


# ── Persistence ──────────────────────────────────────────────────────────────


def save_community_reports(
    reports: list[CommunityReport],
    output_path: Path,
) -> None:
    """Save reports to JSON and an adjacent Markdown file.

    Writes:
      - ``output_path`` (JSON) — machine-readable list of report dicts.
      - ``output_path.with_suffix('.md')`` — human-readable Markdown.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    data = [r.to_dict() for r in reports]
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Markdown
    md_path = output_path.with_suffix(".md")
    lines = ["# Community Reports", ""]
    for r in sorted(reports, key=lambda r: -r.rating):
        lines.append(f"## {r.title} (rating {r.rating}/10)")
        lines.append(f"Community: `{r.community_id}`")
        lines.append("")
        lines.append(r.summary)
        lines.append("")
        if r.key_findings:
            lines.append("**Key findings:**")
            for f in r.key_findings:
                lines.append(f"- {f}")
            lines.append("")
        if r.cited_nodes:
            lines.append(f"Cited: {', '.join(f'`{n}`' for n in r.cited_nodes)}")
            lines.append("")
        lines.append("---")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def load_community_reports(path: Path) -> list[CommunityReport]:
    """Load reports from a JSON file produced by save_community_reports."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [CommunityReport(**item) for item in data]
