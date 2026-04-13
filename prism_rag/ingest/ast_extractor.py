"""Pass 1b: AST extraction from Obsidian markdown.

Extracts the following deterministic signals (all EXTRACTED confidence):

- [[Note Name]]         → links_to           (basic wikilink)
- [[Note#Heading]]      → links_to_section   (section link)
- [[Note^block-id]]     → links_to_block     (block reference)
- ![[embed.png]]        → embeds             (media embed)
- #inline-tag           → tagged_as          (inline tag, creates tag: node)
- frontmatter tags      → tagged_as          (YAML frontmatter tags)
- frontmatter aliases   → aliased_as         (adds aliases to doc_index)
- frontmatter category  → categorized_as    (creates category: node)

All wikilinks are resolved against a case-insensitive index of filename stems
and aliases. Dangling links (targets that don't resolve) are dropped silently
in the MVP — a future version may create 'dangling' nodes for visibility.

Inline tags are stripped of anything inside code blocks/fences to avoid
false positives (e.g., `#include` in C code).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from prism_rag.ingest.vault_loader import VaultDocument
from prism_rag.store.graph import Edge, KnowledgeGraph, Node

# ── Regex patterns ───────────────────────────────────────────────────

# [[target]] or [[target|display]] or [[target#section]] or [[target^block]]
# or prefixed with ! for embeds: ![[target]]
_WIKILINK_RE = re.compile(
    r"""
    (?P<embed>!)?                       # optional ! for embed
    \[\[                                # opening [[
    (?P<target>[^\]\|#\^]+)             # target name (no ] | # ^)
    (?:\#(?P<section>[^\]\|\^]+))?      # optional #section
    (?:\^(?P<block>[^\]\|]+))?          # optional ^block-id
    (?:\|[^\]]+)?                       # optional |display (ignored)
    \]\]                                # closing ]]
    """,
    re.VERBOSE,
)

# Inline tags: #word or #word/nested
# Must be preceded by start-of-string or whitespace to avoid matching
# #include, URL fragments, etc.
# First character must be alphabetic (not digit) to skip section numbers like #31-45
_TAG_RE = re.compile(r"(?:^|(?<=\s))#([A-Za-z_][A-Za-z0-9_\-/]*)")

# Code fence (``` ... ```) and inline code (`...`)
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def _strip_code(text: str) -> str:
    """Remove code blocks and inline code so tags don't get picked up inside them."""
    text = _CODE_FENCE_RE.sub("", text)
    text = _INLINE_CODE_RE.sub("", text)
    return text


def _extract_wikilinks(text: str) -> list[tuple[str, str]]:
    """Extract wikilinks as (target, relation) tuples."""
    results: list[tuple[str, str]] = []
    for match in _WIKILINK_RE.finditer(text):
        target = match.group("target").strip()
        if not target:
            continue
        embed = match.group("embed") == "!"
        section = match.group("section")
        block = match.group("block")
        if embed:
            relation = "embeds"
        elif section:
            relation = "links_to_section"
        elif block:
            relation = "links_to_block"
        else:
            relation = "links_to"
        results.append((target, relation))
    return results


def _extract_inline_tags(text: str) -> set[str]:
    """Extract inline #tags from markdown content (excluding code blocks)."""
    cleaned = _strip_code(text)
    return {match.group(1) for match in _TAG_RE.finditer(cleaned)}


def _build_doc_index(docs: Iterable[VaultDocument]) -> dict[str, str]:
    """Build a case-insensitive index mapping display names/aliases → canonical node IDs.

    Obsidian wikilinks use either the filename stem or a frontmatter alias.
    """
    index: dict[str, str] = {}
    for doc in docs:
        # Primary: filename stem
        index[doc.label.lower()] = doc.id
        # Aliases from frontmatter
        for alias in doc.aliases:
            index[alias.lower()] = doc.id
    return index


def _resolve_wikilink(target: str, doc_index: dict[str, str]) -> str | None:
    """Resolve a wikilink target to a canonical node ID, or None if dangling."""
    return doc_index.get(target.lower())


def _tag_node_id(tag: str) -> str:
    return f"tag:{tag}"


def _category_node_id(category: str) -> str:
    return f"category:{category}"


import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    """Precise token count using tiktoken cl100k_base encoding."""
    return max(1, len(_enc.encode(text)))


# ── Main entrypoint ──────────────────────────────────────────────────


def extract_ast(graph: KnowledgeGraph, docs: list[VaultDocument]) -> None:
    """Pass 1b: populate `graph` with AST-extracted nodes and edges.

    Mutates `graph` in place. Typical call pattern:

        graph = KnowledgeGraph()
        docs = load_vault(vault_path)
        extract_ast(graph, docs)
    """
    # Step 1: Build wikilink resolution index
    doc_index = _build_doc_index(docs)

    # Step 2: Add note nodes (content attached)
    all_tags: set[str] = set()
    all_categories: set[str] = set()

    for doc in docs:
        # Six-space Am attributes: read from frontmatter (populated by Agent at write time)
        fm = doc.frontmatter
        _maturity = fm.get("maturity")
        _confidence = fm.get("confidence")
        _actionability = fm.get("actionability")

        note = Node(
            id=doc.id,
            label=doc.label,
            kind="note",
            source_file=str(doc.relative_path),
            content=doc.content,
            content_hash=doc.content_hash,
            tokens=_token_count(doc.content),
            frontmatter=doc.frontmatter,
            maturity=_maturity if _maturity in ("seed", "growing", "mature", "archived") else None,
            confidence=_confidence if _confidence in ("high", "medium", "low") else None,
            actionability=_actionability if _actionability in ("reference", "decision", "task") else None,
        )
        graph.add_node(note)

        # Collect frontmatter tags + inline tags for this doc
        doc_tags = set(doc.frontmatter_tags) | _extract_inline_tags(doc.content)
        all_tags.update(doc_tags)

        if doc.category:
            all_categories.add(doc.category)

    # Step 3: Create tag and category nodes
    for tag in all_tags:
        graph.add_node(Node(id=_tag_node_id(tag), label=f"#{tag}", kind="tag"))
    for cat in all_categories:
        graph.add_node(Node(id=_category_node_id(cat), label=cat, kind="category"))

    # Step 4: Extract edges
    for doc in docs:
        # 4a: Wikilinks
        for target, relation in _extract_wikilinks(doc.content):
            resolved = _resolve_wikilink(target, doc_index)
            if resolved is None:
                continue  # Skip dangling links for MVP
            if resolved == doc.id:
                continue  # Skip self-links
            graph.add_edge(
                Edge(
                    source=doc.id,
                    target=resolved,
                    relation=relation,
                    confidence="EXTRACTED",
                    confidence_score=1.0,
                    weight=1.0,
                    source_pass="ast",
                )
            )

        # 4b: Tags (frontmatter + inline)
        doc_tags = set(doc.frontmatter_tags) | _extract_inline_tags(doc.content)
        for tag in doc_tags:
            graph.add_edge(
                Edge(
                    source=doc.id,
                    target=_tag_node_id(tag),
                    relation="tagged_as",
                    confidence="EXTRACTED",
                    confidence_score=1.0,
                    weight=1.0,
                    source_pass="ast",
                )
            )

        # 4c: Category
        if doc.category:
            graph.add_edge(
                Edge(
                    source=doc.id,
                    target=_category_node_id(doc.category),
                    relation="categorized_as",
                    confidence="EXTRACTED",
                    confidence_score=1.0,
                    weight=1.0,
                    source_pass="ast",
                )
            )

        # 4d: Aliases (point aliased names → the canonical doc)
        # Note: we don't add edges for aliases since the index already maps them.
        # If we wanted to expose alias relationships in the graph, we'd add
        # 'aliased_as' self-edges or alias nodes, but that's not useful for MVP.
