"""
Obsidian Vault MCP — Markdown operations

Features:
  1. Frontmatter parsing and serialization (PyYAML)
  2. Section splitting (based on heading lines)
  3. Section-based patch operations
  4. Wikilink extraction

No Markdown AST parsing — Obsidian dialects (callouts, dataview) would break the AST.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# --------------------------------------------------------------------------
# Frontmatter
# --------------------------------------------------------------------------

_FM_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """
    Parse YAML frontmatter.

    Returns (frontmatter_dict, body_without_frontmatter).
    Returns ({}, original_content) when no frontmatter is present.
    """
    m = _FM_PATTERN.match(content)
    if not m:
        return {}, content

    try:
        fm = yaml.safe_load(m.group(1))
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError:
        fm = {}

    body = content[m.end():]
    return fm, body


def serialize_frontmatter(fm: dict, body: str) -> str:
    """Merge a frontmatter dict and body into a complete Markdown string."""
    if not fm:
        return body

    fm_str = yaml.dump(
        fm,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).rstrip("\n")

    return f"---\n{fm_str}\n---\n\n{body}"


def update_frontmatter(content: str, updates: dict) -> str:
    """Update frontmatter fields (merge; does not overwrite other fields)."""
    fm, body = parse_frontmatter(content)
    fm.update(updates)
    return serialize_frontmatter(fm, body)


# --------------------------------------------------------------------------
# Section splitting
# --------------------------------------------------------------------------

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class Section:
    """A single heading section."""
    heading: str        # full heading line, e.g. "## Background"
    level: int          # heading level 1-6
    title: str          # heading title text, e.g. "Background"
    content: str        # content under the heading (excluding the heading line itself)
    start_line: int     # starting line number in the original document
    end_line: int       # ending line number in the original document (exclusive)


def split_sections(content: str) -> list[Section]:
    """
    Split Markdown into sections by heading.

    Content before the first heading is treated as the root section with level=0.
    """
    lines = content.split("\n")
    sections: list[Section] = []
    current_heading = ""
    current_level = 0
    current_title = "(root)"
    current_start = 0
    current_lines: list[str] = []

    for i, line in enumerate(lines):
        m = _HEADING_PATTERN.match(line)
        if m:
            # Save the previous section
            sections.append(Section(
                heading=current_heading,
                level=current_level,
                title=current_title,
                content="\n".join(current_lines),
                start_line=current_start,
                end_line=i,
            ))
            # Start new section
            current_heading = line
            current_level = len(m.group(1))
            current_title = m.group(2).strip()
            current_start = i
            current_lines = []
        else:
            current_lines.append(line)

    # Last section
    sections.append(Section(
        heading=current_heading,
        level=current_level,
        title=current_title,
        content="\n".join(current_lines),
        start_line=current_start,
        end_line=len(lines),
    ))

    return sections


def find_section(sections: list[Section], target_heading: str) -> int | None:
    """
    Find the index of a matching section.

    target_heading may be:
      - A full heading line: "## Background"
      - Just the title text: "Background"
    """
    # Normalize target
    target = target_heading.strip()

    for i, sec in enumerate(sections):
        # Full heading match
        if sec.heading.strip() == target:
            return i
        # Title-only match
        if sec.title == target:
            return i

    return None


def reassemble_sections(sections: list[Section]) -> str:
    """Reassemble sections into a complete Markdown string."""
    parts: list[str] = []
    for sec in sections:
        if sec.heading:  # non-root section
            parts.append(sec.heading)
        if sec.content:
            parts.append(sec.content)

    return "\n".join(parts)


# --------------------------------------------------------------------------
# Wikilink extraction
# --------------------------------------------------------------------------

_WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


def extract_wikilinks(content: str) -> list[str]:
    """Extract all [[wikilink]] targets (deduplicated, order preserved)."""
    seen: set[str] = set()
    result: list[str] = []
    for m in _WIKILINK_PATTERN.finditer(content):
        target = m.group(1).strip()
        if target and target not in seen:
            seen.add(target)
            result.append(target)
    return result


# --------------------------------------------------------------------------
# Tag extraction
# --------------------------------------------------------------------------

_TAG_PATTERN = re.compile(r"(?:^|\s)#([a-zA-Z\u4e00-\u9fff][\w\u4e00-\u9fff/\-]*)", re.MULTILINE)


def extract_tags(content: str) -> list[str]:
    """Extract all #tags (deduplicated, order preserved). Excludes # used in headings."""
    # Strip frontmatter first
    _, body = parse_frontmatter(content)

    seen: set[str] = set()
    result: list[str] = []
    for m in _TAG_PATTERN.finditer(body):
        tag = m.group(1)
        if tag and tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result
