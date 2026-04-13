"""
Obsidian Vault MCP — Markdown 操作

功能：
  1. Frontmatter 解析与序列化（PyYAML）
  2. Section 分割（基于 heading 行）
  3. Section-based patch 操作
  4. Wikilink 提取

不做 Markdown AST 解析 — Obsidian 方言（callouts, dataview）会破坏 AST。
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
    解析 YAML frontmatter。

    返回 (frontmatter_dict, body_without_frontmatter)。
    无 frontmatter 时返回 ({}, original_content)。
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
    """将 frontmatter dict + body 合并为完整 Markdown。"""
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
    """更新 frontmatter 字段（合并，不覆盖其他字段）。"""
    fm, body = parse_frontmatter(content)
    fm.update(updates)
    return serialize_frontmatter(fm, body)


# --------------------------------------------------------------------------
# Section 分割
# --------------------------------------------------------------------------

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class Section:
    """一个 Heading Section。"""
    heading: str        # 完整 heading 行，如 "## 背景"
    level: int          # heading 层级 1-6
    title: str          # heading 标题文字，如 "背景"
    content: str        # heading 下的内容（不含 heading 行本身）
    start_line: int     # 在原文中的起始行号
    end_line: int       # 在原文中的结束行号（不含）


def split_sections(content: str) -> list[Section]:
    """
    按 heading 切分 Markdown 为 sections。

    第一个 heading 之前的内容作为 level=0 的根 section。
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
            # 保存上一个 section
            sections.append(Section(
                heading=current_heading,
                level=current_level,
                title=current_title,
                content="\n".join(current_lines),
                start_line=current_start,
                end_line=i,
            ))
            # 开始新 section
            current_heading = line
            current_level = len(m.group(1))
            current_title = m.group(2).strip()
            current_start = i
            current_lines = []
        else:
            current_lines.append(line)

    # 最后一个 section
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
    查找匹配的 section 索引。

    target_heading 可以是：
      - 完整 heading 行："## 背景"
      - 仅标题文字："背景"
    """
    # 清理 target
    target = target_heading.strip()

    for i, sec in enumerate(sections):
        # 完整 heading 匹配
        if sec.heading.strip() == target:
            return i
        # 仅标题匹配
        if sec.title == target:
            return i

    return None


def reassemble_sections(sections: list[Section]) -> str:
    """将 sections 重新组装为完整 Markdown。"""
    parts: list[str] = []
    for sec in sections:
        if sec.heading:  # 非 root section
            parts.append(sec.heading)
        if sec.content:
            parts.append(sec.content)

    return "\n".join(parts)


# --------------------------------------------------------------------------
# Wikilink 提取
# --------------------------------------------------------------------------

_WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


def extract_wikilinks(content: str) -> list[str]:
    """提取所有 [[wikilink]] 目标（去重，保留顺序）。"""
    seen: set[str] = set()
    result: list[str] = []
    for m in _WIKILINK_PATTERN.finditer(content):
        target = m.group(1).strip()
        if target and target not in seen:
            seen.add(target)
            result.append(target)
    return result


# --------------------------------------------------------------------------
# Tag 提取
# --------------------------------------------------------------------------

_TAG_PATTERN = re.compile(r"(?:^|\s)#([a-zA-Z\u4e00-\u9fff][\w\u4e00-\u9fff/\-]*)", re.MULTILINE)


def extract_tags(content: str) -> list[str]:
    """提取所有 #tag（去重，保留顺序）。不含 heading 中的 #。"""
    # 先移除 frontmatter
    _, body = parse_frontmatter(content)

    seen: set[str] = set()
    result: list[str] = []
    for m in _TAG_PATTERN.finditer(body):
        tag = m.group(1)
        if tag and tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result
