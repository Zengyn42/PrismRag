"""CRUD vault tools ported from ZenithLoom's Obsidian MCP server.

This module provides read-side vault tools (read_note, list_files,
get_frontmatter) for PrismRag's MCP interface.  Each tool accepts an
optional ``namespace`` parameter supporting multi-vault federation.

The public entry point is ``register_vault_tools(mcp)`` which registers
all tools against a FastMCP instance.  Each tool's logic also lives in a
private ``_<name>_impl`` function so tests can call it directly without
spinning up FastMCP.

This task (Step 2 Task 1) covers the 3 read-side tools only.  Write and
manage tools will be added in later tasks.
"""

from __future__ import annotations

import json
from typing import Union

from mcp.server.fastmcp import FastMCP

from prism_rag.config import GraphSource, PrismRagSettings
from prism_rag.vault_ops.cas import compute_file_hash, get_mtime_ms
from prism_rag.vault_ops.errors import VaultErrorCode, fail, ok
from prism_rag.vault_ops.markdown_ops import parse_frontmatter
from prism_rag.vault_ops.vault import Vault


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _resolve_vault(namespace: str = "") -> Union[tuple[Vault, GraphSource], dict]:
    """Resolve a vault + GraphSource for the given namespace.

    Returns ``(Vault, GraphSource)`` on success, or an error dict on failure.
    The error dict has the same shape as ``fail()``.
    """
    settings = PrismRagSettings()
    graphs = settings.resolved_graphs

    if namespace:
        sources = {s.namespace: s for s in graphs}
        src = sources.get(namespace)
        if src is None:
            return fail(
                VaultErrorCode.NOT_FOUND,
                f"Unknown namespace: {namespace!r}",
                available=[s.namespace for s in graphs],
            )
    elif len(graphs) == 1:
        src = graphs[0]
    else:
        return fail(
            VaultErrorCode.VALIDATION_ERROR,
            "Multiple namespaces loaded. Specify the namespace parameter.",
            available=[s.namespace for s in graphs],
        )

    return Vault(src.vault_path), src


# ---------------------------------------------------------------------------
# Tool implementations (private, importable for tests)
# ---------------------------------------------------------------------------


async def _read_note_impl(path: str, namespace: str = "") -> str:
    """Read a vault note; return JSON with content, frontmatter, and CAS hash."""
    result = _resolve_vault(namespace)
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)

    vault, src = result

    resolved = vault.resolve_path(path)
    if isinstance(resolved, dict):
        return json.dumps(resolved, ensure_ascii=False)

    if not resolved.exists():
        return json.dumps(
            fail(VaultErrorCode.NOT_FOUND, f"File not found: {path}"),
            ensure_ascii=False,
        )

    if not resolved.is_file():
        return json.dumps(
            fail(VaultErrorCode.VALIDATION_ERROR, f"Not a file: {path}"),
            ensure_ascii=False,
        )

    content = resolved.read_text(encoding="utf-8")
    fm, _ = parse_frontmatter(content)
    cas_hash = compute_file_hash(resolved)
    mtime_ms = get_mtime_ms(resolved)

    return json.dumps(
        ok(data={
            "content": content,
            "frontmatter": fm,
            "cas_hash": cas_hash,
            "mtime_ms": mtime_ms,
            "path": path,
            "namespace": src.namespace,
        }),
        ensure_ascii=False,
        indent=2,
    )


async def _list_files_impl(
    directory: str = "",
    pattern: str = "*.md",
    recursive: bool = False,
    namespace: str = "",
) -> str:
    """List note files in a vault directory; return JSON with file list."""
    result = _resolve_vault(namespace)
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)

    vault, src = result

    resolved_dir = vault.resolve_dir(directory)
    if isinstance(resolved_dir, dict):
        return json.dumps(resolved_dir, ensure_ascii=False)

    if not resolved_dir.exists():
        return json.dumps(
            fail(VaultErrorCode.NOT_FOUND, f"Directory not found: {directory or '/'}"),
            ensure_ascii=False,
        )

    if not resolved_dir.is_dir():
        return json.dumps(
            fail(VaultErrorCode.VALIDATION_ERROR, f"Not a directory: {directory}"),
            ensure_ascii=False,
        )

    glob_method = resolved_dir.rglob if recursive else resolved_dir.glob
    files = []
    for p in sorted(glob_method(pattern)):
        if p.is_file() and vault.is_note(p):
            try:
                stat = p.stat()
                files.append({
                    "path": vault.relative_path(p),
                    "size_bytes": stat.st_size,
                    "mtime_ms": int(stat.st_mtime * 1000),
                })
            except OSError:
                continue

    return json.dumps(
        ok(data={"files": files, "count": len(files)}),
        ensure_ascii=False,
        indent=2,
    )


async def _get_frontmatter_impl(path: str, namespace: str = "") -> str:
    """Return only the YAML frontmatter of a vault note as JSON."""
    result = _resolve_vault(namespace)
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)

    vault, src = result

    resolved = vault.resolve_path(path)
    if isinstance(resolved, dict):
        return json.dumps(resolved, ensure_ascii=False)

    if not resolved.exists():
        return json.dumps(
            fail(VaultErrorCode.NOT_FOUND, f"File not found: {path}"),
            ensure_ascii=False,
        )

    content = resolved.read_text(encoding="utf-8")
    fm, _ = parse_frontmatter(content)

    return json.dumps(
        ok(data={"frontmatter": fm, "path": path, "namespace": src.namespace}),
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_vault_tools(mcp: FastMCP) -> None:
    """Register all ported vault tools against ``mcp``."""

    @mcp.tool()
    async def read_note(path: str, namespace: str = "") -> str:
        """Read a note's content, frontmatter, and CAS hash.

        Args:
            path: Relative path within the vault (e.g. "projects/my-note.md")
            namespace: Target namespace. Required when multiple namespaces are
                       loaded; optional when only one exists.

        Returns:
            JSON with content, frontmatter dict, cas_hash, mtime_ms, path,
            and namespace.
        """
        return await _read_note_impl(path, namespace)

    @mcp.tool()
    async def list_files(
        directory: str = "",
        pattern: str = "*.md",
        recursive: bool = False,
        namespace: str = "",
    ) -> str:
        """List note files in a vault directory.

        Args:
            directory: Relative directory path (empty = vault root)
            pattern: Glob pattern (default "*.md")
            recursive: If True, traverse sub-directories
            namespace: Target namespace. Required when multiple namespaces are
                       loaded; optional when only one exists.

        Returns:
            JSON with files list (path, size_bytes, mtime_ms) and count.
        """
        return await _list_files_impl(directory, pattern, recursive, namespace)

    @mcp.tool()
    async def get_frontmatter(path: str, namespace: str = "") -> str:
        """Get only the YAML frontmatter of a vault note.

        Args:
            path: Relative path within the vault
            namespace: Target namespace. Required when multiple namespaces are
                       loaded; optional when only one exists.

        Returns:
            JSON with frontmatter dict, path, and namespace.
        """
        return await _get_frontmatter_impl(path, namespace)
