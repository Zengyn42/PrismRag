"""CRUD vault tools ported from ZenithLoom's Obsidian MCP server.

This module provides read-side and write-side vault tools (read_note,
list_files, get_frontmatter, write_note, patch_note, update_frontmatter)
for PrismRag's MCP interface.  Each tool accepts an optional ``namespace``
parameter supporting multi-vault federation.

The public entry point is ``register_vault_tools(mcp)`` which registers
all tools against a FastMCP instance.  Each tool's logic also lives in a
private ``_<name>_impl`` function so tests can call it directly without
spinning up FastMCP.

Write tools trigger ``ingest_file()`` after each successful write to keep
the knowledge graph in sync (best-effort: graph failure does not abort the
write).
"""

from __future__ import annotations

import json
import logging
from typing import Union

from mcp.server.fastmcp import FastMCP

from prism_rag.config import GraphSource, PrismRagSettings
from prism_rag.vault_ops.audit_log import log_operation
from prism_rag.vault_ops.cas import (
    CASConflict,
    atomic_write,
    compute_file_hash,
    compute_hash,
    get_mtime_ms,
    verify_cas,
    write_with_cas,
)
from prism_rag.vault_ops.errors import VaultErrorCode, fail, ok
from prism_rag.vault_ops.markdown_ops import (
    find_section,
    parse_frontmatter,
    reassemble_sections,
    serialize_frontmatter,
    split_sections,
    update_frontmatter,
)
from prism_rag.vault_ops.vault import Vault

logger = logging.getLogger(__name__)


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
# Internal: graph sync helper
# ---------------------------------------------------------------------------


def _sync_graph(resolved_path, settings: PrismRagSettings, tool_name: str) -> dict:
    """Trigger incremental graph ingest after a write.  Best-effort only."""
    from prism_rag.ingest.incremental import ingest_file

    try:
        stats = ingest_file(resolved_path, settings=settings, skip_embed=True, skip_persist=False)
        # Invalidate the module-level federated graph cache so next query
        # sees the updated graph.
        import prism_rag.mcp_server.server as _server_mod
        _server_mod._federated = None
        return stats
    except Exception as exc:
        logger.warning(f"[{tool_name}] graph update failed: {exc}")
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Write tool implementations (private, importable for tests)
# ---------------------------------------------------------------------------


async def _write_note_impl(
    path: str,
    content: str,
    cas_hash: str = "",
    namespace: str = "",
) -> str:
    """Write a note to the vault (create or overwrite); sync graph after write.

    Returns JSON with new cas_hash and graph_update stats.
    """
    result = _resolve_vault(namespace)
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)

    vault, src = result
    settings = PrismRagSettings()

    resolved = vault.resolve_path(path)
    if isinstance(resolved, dict):
        return json.dumps(resolved, ensure_ascii=False)

    expected = cas_hash if cas_hash else None
    try:
        new_hash = write_with_cas(resolved, content, expected_hash=expected)
    except CASConflict as exc:
        log_operation(
            tool="write_note",
            target=path,
            action="write",
            status="conflict",
            cas_before=cas_hash,
            namespace=src.namespace,
            expected_hash=str(exc.expected),
            actual_hash=str(exc.actual),
        )
        if expected is None:
            return json.dumps(
                fail(
                    VaultErrorCode.ALREADY_EXISTS,
                    f"File already exists: {path}. Use read_note to get cas_hash first.",
                    actual_hash=exc.actual,
                ),
                ensure_ascii=False,
            )
        return json.dumps(
            fail(
                VaultErrorCode.CONFLICT,
                "CAS conflict: file has been modified.",
                expected_hash=exc.expected,
                actual_hash=exc.actual,
            ),
            ensure_ascii=False,
        )

    log_operation(
        tool="write_note",
        target=path,
        action="create" if expected is None else "overwrite",
        status="ok",
        cas_before=cas_hash,
        cas_after=new_hash,
        namespace=src.namespace,
    )

    graph_stats = _sync_graph(resolved, settings, tool_name="write_note")

    return json.dumps(
        {
            "status": "ok",
            "data": {"cas_hash": new_hash, "path": path, "namespace": src.namespace},
            "graph_update": graph_stats,
        },
        ensure_ascii=False,
        indent=2,
    )


async def _patch_note_impl(
    path: str,
    section_heading: str,
    new_content: str,
    cas_hash: str = "",
    namespace: str = "",
) -> str:
    """Patch one section of a note identified by its heading; sync graph after write.

    Returns JSON with new cas_hash, sections_affected, and graph_update stats.
    """
    result = _resolve_vault(namespace)
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)

    vault, src = result
    settings = PrismRagSettings()

    resolved = vault.resolve_path(path)
    if isinstance(resolved, dict):
        return json.dumps(resolved, ensure_ascii=False)

    if not resolved.exists():
        return json.dumps(
            fail(VaultErrorCode.NOT_FOUND, f"File not found: {path}"),
            ensure_ascii=False,
        )

    # Verify CAS non-destructively before mutating anything
    expected = cas_hash if cas_hash else None
    is_valid, actual = verify_cas(resolved, expected)
    if not is_valid:
        log_operation(
            tool="patch_note",
            target=path,
            action="patch",
            status="conflict",
            cas_before=cas_hash,
            namespace=src.namespace,
            actual_hash=actual,
        )
        if expected is None:
            return json.dumps(
                fail(
                    VaultErrorCode.ALREADY_EXISTS,
                    f"File already exists: {path}. Provide cas_hash.",
                    actual_hash=actual,
                ),
                ensure_ascii=False,
            )
        return json.dumps(
            fail(
                VaultErrorCode.CONFLICT,
                "CAS conflict: file has been modified.",
                expected_hash=expected,
                actual_hash=actual,
            ),
            ensure_ascii=False,
        )

    raw = resolved.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(raw)
    sections = split_sections(body)
    idx = find_section(sections, section_heading)

    if idx is None:
        return json.dumps(
            fail(
                VaultErrorCode.VALIDATION_ERROR,
                f"Heading not found: {section_heading!r}",
            ),
            ensure_ascii=False,
        )

    sections[idx].content = new_content
    new_body = reassemble_sections(sections)
    patched = serialize_frontmatter(fm, new_body) if fm else new_body

    atomic_write(resolved, patched)
    new_hash = compute_hash(patched)

    log_operation(
        tool="patch_note",
        target=path,
        action="patch",
        status="ok",
        cas_before=cas_hash,
        cas_after=new_hash,
        namespace=src.namespace,
        sections_affected=[section_heading],
    )

    graph_stats = _sync_graph(resolved, settings, tool_name="patch_note")

    return json.dumps(
        {
            "status": "ok",
            "data": {
                "cas_hash": new_hash,
                "path": path,
                "namespace": src.namespace,
                "sections_affected": [section_heading],
            },
            "graph_update": graph_stats,
        },
        ensure_ascii=False,
        indent=2,
    )


async def _update_frontmatter_impl(
    path: str,
    updates: dict,
    cas_hash: str = "",
    namespace: str = "",
) -> str:
    """Merge ``updates`` into the note's frontmatter (other fields untouched); sync graph.

    Returns JSON with new cas_hash and graph_update stats.
    """
    result = _resolve_vault(namespace)
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)

    vault, src = result
    settings = PrismRagSettings()

    resolved = vault.resolve_path(path)
    if isinstance(resolved, dict):
        return json.dumps(resolved, ensure_ascii=False)

    if not resolved.exists():
        return json.dumps(
            fail(VaultErrorCode.NOT_FOUND, f"File not found: {path}"),
            ensure_ascii=False,
        )

    # Verify CAS non-destructively
    expected = cas_hash if cas_hash else None
    is_valid, actual = verify_cas(resolved, expected)
    if not is_valid and expected is not None:
        # Only fail on actual conflict when cas_hash was provided
        log_operation(
            tool="update_frontmatter",
            target=path,
            action="update_frontmatter",
            status="conflict",
            cas_before=cas_hash,
            namespace=src.namespace,
            actual_hash=actual,
        )
        return json.dumps(
            fail(
                VaultErrorCode.CONFLICT,
                "CAS conflict: file has been modified.",
                expected_hash=expected,
                actual_hash=actual,
            ),
            ensure_ascii=False,
        )

    raw = resolved.read_text(encoding="utf-8")
    new_content = update_frontmatter(raw, updates)

    atomic_write(resolved, new_content)
    new_hash = compute_hash(new_content)

    log_operation(
        tool="update_frontmatter",
        target=path,
        action="update_frontmatter",
        status="ok",
        cas_before=cas_hash,
        cas_after=new_hash,
        namespace=src.namespace,
    )

    graph_stats = _sync_graph(resolved, settings, tool_name="update_frontmatter")

    return json.dumps(
        {
            "status": "ok",
            "data": {"cas_hash": new_hash, "path": path, "namespace": src.namespace},
            "graph_update": graph_stats,
        },
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

    @mcp.tool()
    async def write_note(
        path: str,
        content: str,
        cas_hash: str = "",
        namespace: str = "",
    ) -> str:
        """Write a note to the vault (create or overwrite).

        After writing, the knowledge graph is automatically updated
        (best-effort; a graph failure does not abort the write).

        Args:
            path: Relative path within the vault (e.g. "projects/my-note.md")
            content: Full markdown content (including frontmatter if desired)
            cas_hash: Empty string = create new file (fails if already exists).
                      Non-empty = overwrite (fails if hash does not match).
                      Obtain cas_hash from read_note first.
            namespace: Target namespace. Required when multiple namespaces are
                       loaded; optional when only one exists.

        Returns:
            JSON with new cas_hash and graph_update stats.
        """
        return await _write_note_impl(path, content, cas_hash, namespace)

    @mcp.tool()
    async def patch_note(
        path: str,
        section_heading: str,
        new_content: str,
        cas_hash: str = "",
        namespace: str = "",
    ) -> str:
        """Patch one section of a note identified by its heading.

        Only the targeted section's content is replaced; all other sections
        and the frontmatter are preserved unchanged.

        Args:
            path: Relative path within the vault
            section_heading: Exact heading text (e.g. "## 决策") or title only
                             (e.g. "决策")
            new_content: Replacement body text for the section (without the
                         heading line itself)
            cas_hash: CAS hash from read_note. Required to avoid overwriting
                      concurrent edits.
            namespace: Target namespace.

        Returns:
            JSON with new cas_hash, sections_affected, and graph_update stats.
        """
        return await _patch_note_impl(path, section_heading, new_content, cas_hash, namespace)

    @mcp.tool()
    async def update_frontmatter(
        path: str,
        updates: dict,
        cas_hash: str = "",
        namespace: str = "",
    ) -> str:
        """Merge updates into the note's frontmatter (other fields untouched).

        Existing frontmatter keys not listed in ``updates`` are preserved
        byte-for-byte.

        Args:
            path: Relative path within the vault
            updates: Dict of frontmatter fields to add or overwrite
            cas_hash: CAS hash from read_note (optional but recommended).
            namespace: Target namespace.

        Returns:
            JSON with new cas_hash and graph_update stats.
        """
        return await _update_frontmatter_impl(path, updates, cas_hash, namespace)
