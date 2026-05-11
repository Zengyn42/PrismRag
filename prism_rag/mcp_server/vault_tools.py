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
import re
import shutil
from datetime import datetime, timezone
from typing import List, Optional, Union

from mcp.server.fastmcp import FastMCP

from prism_rag.config import GraphSource, PrismRagSettings
from prism_rag.store.graph import _json_default as _vault_json_default
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
        # Multi-namespace: prefer the first non-code namespace (i.e. primary vault).
        # LLMs should not need to specify namespace for everyday vault reads/writes.
        non_code = [s for s in graphs if s.namespace != "code"]
        src = non_code[0] if non_code else graphs[0]

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
        default=_vault_json_default,
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
        default=_vault_json_default,
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
# Management tool implementations (private, importable for tests)
# ---------------------------------------------------------------------------


async def _move_note_impl(
    source: str,
    dest: str,
    cas_hash: str = "",
    namespace: str = "",
) -> str:
    """Move or rename a note; update graph node if ID changes.

    Returns JSON with source, dest, new_cas_hash, id_changed, and graph_update.
    """
    result = _resolve_vault(namespace)
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)

    vault, src = result
    settings = PrismRagSettings()

    src_resolved = vault.resolve_path(source)
    if isinstance(src_resolved, dict):
        return json.dumps(src_resolved, ensure_ascii=False)

    dst_resolved = vault.resolve_path(dest)
    if isinstance(dst_resolved, dict):
        return json.dumps(dst_resolved, ensure_ascii=False)

    if not src_resolved.exists():
        return json.dumps(
            fail(VaultErrorCode.NOT_FOUND, f"File not found: {source}"),
            ensure_ascii=False,
        )

    if dst_resolved.exists():
        return json.dumps(
            fail(VaultErrorCode.ALREADY_EXISTS, f"Destination already exists: {dest}"),
            ensure_ascii=False,
        )

    # Optional CAS check on source
    if cas_hash:
        is_valid, actual = verify_cas(src_resolved, cas_hash)
        if not is_valid:
            log_operation(
                tool="move_note",
                target=source,
                action="move",
                status="conflict",
                cas_before=cas_hash,
                namespace=src.namespace,
                actual_hash=actual,
            )
            return json.dumps(
                fail(
                    VaultErrorCode.CONFLICT,
                    "CAS conflict: file has been modified.",
                    expected_hash=cas_hash,
                    actual_hash=actual,
                ),
                ensure_ascii=False,
            )

    # Compute source node ID BEFORE the move
    from prism_rag.ingest.vault_loader import VaultDocument
    vault_root = vault.base_dir
    src_doc = VaultDocument.from_path(src_resolved, vault_root)
    source_node_id = src_doc.id

    # Perform the move (cross-device safe)
    dst_resolved.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_resolved), str(dst_resolved))

    new_hash = compute_file_hash(dst_resolved)

    # Compute dest node ID AFTER the move
    dst_doc = VaultDocument.from_path(dst_resolved, vault_root)
    dest_node_id = dst_doc.id

    id_changed = source_node_id != dest_node_id

    # Graph cleanup: remove stale node if ID changed
    if id_changed:
        from prism_rag.store.graph import KnowledgeGraph
        graph_path = src.graph_path
        if graph_path.exists():
            graph = KnowledgeGraph.load(graph_path)
            if source_node_id in graph.g:
                graph.g.remove_node(source_node_id)
                graph.save(graph_path)
                # Invalidate federated cache
                try:
                    import prism_rag.mcp_server.server as _server_mod
                    _server_mod._federated = None
                except Exception:
                    pass

    graph_stats = _sync_graph(dst_resolved, settings, tool_name="move_note")

    log_operation(
        tool="move_note",
        target=source,
        action="move",
        status="ok",
        namespace=src.namespace,
        destination=dest,
        cas_after=new_hash,
    )

    return json.dumps(
        {
            "status": "ok",
            "data": {
                "source": source,
                "dest": dest,
                "new_cas_hash": new_hash,
                "namespace": src.namespace,
                "id_changed": id_changed,
            },
            "graph_update": graph_stats,
        },
        ensure_ascii=False,
        indent=2,
    )


async def _delete_note_impl(
    path: str,
    cas_hash: str = "",
    namespace: str = "",
) -> str:
    """Soft-delete a note into <vault>/.trash/; remove it from the graph.

    Returns JSON with path, trash_path, and namespace.
    """
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

    # Optional CAS check
    if cas_hash:
        is_valid, actual = verify_cas(resolved, cas_hash)
        if not is_valid:
            log_operation(
                tool="delete_note",
                target=path,
                action="delete",
                status="conflict",
                cas_before=cas_hash,
                namespace=src.namespace,
                actual_hash=actual,
            )
            return json.dumps(
                fail(
                    VaultErrorCode.CONFLICT,
                    "CAS conflict: file has been modified.",
                    expected_hash=cas_hash,
                    actual_hash=actual,
                ),
                ensure_ascii=False,
            )

    # Compute node ID BEFORE deletion for graph cleanup
    from prism_rag.ingest.vault_loader import VaultDocument
    vault_root = vault.base_dir
    doc = VaultDocument.from_path(resolved, vault_root)
    node_id = doc.id

    # Soft delete: move to .trash/<relative_path>
    rel_path = resolved.relative_to(vault_root)
    trash_root = vault_root / ".trash"
    trash_dest = trash_root / rel_path
    trash_dest.parent.mkdir(parents=True, exist_ok=True)

    # Avoid collision in .trash
    if trash_dest.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        trash_dest = trash_dest.with_name(
            f"{trash_dest.stem}_{ts}{trash_dest.suffix}"
        )

    shutil.move(str(resolved), str(trash_dest))
    trash_rel = str(trash_dest.relative_to(vault_root))

    # Remove from graph
    from prism_rag.store.graph import KnowledgeGraph
    graph_path = src.graph_path
    if graph_path.exists():
        graph = KnowledgeGraph.load(graph_path)
        if node_id in graph.g:
            graph.g.remove_node(node_id)
            graph.save(graph_path)
            try:
                import prism_rag.mcp_server.server as _server_mod
                _server_mod._federated = None
            except Exception:
                pass

    log_operation(
        tool="delete_note",
        target=path,
        action="trash",
        status="ok",
        namespace=src.namespace,
        trash_path=trash_rel,
    )

    return json.dumps(
        {
            "status": "ok",
            "data": {
                "path": path,
                "trash_path": trash_rel,
                "namespace": src.namespace,
            },
        },
        ensure_ascii=False,
        indent=2,
    )


async def _manage_tags_impl(
    path: str,
    add: Optional[List[str]] = None,
    remove: Optional[List[str]] = None,
    cas_hash: str = "",
    namespace: str = "",
) -> str:
    """Merge add/remove tag deltas into frontmatter tags; sync graph.

    Only the YAML frontmatter ``tags`` list is modified.  Inline ``#tags`` in
    the note body are intentionally left untouched.

    Returns JSON with tags_before, tags_after, new cas_hash, and graph_update.
    """
    add = list(add) if add else []
    remove = list(remove) if remove else []

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

    # Verify CAS before mutating
    expected = cas_hash if cas_hash else None
    is_valid, actual = verify_cas(resolved, expected)
    if not is_valid and expected is not None:
        log_operation(
            tool="manage_tags",
            target=path,
            action="manage_tags",
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
    fm, body = parse_frontmatter(raw)

    # Capture before state
    tags_before = list(fm.get("tags", []) or [])
    tags = list(tags_before)

    # Normalize and apply add
    for tag in add:
        t = tag.lstrip("#").strip()
        if t and t not in tags:
            tags.append(t)

    # Normalize and apply remove
    if remove:
        remove_set = {r.lstrip("#").strip() for r in remove}
        tags = [t for t in tags if t not in remove_set]

    fm["tags"] = tags
    new_content = serialize_frontmatter(fm, body)

    atomic_write(resolved, new_content)
    new_hash = compute_hash(new_content)

    log_operation(
        tool="manage_tags",
        target=path,
        action="manage_tags",
        status="ok",
        cas_before=cas_hash,
        cas_after=new_hash,
        namespace=src.namespace,
    )

    graph_stats = _sync_graph(resolved, settings, tool_name="manage_tags")

    return json.dumps(
        {
            "status": "ok",
            "data": {
                "path": path,
                "tags_before": tags_before,
                "tags_after": tags,
                "cas_hash": new_hash,
                "namespace": src.namespace,
            },
            "graph_update": graph_stats,
        },
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# Search tool implementations (private, importable for tests)
# ---------------------------------------------------------------------------

# Wikilink regex — mirrors prism_rag.ingest.ast_extractor._WIKILINK_RE
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


def _raw_wikilinks(content: str) -> list[str]:
    """Return list of raw target strings from wikilinks in *content*."""
    targets = []
    for m in _WIKILINK_RE.finditer(content):
        target = m.group("target").strip()
        if target:
            targets.append(target)
    return targets


async def _search_files_impl(
    query: str,
    directory: str = "",
    case_sensitive: bool = False,
    filename_only: bool = False,
    max_results: int = 50,
    namespace: str = "",
) -> str:
    """Keyword search across vault filenames and (optionally) note content.

    Returns JSON with matches list, count, and a truncated flag.
    Each match entry: {path, line_hits: [{line_number, line_text}], filename_match}.
    When filename_only=True, line_hits is always [].
    """
    if not query.strip():
        return json.dumps(
            fail(VaultErrorCode.VALIDATION_ERROR, "query must not be empty"),
            ensure_ascii=False,
        )

    result = _resolve_vault(namespace)
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)
    vault, src = result

    base = vault.resolve_dir(directory)
    if isinstance(base, dict):
        return json.dumps(base, ensure_ascii=False)

    if not base.exists():
        return json.dumps(
            fail(VaultErrorCode.NOT_FOUND, f"Directory not found: {directory or '/'}"),
            ensure_ascii=False,
        )

    q_cmp = query if case_sensitive else query.lower()
    pattern = re.compile(re.escape(query), 0 if case_sensitive else re.IGNORECASE)

    matches: list[dict] = []
    truncated = False

    for p in sorted(base.rglob("*.md")):
        if not p.is_file() or not vault.is_note(p):
            continue

        rel = vault.relative_path(p)

        # --- filename check ---
        fname = p.name if case_sensitive else p.name.lower()
        fname_match = q_cmp in fname

        if filename_only:
            if fname_match:
                matches.append({
                    "path": rel,
                    "filename_match": True,
                    "line_hits": [],
                })
            if len(matches) >= max_results:
                # Check if more files exist to set truncated properly
                # We detect truncation by trying to continue the loop
                truncated = True
                break
            continue

        # --- content check (also covers filename match) ---
        line_hits: list[dict] = []
        if not fname_match:
            try:
                content = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for i, line in enumerate(content.split("\n"), 1):
                if pattern.search(line):
                    line_hits.append({
                        "line_number": i,
                        "line_text": line.strip()[:200],
                    })
            if not line_hits:
                continue
        # fname matched — content check skipped; line_hits remains []

        matches.append({
            "path": rel,
            "filename_match": fname_match,
            "line_hits": line_hits,
        })

        if len(matches) >= max_results:
            truncated = True
            break

    return json.dumps(
        ok(data={
            "matches": matches,
            "count": len(matches),
            "truncated": truncated,
        }),
        ensure_ascii=False,
        indent=2,
    )


async def _get_links_impl(path: str, namespace: str = "") -> str:
    """Return outgoing wikilinks from a note and incoming links from the vault.

    Outgoing links are extracted by scanning the file's raw text for [[...]]
    syntax.  Incoming links are found by walking the entire vault and checking
    which files reference this note's stem.

    # TODO: when vault scale demands it, replace the incoming-link scan with a
    #       graph-backed lookup using the precomputed backlinks already stored
    #       as edges in PrismRag's KnowledgeGraph (graph.g predecessors keyed
    #       by node ID).  The current O(N) full-vault scan is acceptable for
    #       vaults up to ~hundreds of files.
    """
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

    # Outgoing links
    content = resolved.read_text(encoding="utf-8")
    outgoing = _raw_wikilinks(content)

    # Incoming links: scan every other .md file in the vault
    note_stem = resolved.stem  # filename without .md
    note_rel = vault.relative_path(resolved).removesuffix(".md")  # e.g. "folder/note"
    incoming: list[str] = []

    for p in sorted(vault.base_dir.rglob("*.md")):
        if p == resolved or not p.is_file():
            continue
        try:
            other_content = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        other_links = _raw_wikilinks(other_content)
        for link in other_links:
            link = link.strip()
            # wikilink may be bare "note" or "folder/note"
            link_stem = link.split("/")[-1]
            if link_stem == note_stem or link == note_rel:
                incoming.append(vault.relative_path(p))
                break  # count each referencing file once

    return json.dumps(
        ok(data={
            "path": path,
            "outgoing": outgoing,
            "incoming": incoming,
            "namespace": src.namespace,
        }),
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
        """Read a vault note by its relative path within the namespace.
        Returns JSON with content (full text), frontmatter (parsed YAML dict), cas_hash, mtime_ms, and path.
        Does NOT search by title or keyword — path must be the exact relative path (e.g. "folder/note.md").
        Use search_files() to find notes by filename or keyword; use search_knowledge() for semantic lookup.
        """
        return await _read_note_impl(path, namespace)

    @mcp.tool()
    async def list_files(
        directory: str = "",
        pattern: str = "*.md",
        recursive: bool = False,
        namespace: str = "",
    ) -> str:
        """List markdown files in a vault directory.
        Returns JSON with files list (path, size_bytes, mtime_ms) and count.
        Does NOT search file content — use search_files() for keyword search or search_knowledge() for semantic search.

        directory defaults to vault root; pattern defaults to "*.md"; set recursive=True to include subdirectories.
        """
        return await _list_files_impl(directory, pattern, recursive, namespace)

    @mcp.tool()
    async def write_note(
        path: str,
        content: str,
        cas_hash: str = "",
        namespace: str = "",
    ) -> str:
        """Create or overwrite a vault note with the given content. Knowledge graph is synced automatically after write.
        Returns JSON with new cas_hash, path, namespace, and graph_update stats.
        To overwrite an existing file, provide cas_hash from read_note() — returns CONFLICT error on mismatch.
        Passing cas_hash="" creates a new file; returns ALREADY_EXISTS error if the file already exists.

        Use patch_note() to update one section; use update_frontmatter() for YAML-only changes.
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
        """Replace one heading-delimited section in a note; all other sections and frontmatter are preserved.
        Returns JSON with new cas_hash, path, namespace, sections_affected, and graph_update stats.
        Returns VALIDATION_ERROR if section_heading is not found in the note.

        section_heading accepts the full heading with markers (e.g. "## 决策") or just the text ("决策").
        Use write_note() to replace the entire file; use update_frontmatter() for YAML-only changes.
        """
        return await _patch_note_impl(path, section_heading, new_content, cas_hash, namespace)

    @mcp.tool()
    async def update_frontmatter(
        path: str,
        updates: dict,
        cas_hash: str = "",
        namespace: str = "",
    ) -> str:
        """Merge key-value updates into a note's YAML frontmatter; note body is not modified.
        Returns JSON with new cas_hash, path, namespace, and graph_update stats.
        Unlisted frontmatter keys are preserved unchanged. To modify only the tags list, use manage_tags() instead.
        """
        return await _update_frontmatter_impl(path, updates, cas_hash, namespace)

    @mcp.tool()
    async def move_note(
        source: str,
        dest: str,
        cas_hash: str = "",
        namespace: str = "",
    ) -> str:
        """Move or rename a vault note to a new path. The knowledge graph node is remapped to the new path.
        Returns JSON with source, dest, new_cas_hash, id_changed (whether the graph node ID changed), and graph_update.
        Returns ALREADY_EXISTS error if the destination path already exists.

        cas_hash is optional but recommended to detect concurrent edits before the move.
        """
        return await _move_note_impl(source, dest, cas_hash, namespace)

    @mcp.tool()
    async def delete_note(
        path: str,
        cas_hash: str = "",
        namespace: str = "",
    ) -> str:
        """Soft-delete a vault note by moving it to <vault>/.trash/. Removes the node from the knowledge graph.
        Returns JSON with path, trash_path (location under .trash/), and namespace.
        The deleted file is NOT permanently erased — it is recoverable from .trash/.

        cas_hash is optional; if provided and mismatched, returns CONFLICT error before deleting.
        """
        return await _delete_note_impl(path, cas_hash, namespace)

    @mcp.tool()
    async def manage_tags(
        path: str,
        add: Optional[List[str]] = None,
        remove: Optional[List[str]] = None,
        cas_hash: str = "",
        namespace: str = "",
    ) -> str:
        """Add or remove tags in a note's YAML frontmatter tags list.
        Returns JSON with tags_before, tags_after, new cas_hash, and graph_update stats.
        Only modifies the frontmatter tags field — inline #tags in the note body are not touched.

        Duplicate adds and absent removes are silently ignored.
        For non-tag frontmatter fields, use update_frontmatter() instead.
        """
        return await _manage_tags_impl(path, add, remove, cas_hash, namespace)

    @mcp.tool()
    async def search_files(
        query: str,
        directory: str = "",
        case_sensitive: bool = False,
        filename_only: bool = False,
        max_results: int = 50,
        namespace: str = "",
    ) -> str:
        """Search vault notes by exact keyword across filenames and/or note content.
        Returns JSON with matches list (path, filename_match, line_hits with line_number and line_text) and truncated flag.
        Does NOT do semantic or conceptual search — use search_knowledge() for that.

        filename_only=True restricts to filename matching (no content read, faster).
        max_results caps results; truncated=True means more matches exist beyond the limit.
        """
        return await _search_files_impl(
            query, directory, case_sensitive, filename_only, max_results, namespace
        )

    @mcp.tool()
    async def get_links(path: str, namespace: str = "") -> str:
        """Return outgoing [[wikilinks]] from a note and all notes in the vault that link back to it.
        Returns JSON with outgoing (list of wikilink target strings) and incoming (list of file paths referencing this note).
        Does NOT use the graph index — outgoing links are parsed from raw text; incoming links require a full vault scan (O(N)).
        Use trace_path() or search_knowledge() for graph-based connection traversal.
        """
        return await _get_links_impl(path, namespace)
