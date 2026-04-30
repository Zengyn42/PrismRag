"""CodeParser — Python source repository parser for the code:: namespace.

Two-phase pipeline + post-processing:
  Phase 1  — Tree-sitter AST walk: nodes (module/class/function), edges
             (inherits, imports), per-file call-site capture, import tables.
  Phase 2  — Cross-file name resolution → calls edges.
             Includes relative import resolution and MRO-aware method lookup.
  Phase 3  — Execution flow detection: entry-point scoring + BFS tracing
             → flow nodes + step_of edges.

Confidence tiers for calls/step_of edges:
  EXTRACTED (1.0)  — deterministic:
      • self.method() within same class (or MRO ancestor)
      • import-resolved free calls  (from b import foo; foo())
      • module-qualified calls      (import b; b.foo())
  INFERRED  (0.8)  — probabilistic:
      • type-annotation member calls (obj: FooClass → obj.method())
        resolved via class registry + MRO walk.
  INFERRED  (0.7)  — execution flows (step_of edges).
  Skipped   — bare member calls without annotation, dynamic dispatch,
               ambiguous class names.

Node IDs:
  module   code::path/to/file.py
  class    code::path/to/file.py::ClassName
  method   code::path/to/file.py::ClassName::method_name
  fn       code::path/to/file.py::func_name
  flow     code::<repo>::flow::<entry_fn_name>
  flows    code::<repo>::flows  (virtual container)

Requires: tree-sitter, tree-sitter-python
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node as TSNode
from tree_sitter import Parser as TSParser

from prism_rag.ingest.ast_extractor import _token_count
from prism_rag.ingest.base_parser import Parser
from prism_rag.ingest.base_tree import ParseTree, TreeNode
from prism_rag.ingest.parse_result import EdgeRecord, ParseResult

_PY_LANGUAGE = Language(tspython.language())

_EXCLUDE_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    ".venv", "venv", ".env", "env",
    "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".tox",
    "node_modules", "build", "dist", "site-packages",
    ".gitnexus", ".obsidian",
})

_ENTRY_POINT_NAMES: frozenset[str] = frozenset({
    "main", "run", "start", "serve", "execute", "dispatch",
    "handle", "process", "boot", "launch", "init", "setup",
})


# ── Data transfer objects ─────────────────────────────────────────────────────

@dataclass
class _CallSite:
    caller_id: str
    call_form: str           # "free" | "attr"
    callee: str
    receiver: str
    method: str
    receiver_type: str       # "__self__" | annotated type name | ""
    enclosing_class_id: str


@dataclass
class _FileData:
    module_id: str
    module_node: TreeNode
    extra_edges: list[EdgeRecord]
    import_table: dict[str, tuple[str, bool]] = field(default_factory=dict)
    call_sites: list[_CallSite] = field(default_factory=list)


# ── Public parser class ───────────────────────────────────────────────────────

class CodeParser(Parser):
    """Parser for Python source repositories (code:: namespace)."""

    @property
    def namespace(self) -> str:
        return "code"

    def parse(self, source: Path) -> ParseResult:
        source = source.expanduser().resolve()
        if source.is_file():
            py_files = [source]
            repo_root = source.parent
        else:
            repo_root = source
            py_files = _find_py_files(source)
        return _build_result(py_files, repo_root)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def _build_result(py_files: list[Path], repo_root: Path) -> ParseResult:
    ts_parser = TSParser(_PY_LANGUAGE)
    module_index = _build_module_index(py_files, repo_root)

    repo_hash = hashlib.sha1(
        "".join(sorted(str(f) for f in py_files)).encode()
    ).hexdigest()

    root = TreeNode(
        id=f"code::{repo_root.name}",
        kind="module",
        label=repo_root.name,
        content="",
        namespace="code",
        source_file=str(repo_root),
        content_hash=repo_hash,
        metadata={"language": "python", "line_count": 0},
    )

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    file_datas: list[_FileData] = []
    for py_file in py_files:
        rel = py_file.relative_to(repo_root)
        fd = _parse_file(ts_parser, py_file, rel, module_index)
        if fd is not None:
            file_datas.append(fd)
            root.add_child(fd.module_node)

    # ── Phase 2: resolve calls ────────────────────────────────────────────────
    all_extra: list[EdgeRecord] = []
    for fd in file_datas:
        all_extra.extend(fd.extra_edges)

    full_ids, class_index, inherits_index, fn_meta = _build_def_index(root)
    calls_edges = _resolve_calls(file_datas, full_ids, class_index, inherits_index)
    all_extra.extend(calls_edges)

    # ── Phase 3: execution flow detection ─────────────────────────────────────
    flow_nodes, flow_edges = _detect_flows(fn_meta, calls_edges, full_ids)
    if flow_nodes:
        flows_root = TreeNode(
            id=f"code::{repo_root.name}::flows",
            kind="flows",
            label="execution flows",
            content="",
            namespace="code",
            source_file=str(repo_root),
            metadata={"flow_count": len(flow_nodes)},
        )
        for fn in flow_nodes:
            flows_root.add_child(fn)
        root.add_child(flows_root)
        all_extra.extend(flow_edges)

    tree = ParseTree(root=root, namespace="code", source_file=str(repo_root))
    return ParseResult.from_tree(tree, parser_id="CodeParser", extra_edges=all_extra)


# ── Phase-1 per-file parse ────────────────────────────────────────────────────

def _parse_file(
    ts_parser: TSParser,
    py_file: Path,
    rel_path: Path,
    module_index: dict[str, str],
) -> _FileData | None:
    try:
        src_bytes = py_file.read_bytes()
        src_text = src_bytes.decode("utf-8", errors="replace")
    except OSError:
        return None

    ts_tree = ts_parser.parse(src_bytes)
    ts_root = ts_tree.root_node

    module_id = f"code::{rel_path.as_posix()}"
    module_content = _module_own_text(src_bytes, ts_root)

    module_node = TreeNode(
        id=module_id,
        kind="module",
        label=rel_path.stem,
        content=module_content,
        namespace="code",
        source_file=str(rel_path),
        content_hash=hashlib.sha1(src_bytes).hexdigest(),
        tokens=_token_count(module_content),
        metadata={"language": "python", "line_count": src_text.count("\n") + 1},
    )

    import_table = _build_import_table(ts_root, module_index, rel_path)
    extra_edges: list[EdgeRecord] = []
    call_sites: list[_CallSite] = []

    for child in ts_root.named_children:
        actual, _ = _unwrap_decorated(child)

        if actual.type == "class_definition":
            class_node, edges, sites = _parse_class(
                actual, src_bytes, module_id, rel_path
            )
            if class_node is not None:
                module_node.add_child(class_node)
                extra_edges.extend(edges)
                call_sites.extend(sites)

        elif actual.type == "function_definition":
            fn_node, sites = _parse_function(
                actual, src_bytes, module_id, rel_path, enclosing_class_id=""
            )
            if fn_node is not None:
                module_node.add_child(fn_node)
                call_sites.extend(sites)

        elif actual.type in ("import_statement", "import_from_statement"):
            extra_edges.extend(_parse_import(actual, module_id))

    return _FileData(
        module_id=module_id,
        module_node=module_node,
        extra_edges=extra_edges,
        import_table=import_table,
        call_sites=call_sites,
    )


# ── Class / function / import parsers ─────────────────────────────────────────

def _parse_class(
    node: TSNode,
    src_bytes: bytes,
    module_id: str,
    rel_path: Path,
) -> tuple[TreeNode | None, list[EdgeRecord], list[_CallSite]]:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None, [], []

    class_name = name_node.text.decode("utf-8")
    class_id = f"{module_id}::{class_name}"

    bases: list[str] = []
    sup = node.child_by_field_name("superclasses")
    if sup is not None:
        for base in sup.named_children:
            if base.type in ("identifier", "attribute"):
                bases.append(base.text.decode("utf-8"))

    body = node.child_by_field_name("body")
    docstring = _extract_docstring(body)
    class_header = f"class {class_name}"
    if sup is not None:
        class_header += sup.text.decode("utf-8")
    class_content = class_header + ":\n"
    if docstring:
        class_content += f'    """{docstring}"""\n'

    class_node = TreeNode(
        id=class_id,
        kind="class",
        label=class_name,
        content=class_content,
        namespace="code",
        source_file=str(rel_path),
        tokens=_token_count(class_content),
        metadata={
            "language": "python",
            "line_start": node.start_point[0] + 1,
            "line_end": node.end_point[0] + 1,
            "is_exported": not class_name.startswith("_"),
            "bases": bases,
            "docstring": docstring,
        },
    )

    extra_edges: list[EdgeRecord] = [
        EdgeRecord(
            source_id=class_id,
            target_id=f"code::{base}",
            kind="inherits",
            confidence_tier="EXTRACTED",
            confidence=1.0,
            weight=1.0,
            evidence=[f"class_definition:{class_name}"],
        )
        for base in bases
    ]

    all_sites: list[_CallSite] = []
    if body is not None:
        for child in body.named_children:
            actual, _ = _unwrap_decorated(child)
            if actual.type == "function_definition":
                fn_node, sites = _parse_function(
                    actual, src_bytes, class_id, rel_path,
                    enclosing_class_id=class_id,
                )
                if fn_node is not None:
                    class_node.add_child(fn_node)
                    all_sites.extend(sites)

    return class_node, extra_edges, all_sites


def _parse_function(
    node: TSNode,
    src_bytes: bytes,
    parent_id: str,
    rel_path: Path,
    enclosing_class_id: str = "",
) -> tuple[TreeNode | None, list[_CallSite]]:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None, []

    fn_name = name_node.text.decode("utf-8")
    fn_id = f"{parent_id}::{fn_name}"

    is_async = node.children[0].type == "async" if node.children else False
    params_node = node.child_by_field_name("parameters")
    params_text = params_node.text.decode("utf-8") if params_node is not None else "()"
    signature = f"{'async ' if is_async else ''}def {fn_name}{params_text}"
    parameters = _extract_param_names(params_node) if params_node is not None else []
    param_types = _extract_param_type_map(params_node)

    ret_node = node.child_by_field_name("return_type")
    return_type = ret_node.text.decode("utf-8") if ret_node is not None else None

    body = node.child_by_field_name("body")
    docstring = _extract_docstring(body)
    content = src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    fn_node = TreeNode(
        id=fn_id,
        kind="function",
        label=fn_name,
        content=content,
        namespace="code",
        source_file=str(rel_path),
        tokens=_token_count(content),
        metadata={
            "language": "python",
            "line_start": node.start_point[0] + 1,
            "line_end": node.end_point[0] + 1,
            "signature": signature,
            "is_exported": not fn_name.startswith("_"),
            "is_async": is_async,
            "parameters": parameters,
            "return_type": return_type,
            "docstring": docstring,
        },
    )

    sites = _extract_call_sites(body, fn_id, param_types, enclosing_class_id)
    return fn_node, sites


def _parse_import(node: TSNode, module_id: str) -> list[EdgeRecord]:
    edges: list[EdgeRecord] = []

    if node.type == "import_statement":
        for name in node.named_children:
            if name.type == "dotted_name":
                mod = name.text.decode("utf-8")
                edges.append(EdgeRecord(
                    source_id=module_id,
                    target_id=f"code::{mod.replace('.', '/')}",
                    kind="imports",
                    confidence_tier="EXTRACTED",
                    confidence=1.0,
                    weight=1.0,
                    evidence=["import_statement"],
                ))
            elif name.type == "aliased_import":
                inner = name.child_by_field_name("name")
                if inner:
                    mod = inner.text.decode("utf-8")
                    edges.append(EdgeRecord(
                        source_id=module_id,
                        target_id=f"code::{mod.replace('.', '/')}",
                        kind="imports",
                        confidence_tier="EXTRACTED",
                        confidence=1.0,
                        weight=1.0,
                        evidence=["import_statement"],
                    ))

    elif node.type == "import_from_statement":
        mod_node = node.child_by_field_name("module_name")
        if mod_node is None:
            return edges
        mod = mod_node.text.decode("utf-8").lstrip(".")
        if not mod:
            mod = "__relative__"
        for name in node.children_by_field_name("name"):
            imported = name.text.decode("utf-8")
            edges.append(EdgeRecord(
                source_id=module_id,
                target_id=f"code::{mod.replace('.', '/')}.{imported}",
                kind="imports",
                confidence_tier="EXTRACTED",
                confidence=1.0,
                weight=1.0,
                evidence=["import_from_statement"],
            ))

    return edges


# ── Phase-2: indexes and resolution ──────────────────────────────────────────

def _build_module_index(py_files: list[Path], repo_root: Path) -> dict[str, str]:
    """Map dotted module names → module_id (e.g. "a.b.c" → "code::a/b/c.py")."""
    index: dict[str, str] = {}
    for f in py_files:
        rel = f.relative_to(repo_root)
        module_id = f"code::{rel.as_posix()}"
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            dotted = ".".join(parts[:-1])
        elif parts[-1].endswith(".py"):
            dotted = ".".join(parts)[:-3]
        else:
            continue
        index[dotted] = module_id
        index.setdefault(parts[-1][:-3] if parts[-1].endswith(".py") else parts[-1], module_id)
    return index


def _build_import_table(
    ts_root: TSNode,
    module_index: dict[str, str],
    rel_path: Path,
) -> dict[str, tuple[str, bool]]:
    """Return local_name → (resolved_node_id, is_module) for all imports."""
    table: dict[str, tuple[str, bool]] = {}

    for child in ts_root.named_children:
        if child.type == "import_statement":
            for name_node in child.named_children:
                if name_node.type == "dotted_name":
                    mod_name = name_node.text.decode("utf-8")
                    local = mod_name.split(".")[-1]
                    mid = module_index.get(mod_name) or f"code::{mod_name.replace('.', '/')}"
                    table[local] = (mid, True)
                elif name_node.type == "aliased_import":
                    inner = name_node.child_by_field_name("name")
                    alias = name_node.child_by_field_name("alias")
                    if inner and alias:
                        mod_name = inner.text.decode("utf-8")
                        local = alias.text.decode("utf-8")
                        mid = module_index.get(mod_name) or f"code::{mod_name.replace('.', '/')}"
                        table[local] = (mid, True)

        elif child.type == "import_from_statement":
            mod_node = child.child_by_field_name("module_name")
            if mod_node is None:
                continue

            # tree-sitter wraps relative imports in a "relative_import" node
            if mod_node.type == "relative_import":
                prefix = mod_node.text.decode("utf-8")   # e.g. "." or ".utils" or ".."
                level = len(prefix) - len(prefix.lstrip("."))
                mod_name = prefix.lstrip(".")
            else:
                level = 0
                mod_name = mod_node.text.decode("utf-8")

            if level > 0 and not mod_name:
                # from . import X  —  each imported name is a submodule
                for name_node in child.children:
                    if name_node == mod_node:
                        continue
                    if name_node.type in ("identifier", "dotted_name"):
                        sym = name_node.text.decode("utf-8")
                        sub_id = _resolve_relative_import(level, sym, rel_path, module_index)
                        if sub_id:
                            table[sym] = (sub_id, True)
                    elif name_node.type == "aliased_import":
                        inner = name_node.child_by_field_name("name")
                        alias = name_node.child_by_field_name("alias")
                        if inner and alias:
                            sym = inner.text.decode("utf-8")
                            local = alias.text.decode("utf-8")
                            sub_id = _resolve_relative_import(level, sym, rel_path, module_index)
                            if sub_id:
                                table[local] = (sub_id, True)
                continue

            if level > 0:
                mod_id = _resolve_relative_import(level, mod_name, rel_path, module_index)
            else:
                mod_id = module_index.get(mod_name) if mod_name else None

            for name_node in child.children:
                if name_node == mod_node:
                    continue
                if name_node.type in ("identifier", "dotted_name"):
                    sym = name_node.text.decode("utf-8")
                    target = (f"{mod_id}::{sym}" if mod_id
                              else f"code::{mod_name.replace('.', '/')}.{sym}")
                    table[sym] = (target, False)
                elif name_node.type == "aliased_import":
                    inner = name_node.child_by_field_name("name")
                    alias = name_node.child_by_field_name("alias")
                    if inner and alias:
                        sym = inner.text.decode("utf-8")
                        local = alias.text.decode("utf-8")
                        target = (f"{mod_id}::{sym}" if mod_id
                                  else f"code::{mod_name.replace('.', '/')}.{sym}")
                        table[local] = (target, False)

    return table


def _resolve_relative_import(
    level: int,
    mod_name: str,
    rel_path: Path,
    module_index: dict[str, str],
) -> str | None:
    """Resolve a relative import to a module_id.

    from . import x      → level=1, mod_name=""  → same package as rel_path
    from .utils import f → level=1, mod_name="utils"
    from .. import x     → level=2, mod_name=""  → parent package
    """
    parts = list(rel_path.parts)  # e.g. ["a", "b", "c.py"]
    # Anchor = package directory, going up (level-1) from current package
    package_parts = parts[:-1]  # remove filename → ["a", "b"]
    up = level - 1
    if up > len(package_parts):
        return None
    anchor = package_parts[:len(package_parts) - up] if up > 0 else package_parts

    if mod_name:
        dotted = ".".join(anchor + [mod_name])
    else:
        dotted = ".".join(anchor)

    return module_index.get(dotted)


def _build_def_index(root: TreeNode) -> tuple[
    set[str],
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, dict],
]:
    """Walk the built tree and return:
      full_ids       — all node_ids
      class_index    — short class_name → [node_ids]
      inherits_index — class_id → [resolved base class_ids] (via class_index)
      fn_meta        — fn_id → metadata dict (for flow detection)
    """
    full_ids: set[str] = set()
    class_index: dict[str, list[str]] = {}
    fn_meta: dict[str, dict] = {}

    def _walk(node: TreeNode) -> None:
        full_ids.add(node.id)
        if node.kind == "class":
            class_index.setdefault(node.label, []).append(node.id)
        elif node.kind == "function":
            fn_meta[node.id] = dict(node.metadata) if node.metadata else {}
            fn_meta[node.id]["label"] = node.label
        for child in node.children:
            _walk(child)

    _walk(root)

    # Second pass: resolve inherits using class_index
    inherits_index: dict[str, list[str]] = {}

    def _walk2(node: TreeNode) -> None:
        if node.kind == "class":
            bases = node.metadata.get("bases", []) if node.metadata else []
            resolved: list[str] = []
            for base_name in bases:
                short = base_name.split(".")[-1]
                candidates = class_index.get(short, [])
                if len(candidates) == 1:
                    resolved.append(candidates[0])
            if resolved:
                inherits_index[node.id] = resolved
        for child in node.children:
            _walk2(child)

    _walk2(root)
    return full_ids, class_index, inherits_index, fn_meta


def _resolve_calls(
    file_datas: list[_FileData],
    full_ids: set[str],
    class_index: dict[str, list[str]],
    inherits_index: dict[str, list[str]],
) -> list[EdgeRecord]:
    edges: list[EdgeRecord] = []
    seen: set[tuple[str, str]] = set()

    for fd in file_datas:
        for site in fd.call_sites:
            target_id: str | None = None
            tier = "EXTRACTED"
            conf = 1.0

            if site.call_form == "free":
                target_id, tier, conf = _resolve_free_call(site.method, fd)
            elif site.call_form == "attr":
                target_id, tier, conf = _resolve_attr_call(
                    site.receiver, site.method, site.receiver_type,
                    site.enclosing_class_id, fd, full_ids, class_index, inherits_index,
                )

            if target_id is None or target_id not in full_ids:
                continue
            if target_id == site.caller_id:
                continue
            key = (site.caller_id, target_id)
            if key in seen:
                continue
            seen.add(key)

            edges.append(EdgeRecord(
                source_id=site.caller_id,
                target_id=target_id,
                kind="calls",
                confidence_tier=tier,
                confidence=conf,
                weight=conf,
                evidence=[f"call:{site.callee}"],
            ))

    return edges


def _resolve_free_call(
    name: str,
    fd: _FileData,
) -> tuple[str | None, str, float]:
    if name in fd.import_table:
        candidate, _ = fd.import_table[name]
        return candidate, "EXTRACTED", 1.0
    local_id = f"{fd.module_id}::{name}"
    return local_id, "EXTRACTED", 1.0


def _resolve_attr_call(
    receiver: str,
    method: str,
    receiver_type: str,
    enclosing_class_id: str,
    fd: _FileData,
    full_ids: set[str],
    class_index: dict[str, list[str]],
    inherits_index: dict[str, list[str]],
) -> tuple[str | None, str, float]:
    # 1. self.method() — MRO walk in enclosing class
    if receiver_type == "__self__" and enclosing_class_id:
        result = _mro_lookup(enclosing_class_id, method, full_ids, inherits_index)
        return result, "EXTRACTED", 1.0

    # 2. b.foo() — receiver in import table
    if receiver in fd.import_table:
        base_id, is_module = fd.import_table[receiver]
        candidate = f"{base_id}::{method}"
        tier = "EXTRACTED" if is_module else "INFERRED"
        conf = 1.0 if is_module else 0.8
        return candidate, tier, conf

    # 3. obj.method() with type annotation: obj: FooClass — MRO walk
    if receiver_type and receiver_type != "__self__":
        candidates = class_index.get(receiver_type, [])
        if len(candidates) == 1:
            result = _mro_lookup(candidates[0], method, full_ids, inherits_index)
            if result:
                return result, "INFERRED", 0.8

    return None, "EXTRACTED", 1.0


def _mro_lookup(
    class_id: str,
    method: str,
    full_ids: set[str],
    inherits_index: dict[str, list[str]],
    _visited: set[str] | None = None,
) -> str | None:
    """BFS up the MRO to find the first ancestor that defines method."""
    if _visited is None:
        _visited = set()
    queue: deque[str] = deque([class_id])
    while queue:
        cid = queue.popleft()
        if cid in _visited:
            continue
        _visited.add(cid)
        candidate = f"{cid}::{method}"
        if candidate in full_ids:
            return candidate
        for base_id in inherits_index.get(cid, []):
            if base_id not in _visited:
                queue.append(base_id)
    return None


# ── Phase-3: execution flow detection ─────────────────────────────────────────

def _detect_flows(
    fn_meta: dict[str, dict],
    calls_edges: list[EdgeRecord],
    full_ids: set[str],
    min_entry_score: float = 0.5,
    max_entry_points: int = 30,
    max_depth: int = 8,
    max_branches: int = 5,
    min_flow_length: int = 3,
) -> tuple[list[TreeNode], list[EdgeRecord]]:
    """Detect execution flows from entry points via BFS on calls graph."""
    if not calls_edges:
        return [], []

    # Build adjacency structures from calls edges
    callees: dict[str, list[str]] = {}   # caller → [callee_ids]
    caller_count: dict[str, int] = {}    # callee → number of distinct callers

    for e in calls_edges:
        if e.kind != "calls":
            continue
        callees.setdefault(e.source_id, []).append(e.target_id)
        caller_count[e.target_id] = caller_count.get(e.target_id, 0) + 1

    # Score every function as a potential entry point
    scored: list[tuple[float, str]] = []
    for fn_id, meta in fn_meta.items():
        label = meta.get("label", "")
        n_callers = caller_count.get(fn_id, 0)
        n_callees = len(callees.get(fn_id, []))
        score = _score_entry_point(label, meta, n_callers, n_callees)
        if score >= min_entry_score:
            scored.append((score, fn_id))

    scored.sort(reverse=True)
    candidates = [fn_id for _, fn_id in scored[:max_entry_points]]

    # BFS from each candidate, collect flow steps
    flow_nodes: list[TreeNode] = []
    flow_edges: list[EdgeRecord] = []
    seen_entry_ids: set[str] = set()

    for entry_id in candidates:
        if entry_id in seen_entry_ids:
            continue

        steps = _trace_flow_bfs(entry_id, callees, max_depth, max_branches)
        if len(steps) < min_flow_length:
            continue

        seen_entry_ids.add(entry_id)

        label = fn_meta.get(entry_id, {}).get("label", entry_id.split("::")[-1])
        safe_label = label.replace("/", "_").replace(".", "_")
        # Derive repo name from entry_id: "code::repo_name::..."
        parts = entry_id.split("::")
        repo_name = parts[1] if len(parts) > 1 else "repo"

        flow_id = f"code::{repo_name}::flow::{safe_label}"
        flow_node = TreeNode(
            id=flow_id,
            kind="flow",
            label=f"{label} flow",
            content="",
            namespace="code",
            source_file="",
            metadata={
                "entry_point": entry_id,
                "step_count": len(steps),
                "steps": steps,
            },
        )
        flow_nodes.append(flow_node)

        for step_id in steps:
            if step_id not in full_ids:
                continue
            flow_edges.append(EdgeRecord(
                source_id=step_id,
                target_id=flow_id,
                kind="step_of",
                confidence_tier="INFERRED",
                confidence=0.7,
                weight=0.7,
                evidence=["flow_detection"],
            ))

    # Remove dominated flows (A's steps ⊂ B's steps → remove A)
    flow_step_sets = [(fn, set(fn.metadata["steps"])) for fn in flow_nodes]
    kept: list[TreeNode] = []
    for i, (fn_i, steps_i) in enumerate(flow_step_sets):
        dominated = any(
            steps_i < steps_j
            for j, (fn_j, steps_j) in enumerate(flow_step_sets)
            if i != j
        )
        if not dominated:
            kept.append(fn_i)

    kept_ids = {fn.id for fn in kept}
    kept_edges = [e for e in flow_edges if e.target_id in kept_ids]

    return kept, kept_edges


def _score_entry_point(
    label: str,
    meta: dict,
    caller_count: int,
    callee_count: int,
) -> float:
    score = 0.0
    if meta.get("is_exported", False):
        score += 0.2
    if caller_count == 0:
        score += 0.4
    if callee_count >= 2:
        score += 0.2
    name_lower = label.lower()
    if name_lower in _ENTRY_POINT_NAMES:
        score += 0.3
    elif any(name_lower.startswith(p) for p in ("on_", "handle_", "process_", "run_", "dispatch_")):
        score += 0.25
    return min(score, 1.0)


def _trace_flow_bfs(
    entry_id: str,
    callees: dict[str, list[str]],
    max_depth: int,
    max_branches: int,
) -> list[str]:
    """BFS downstream from entry_id following calls edges. Returns ordered step list."""
    visited: set[str] = {entry_id}
    steps: list[str] = [entry_id]
    queue: deque[tuple[str, int]] = deque([(entry_id, 0)])

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for callee_id in callees.get(current, [])[:max_branches]:
            if callee_id not in visited:
                visited.add(callee_id)
                steps.append(callee_id)
                queue.append((callee_id, depth + 1))

    return steps


# ── Call site extraction ──────────────────────────────────────────────────────

def _extract_call_sites(
    body: TSNode | None,
    caller_id: str,
    param_types: dict[str, str],
    enclosing_class_id: str,
) -> list[_CallSite]:
    if body is None:
        return []

    sites: list[_CallSite] = []

    def _walk(node: TSNode) -> None:
        if node.type == "call":
            func = node.child_by_field_name("function")
            if func is not None:
                if func.type == "identifier":
                    name = func.text.decode("utf-8")
                    sites.append(_CallSite(
                        caller_id=caller_id, call_form="free",
                        callee=name, receiver="", method=name,
                        receiver_type="", enclosing_class_id="",
                    ))
                elif func.type == "attribute":
                    obj_node = func.child_by_field_name("object")
                    attr_node = func.child_by_field_name("attribute")
                    if (obj_node is not None and attr_node is not None
                            and obj_node.type == "identifier"):
                        recv = obj_node.text.decode("utf-8")
                        meth = attr_node.text.decode("utf-8")
                        if recv == "self" and enclosing_class_id:
                            rtype, enc = "__self__", enclosing_class_id
                        else:
                            rtype, enc = param_types.get(recv, ""), ""
                        sites.append(_CallSite(
                            caller_id=caller_id, call_form="attr",
                            callee=f"{recv}.{meth}", receiver=recv, method=meth,
                            receiver_type=rtype, enclosing_class_id=enc,
                        ))
        for child in node.named_children:
            _walk(child)

    _walk(body)
    return sites


def _extract_param_type_map(params_node: TSNode | None) -> dict[str, str]:
    """Return {param_name: type_name} for simply-typed parameters."""
    if params_node is None:
        return {}
    types: dict[str, str] = {}
    for p in params_node.named_children:
        if p.type in ("typed_parameter", "typed_default_parameter"):
            name_child = (p.children[0]
                          if p.children and p.children[0].type == "identifier"
                          else None)
            type_child = p.child_by_field_name("type")
            if name_child and type_child:
                type_text = type_child.text.decode("utf-8").strip()
                if type_text.isidentifier():
                    types[name_child.text.decode("utf-8")] = type_text
    return types


# ── Tree-sitter helpers ───────────────────────────────────────────────────────

def _unwrap_decorated(node: TSNode) -> tuple[TSNode, list[TSNode]]:
    if node.type == "decorated_definition":
        defn = node.child_by_field_name("definition")
        decorators = [c for c in node.named_children if c.type == "decorator"]
        if defn is not None:
            return defn, decorators
    return node, []


def _extract_docstring(body: TSNode | None) -> str:
    if body is None or not body.named_children:
        return ""
    first = body.named_children[0]
    if first.type != "expression_statement" or not first.named_children:
        return ""
    s = first.named_children[0]
    if s.type != "string":
        return ""
    raw = s.text.decode("utf-8")
    for q in ('"""', "'''", '"', "'"):
        if raw.startswith(q) and raw.endswith(q) and len(raw) > 2 * len(q):
            return raw[len(q):-len(q)].strip()
    return raw.strip()


def _extract_param_names(params_node: TSNode) -> list[str]:
    names: list[str] = []
    for p in params_node.named_children:
        if p.type == "identifier":
            names.append(p.text.decode("utf-8"))
        elif p.type in (
            "typed_parameter", "default_parameter", "typed_default_parameter"
        ):
            if p.children and p.children[0].type == "identifier":
                names.append(p.children[0].text.decode("utf-8"))
        elif p.type in ("list_splat_pattern", "dictionary_splat_pattern"):
            for c in p.named_children:
                if c.type == "identifier":
                    names.append(c.text.decode("utf-8"))
                    break
    return names


def _module_own_text(src_bytes: bytes, ts_root: TSNode) -> str:
    src_text = src_bytes.decode("utf-8", errors="replace")
    lines = src_text.splitlines(keepends=True)
    parts: list[str] = []
    for child in ts_root.named_children:
        actual, _ = _unwrap_decorated(child)
        if actual.type in ("class_definition", "function_definition"):
            continue
        start = child.start_point[0]
        end = child.end_point[0] + 1
        parts.extend(lines[start:end])
    return "".join(parts)


def _find_py_files(root: Path) -> list[Path]:
    results: list[Path] = []
    for path in root.rglob("*.py"):
        parts = path.relative_to(root).parts
        if any(p in _EXCLUDE_DIRS for p in parts):
            continue
        results.append(path)
    return sorted(results)
