"""CodeParser — Python source repository parser for the code:: namespace.

Two-phase pipeline:
  Phase 1  — Tree-sitter AST walk: builds module/class/function node tree,
             inherits + imports edges, per-file call-site capture, and
             per-file import tables.
  Phase 2  — Cross-file name resolution: resolves call sites to actual node
             IDs and emits calls edges.

Confidence tiers for calls edges:
  EXTRACTED (1.0)  — deterministic:
      • self.method() resolved within same class hierarchy
      • import-resolved free calls  (from b import foo; foo())
      • module-qualified calls      (import b; b.foo())
  INFERRED  (0.8)  — probabilistic:
      • type-annotation member calls (obj: FooClass → obj.method())
        mapped to FooClass::method; correct when type is not overridden.
  Skipped   — bare member calls without annotation, dynamic dispatch.

Node IDs:
  module   code::path/to/file.py
  class    code::path/to/file.py::ClassName
  method   code::path/to/file.py::ClassName::method_name
  fn       code::path/to/file.py::func_name

Requires: tree-sitter, tree-sitter-python
"""

from __future__ import annotations

import hashlib
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


# ── Data transfer objects ─────────────────────────────────────────────────────

@dataclass
class _CallSite:
    """One call expression found inside a function body."""
    caller_id: str           # node_id of the calling function/method
    call_form: str           # "free" | "attr"
    callee: str              # display string: "foo" or "obj.method"
    receiver: str            # "" for free calls; "obj" / "self" for attr calls
    method: str              # callee name for free; method name for attr
    receiver_type: str       # "__self__" | annotated type name | ""
    enclosing_class_id: str  # non-empty only when receiver_type == "__self__"


@dataclass
class _FileData:
    """All Phase-1 output for one .py file."""
    module_id: str
    module_node: TreeNode
    extra_edges: list[EdgeRecord]
    # local_name → (resolved_node_id_or_best_guess, is_module)
    import_table: dict[str, tuple[str, bool]] = field(default_factory=dict)
    call_sites: list[_CallSite] = field(default_factory=list)


# ── Public parser class ───────────────────────────────────────────────────────

class CodeParser(Parser):
    """Parser for Python source repositories (code:: namespace).

    parse(source) accepts either a directory (whole repo) or a single .py file.
    For a directory the returned ParseResult contains all modules under it; for
    a file it returns just that module's subtree.
    """

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


# ── Two-phase build ───────────────────────────────────────────────────────────

def _build_result(py_files: list[Path], repo_root: Path) -> ParseResult:
    ts_parser = TSParser(_PY_LANGUAGE)

    # Pre-build module name → module_id index (path arithmetic, no parsing needed)
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

    # ── Phase 1: parse all files ──────────────────────────────────────────────
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

    full_ids, class_index = _build_def_index(root)
    calls_edges = _resolve_calls(file_datas, full_ids, class_index)
    all_extra.extend(calls_edges)

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
        metadata={
            "language": "python",
            "line_count": src_text.count("\n") + 1,
        },
    )

    import_table = _build_import_table(ts_root, module_index)
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


# ── Class / function parsers ──────────────────────────────────────────────────

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
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

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
            "line_start": line_start,
            "line_end": line_end,
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
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

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
            "line_start": line_start,
            "line_end": line_end,
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


# ── Phase-2 resolution ────────────────────────────────────────────────────────

def _build_module_index(py_files: list[Path], repo_root: Path) -> dict[str, str]:
    """Map Python dotted module names → module node_id.

    e.g. "prism_rag.retrieve.bfs" → "code::prism_rag/retrieve/bfs.py"
    """
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
        # Also index by last component for unqualified lookups
        index.setdefault(parts[-1][:-3] if parts[-1].endswith(".py") else parts[-1], module_id)
    return index


def _build_import_table(
    ts_root: TSNode,
    module_index: dict[str, str],
) -> dict[str, tuple[str, bool]]:
    """Return local_name → (resolved_node_id, is_module) for all top-level imports."""
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
            mod_text = mod_node.text.decode("utf-8")
            # Skip relative imports — can't resolve without package context
            if any(c.type == "." for c in child.children if c != mod_node):
                pass  # might still have module_text
            mod_name = mod_text.lstrip(".")
            if not mod_name:
                continue
            mod_id = module_index.get(mod_name)

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


def _build_def_index(
    root: TreeNode,
) -> tuple[set[str], dict[str, list[str]]]:
    """Walk the built tree and return:
      full_ids       — set of all node_ids (for O(1) existence check)
      class_index    — short class_name → [node_ids] (for annotation lookup)
    """
    full_ids: set[str] = set()
    class_index: dict[str, list[str]] = {}

    def _walk(node: TreeNode) -> None:
        full_ids.add(node.id)
        if node.kind == "class":
            label = node.label
            class_index.setdefault(label, []).append(node.id)
        for child in node.children:
            _walk(child)

    _walk(root)
    return full_ids, class_index


def _resolve_calls(
    file_datas: list[_FileData],
    full_ids: set[str],
    class_index: dict[str, list[str]],
) -> list[EdgeRecord]:
    """Phase 2: resolve collected call sites to EdgeRecords."""
    edges: list[EdgeRecord] = []
    seen: set[tuple[str, str]] = set()  # (caller_id, target_id) dedup

    for fd in file_datas:
        for site in fd.call_sites:
            target_id: str | None = None
            tier = "EXTRACTED"
            conf = 1.0

            if site.call_form == "free":
                target_id, tier, conf = _resolve_free_call(
                    site.method, site.caller_id, fd
                )

            elif site.call_form == "attr":
                target_id, tier, conf = _resolve_attr_call(
                    site.receiver, site.method, site.receiver_type,
                    site.enclosing_class_id, fd, full_ids, class_index
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
    caller_id: str,
    fd: _FileData,
) -> tuple[str | None, str, float]:
    # 1. Import table: name is an imported symbol
    if name in fd.import_table:
        candidate, is_module = fd.import_table[name]
        return candidate, "EXTRACTED", 1.0

    # 2. Local definition: name defined in the same module
    # caller_id might be "code::mod.py::ClassName::fn" — extract module prefix
    module_id = fd.module_id
    local_id = f"{module_id}::{name}"
    return local_id, "EXTRACTED", 1.0


def _resolve_attr_call(
    receiver: str,
    method: str,
    receiver_type: str,
    enclosing_class_id: str,
    fd: _FileData,
    full_ids: set[str],
    class_index: dict[str, list[str]],
) -> tuple[str | None, str, float]:
    # 1. self.method() — look in enclosing class
    if receiver_type == "__self__" and enclosing_class_id:
        candidate = f"{enclosing_class_id}::{method}"
        if candidate in full_ids:
            return candidate, "EXTRACTED", 1.0
        return None, "EXTRACTED", 1.0

    # 2. b.foo() — receiver is in import table
    if receiver in fd.import_table:
        base_id, is_module = fd.import_table[receiver]
        candidate = f"{base_id}::{method}"
        if is_module:
            return candidate, "EXTRACTED", 1.0
        else:
            # from x import SomeClass as b; b.method()
            return candidate, "INFERRED", 0.8

    # 3. obj.method() with type annotation: obj: FooClass
    if receiver_type and receiver_type != "__self__":
        candidates = class_index.get(receiver_type, [])
        if len(candidates) == 1:
            candidate = f"{candidates[0]}::{method}"
            return candidate, "INFERRED", 0.8

    return None, "EXTRACTED", 1.0


# ── Call site extraction ──────────────────────────────────────────────────────

def _extract_call_sites(
    body: TSNode | None,
    caller_id: str,
    param_types: dict[str, str],
    enclosing_class_id: str,
) -> list[_CallSite]:
    """Walk a function body and collect all call expressions."""
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
                        caller_id=caller_id,
                        call_form="free",
                        callee=name,
                        receiver="",
                        method=name,
                        receiver_type="",
                        enclosing_class_id="",
                    ))
                elif func.type == "attribute":
                    obj_node = func.child_by_field_name("object")
                    attr_node = func.child_by_field_name("attribute")
                    if (obj_node is not None and attr_node is not None
                            and obj_node.type == "identifier"):
                        recv = obj_node.text.decode("utf-8")
                        meth = attr_node.text.decode("utf-8")
                        if recv == "self" and enclosing_class_id:
                            rtype = "__self__"
                            enc = enclosing_class_id
                        else:
                            rtype = param_types.get(recv, "")
                            enc = ""
                        sites.append(_CallSite(
                            caller_id=caller_id,
                            call_form="attr",
                            callee=f"{recv}.{meth}",
                            receiver=recv,
                            method=meth,
                            receiver_type=rtype,
                            enclosing_class_id=enc,
                        ))
        for child in node.named_children:
            _walk(child)

    _walk(body)
    return sites


def _extract_param_type_map(params_node: TSNode | None) -> dict[str, str]:
    """Return {param_name: type_name} for type-annotated parameters.

    Only captures simple identifier types (e.g. "FooClass"), not generics.
    """
    if params_node is None:
        return {}
    types: dict[str, str] = {}
    for p in params_node.named_children:
        if p.type in ("typed_parameter", "typed_default_parameter"):
            name_child = p.children[0] if (p.children and p.children[0].type == "identifier") else None
            type_child = p.child_by_field_name("type")
            if name_child and type_child:
                param_name = name_child.text.decode("utf-8")
                type_text = type_child.text.decode("utf-8").strip()
                # Only map simple identifiers; skip Optional[X], list[X], etc.
                if type_text.isidentifier():
                    types[param_name] = type_text
    return types


# ── Tree-sitter helpers ───────────────────────────────────────────────────────

def _unwrap_decorated(node: TSNode) -> tuple[TSNode, list[TSNode]]:
    """Return (inner definition, decorators) for decorated_definition, else (node, [])."""
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
    """Module content = everything except class and function bodies."""
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
