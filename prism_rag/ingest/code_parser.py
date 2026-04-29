"""CodeParser — Python source repository parser for the code:: namespace.

Uses Tree-sitter to extract module/class/function hierarchy and emits
deterministic EXTRACTED edges for inherits and imports relationships.

Pipeline output for a directory source:
  ParseTree
    repo-root (kind="module")           ← virtual, carries repo hash
    ├── module (one per .py file)
    │   ├── class
    │   │   └── function (methods)
    │   └── function (top-level)
  ExtraEdges
    inherits   class → base class       EXTRACTED / 1.0
    imports    module → imported name   EXTRACTED / 1.0

Node IDs:
  module   code::path/to/file.py
  class    code::path/to/file.py::ClassName
  method   code::path/to/file.py::ClassName::method_name
  fn       code::path/to/file.py::func_name

Requires: tree-sitter, tree-sitter-python (pip install tree-sitter tree-sitter-python)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

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


# ── Build result ──────────────────────────────────────────────────────────────

def _build_result(py_files: list[Path], repo_root: Path) -> ParseResult:
    ts_parser = TSParser(_PY_LANGUAGE)

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

    extra_edges: list[EdgeRecord] = []

    for py_file in py_files:
        rel = py_file.relative_to(repo_root)
        module_node, file_edges = _parse_file(ts_parser, py_file, rel)
        if module_node is not None:
            root.add_child(module_node)
            extra_edges.extend(file_edges)

    tree = ParseTree(root=root, namespace="code", source_file=str(repo_root))
    return ParseResult.from_tree(tree, parser_id="CodeParser", extra_edges=extra_edges)


# ── Per-file parse ────────────────────────────────────────────────────────────

def _parse_file(
    ts_parser: TSParser,
    py_file: Path,
    rel_path: Path,
) -> tuple[TreeNode | None, list[EdgeRecord]]:
    try:
        src_bytes = py_file.read_bytes()
        src_text = src_bytes.decode("utf-8", errors="replace")
    except OSError:
        return None, []

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

    extra_edges: list[EdgeRecord] = []

    for child in ts_root.named_children:
        actual, decorators = _unwrap_decorated(child)

        if actual.type == "class_definition":
            class_node, edges = _parse_class(actual, src_bytes, module_id, rel_path)
            if class_node is not None:
                module_node.add_child(class_node)
                extra_edges.extend(edges)

        elif actual.type == "function_definition":
            fn_node = _parse_function(actual, src_bytes, module_id, rel_path)
            if fn_node is not None:
                module_node.add_child(fn_node)

        elif actual.type in ("import_statement", "import_from_statement"):
            extra_edges.extend(_parse_import(actual, module_id))

    return module_node, extra_edges


# ── Class / function / import parsers ─────────────────────────────────────────

def _parse_class(
    node: TSNode,
    src_bytes: bytes,
    module_id: str,
    rel_path: Path,
) -> tuple[TreeNode | None, list[EdgeRecord]]:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None, []

    class_name = name_node.text.decode("utf-8")
    class_id = f"{module_id}::{class_name}"
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    # Superclasses
    bases: list[str] = []
    sup = node.child_by_field_name("superclasses")
    if sup is not None:
        for base in sup.named_children:
            if base.type in ("identifier", "attribute"):
                bases.append(base.text.decode("utf-8"))

    # Class content: header + docstring (not method bodies)
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

    if body is not None:
        for child in body.named_children:
            actual, _ = _unwrap_decorated(child)
            if actual.type == "function_definition":
                fn_node = _parse_function(actual, src_bytes, class_id, rel_path)
                if fn_node is not None:
                    class_node.add_child(fn_node)

    return class_node, extra_edges


def _parse_function(
    node: TSNode,
    src_bytes: bytes,
    parent_id: str,
    rel_path: Path,
) -> TreeNode | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None

    fn_name = name_node.text.decode("utf-8")
    fn_id = f"{parent_id}::{fn_name}"
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    is_async = node.children[0].type == "async" if node.children else False

    params_node = node.child_by_field_name("parameters")
    params_text = params_node.text.decode("utf-8") if params_node is not None else "()"
    signature = f"{'async ' if is_async else ''}def {fn_name}{params_text}"
    parameters = _extract_param_names(params_node) if params_node is not None else []

    ret_node = node.child_by_field_name("return_type")
    return_type = ret_node.text.decode("utf-8") if ret_node is not None else None

    body = node.child_by_field_name("body")
    docstring = _extract_docstring(body)

    content = src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    return TreeNode(
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
            # The parameter name is the first child (an identifier) — not a named field.
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
