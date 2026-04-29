"""Tests for CodeParser — Tree-sitter Python extraction pipeline."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from prism_rag.ingest.code_parser import CodeParser
from prism_rag.ingest.parse_result import ParseResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Minimal Python repo with a module, class, methods, and imports."""
    src = tmp_path / "myrepo"
    src.mkdir()

    (src / "utils.py").write_text(textwrap.dedent("""\
        import os

        CONSTANT = 42

        def helper(x: int) -> str:
            \"\"\"Return string.\"\"\"
            return str(x)
    """))

    (src / "core.py").write_text(textwrap.dedent("""\
        from pathlib import Path
        from utils import helper

        class Base:
            \"\"\"Base class.\"\"\"

            def __init__(self) -> None:
                pass

        class Derived(Base):
            \"\"\"Derived from Base.\"\"\"

            def run(self, path: Path) -> str:
                result = helper(1)
                return result

        async def top_fn(x: int, y: str = "hi") -> bool:
            \"\"\"Async top-level.\"\"\"
            return True
    """))

    (src / "__init__.py").write_text("# package\n")

    # Should be excluded
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "utils.cpython-312.pyc").write_bytes(b"\x00\x01")

    return src


# ── ParseResult shape ─────────────────────────────────────────────────────────

def test_parse_returns_parse_result(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    assert isinstance(result, ParseResult)
    assert result.namespace == "code"
    assert result.parser_id == "CodeParser"


def test_node_namespaces(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    for node in result.nodes:
        assert node.namespace == "code"


def test_all_nodes_extracted_tier(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    for node in result.nodes:
        assert node.confidence_tier == "EXTRACTED"
        assert node.confidence == 1.0


# ── Node kinds ────────────────────────────────────────────────────────────────

def _node_ids(result: ParseResult) -> set[str]:
    return {n.id for n in result.nodes}


def _node_kinds(result: ParseResult) -> dict[str, str]:
    return {n.id: n.kind for n in result.nodes}


def test_module_nodes_present(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    kinds = _node_kinds(result)
    assert any(k == "module" and "utils.py" in nid for nid, k in kinds.items())
    assert any(k == "module" and "core.py" in nid for nid, k in kinds.items())


def test_class_nodes(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    kinds = _node_kinds(result)
    class_ids = [nid for nid, k in kinds.items() if k == "class"]
    labels = {n.label for n in result.nodes if n.kind == "class"}
    assert "Base" in labels
    assert "Derived" in labels


def test_function_nodes(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    labels = {n.label for n in result.nodes if n.kind == "function"}
    assert "helper" in labels
    assert "__init__" in labels
    assert "run" in labels
    assert "top_fn" in labels


def test_dunder_init_excluded_from_files(tmp_repo: Path) -> None:
    # __init__.py is a valid Python file — should be parsed as a module node
    result = CodeParser().parse(tmp_repo)
    kinds = _node_kinds(result)
    assert any("__init__.py" in nid for nid in kinds)


def test_pycache_excluded(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    for node in result.nodes:
        assert "__pycache__" not in node.id


# ── Function metadata ─────────────────────────────────────────────────────────

def _find_node(result: ParseResult, label: str):
    return next((n for n in result.nodes if n.label == label), None)


def test_function_metadata(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    helper = _find_node(result, "helper")
    assert helper is not None
    assert helper.metadata["language"] == "python"
    assert helper.metadata["line_start"] >= 1
    assert helper.metadata["return_type"] == "str"
    assert "x" in helper.metadata["parameters"]
    assert helper.metadata["docstring"] == "Return string."
    assert helper.metadata["is_exported"] is True
    assert helper.metadata["is_async"] is False


def test_async_function_flag(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    fn = _find_node(result, "top_fn")
    assert fn is not None
    assert fn.metadata["is_async"] is True


def test_class_metadata(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    derived = _find_node(result, "Derived")
    assert derived is not None
    assert derived.metadata["bases"] == ["Base"]
    assert derived.metadata["docstring"] == "Derived from Base."
    assert derived.metadata["is_exported"] is True


# ── Edges ─────────────────────────────────────────────────────────────────────

def _edge_kinds(result: ParseResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in result.edges:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    return counts


def test_contains_edges_present(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    kinds = _edge_kinds(result)
    assert kinds.get("contains", 0) > 0


def test_inherits_edge(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    inherits = [e for e in result.edges if e.kind == "inherits"]
    assert len(inherits) == 1
    assert "Derived" in inherits[0].source_id
    assert inherits[0].confidence_tier == "EXTRACTED"
    assert inherits[0].confidence == 1.0


def test_imports_edges(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    imports = [e for e in result.edges if e.kind == "imports"]
    assert len(imports) > 0
    sources = {e.source_id for e in imports}
    assert any("utils.py" in s for s in sources)
    assert any("core.py" in s for s in sources)


def test_all_edges_extracted(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    for e in result.edges:
        assert e.confidence_tier == "EXTRACTED"


# ── Single-file parse ─────────────────────────────────────────────────────────

def test_parse_single_file(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo / "utils.py")
    kinds = {n.kind for n in result.nodes}
    assert "module" in kinds
    assert "function" in kinds
    labels = {n.label for n in result.nodes if n.kind == "function"}
    assert "helper" in labels


# ── Content rule ──────────────────────────────────────────────────────────────

def test_function_content_has_source(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    fn = _find_node(result, "helper")
    assert fn is not None
    assert "def helper" in fn.content
    assert "return str(x)" in fn.content


def test_class_content_is_header_only(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    base = _find_node(result, "Base")
    assert base is not None
    # Class content should have header + docstring but NOT method bodies
    assert "class Base" in base.content
    assert "def __init__" not in base.content


# ── Token counts ──────────────────────────────────────────────────────────────

def test_tokens_positive(tmp_repo: Path) -> None:
    result = CodeParser().parse(tmp_repo)
    fn_nodes = [n for n in result.nodes if n.kind == "function"]
    assert all(n.tokens > 0 for n in fn_nodes)
