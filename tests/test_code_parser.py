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


# ── calls edge tests ──────────────────────────────────────────────────────────

import tempfile
import textwrap


def _make_repo(files: dict[str, str]) -> tuple:
    """Write files to a temp dir and parse with CodeParser. Returns (result, tmp_path)."""
    import atexit, shutil
    tmp = Path(tempfile.mkdtemp())
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    for name, content in files.items():
        p = tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(content))
    result = CodeParser().parse(tmp)
    return result, tmp


def _calls(result) -> list[tuple[str, str, str]]:
    """Return [(short_caller, short_target, tier)] for all calls edges."""
    return [
        (
            e.source_id.rsplit("::", 1)[-1],
            e.target_id.rsplit("::", 1)[-1],
            e.confidence_tier,
        )
        for e in result.edges
        if e.kind == "calls"
    ]


class TestSelfCalls:
    def test_self_method_resolved_extracted(self):
        result, _ = _make_repo({"a.py": """
            class Foo:
                def run(self):
                    self.helper()
                def helper(self):
                    pass
        """})
        c = _calls(result)
        assert any(src == "run" and tgt == "helper" and tier == "EXTRACTED"
                   for src, tgt, tier in c)

    def test_self_call_confidence_1(self):
        result, _ = _make_repo({"a.py": """
            class Foo:
                def run(self):
                    self.helper()
                def helper(self): pass
        """})
        edge = next(e for e in result.edges
                    if e.kind == "calls" and e.source_id.endswith("::run"))
        assert edge.confidence == 1.0

    def test_self_call_not_duplicated(self):
        result, _ = _make_repo({"a.py": """
            class Foo:
                def run(self):
                    self.helper()
                    self.helper()
                def helper(self): pass
        """})
        c = [(e.source_id, e.target_id) for e in result.edges
             if e.kind == "calls" and e.source_id.endswith("::run")]
        assert len(c) == 1


class TestFreeCallFromImport:
    def test_from_import_free_call_extracted(self):
        result, _ = _make_repo({
            "b.py": "def foo(): pass\n",
            "a.py": "from b import foo\ndef caller():\n    foo()\n",
        })
        c = _calls(result)
        assert any(src == "caller" and tgt == "foo" and tier == "EXTRACTED"
                   for src, tgt, tier in c)

    def test_aliased_import_resolved(self):
        result, _ = _make_repo({
            "b.py": "def foo(): pass\n",
            "a.py": "from b import foo as bar\ndef caller():\n    bar()\n",
        })
        c = _calls(result)
        assert any(src == "caller" and tgt == "foo" and tier == "EXTRACTED"
                   for src, tgt, tier in c)


class TestModuleQualifiedCall:
    def test_import_module_qualified_extracted(self):
        result, _ = _make_repo({
            "b.py": "def foo(): pass\n",
            "a.py": "import b\ndef caller():\n    b.foo()\n",
        })
        c = _calls(result)
        assert any(src == "caller" and tgt == "foo" and tier == "EXTRACTED"
                   for src, tgt, tier in c)

    def test_dedup_free_and_module_qualified_same_target(self):
        # foo() and b.foo() both resolve to code::b.py::foo → only one edge
        result, _ = _make_repo({
            "b.py": "def foo(): pass\n",
            "a.py": "from b import foo\nimport b\ndef caller():\n    foo()\n    b.foo()\n",
        })
        edges = [e for e in result.edges
                 if e.kind == "calls" and e.source_id.endswith("::caller")]
        assert len(edges) == 1


class TestTypeAnnotatedCall:
    def test_annotated_member_call_inferred(self):
        result, _ = _make_repo({
            "b.py": "class Bar:\n    def method(self): pass\n",
            "a.py": "from b import Bar\ndef caller(obj: Bar):\n    obj.method()\n",
        })
        c = _calls(result)
        assert any(src == "caller" and tgt == "method" and tier == "INFERRED"
                   for src, tgt, tier in c)

    def test_annotated_member_call_confidence_08(self):
        result, _ = _make_repo({
            "b.py": "class Bar:\n    def method(self): pass\n",
            "a.py": "from b import Bar\ndef caller(obj: Bar):\n    obj.method()\n",
        })
        edge = next(e for e in result.edges
                    if e.kind == "calls" and e.source_id.endswith("::caller"))
        assert abs(edge.confidence - 0.8) < 1e-9

    def test_unannotated_member_call_skipped(self):
        result, _ = _make_repo({
            "b.py": "class Bar:\n    def method(self): pass\n",
            "a.py": "def caller(obj):\n    obj.method()\n",
        })
        c = _calls(result)
        # No annotation → cannot resolve → no calls edge
        assert not any(src == "caller" for src, tgt, tier in c)

    def test_ambiguous_class_name_skipped(self):
        # Two classes with same name → don't guess
        result, _ = _make_repo({
            "x.py": "class Foo:\n    def run(self): pass\n",
            "y.py": "class Foo:\n    def run(self): pass\n",
            "a.py": "def caller(obj: Foo):\n    obj.run()\n",
        })
        c = _calls(result)
        assert not any(src == "caller" for src, tgt, tier in c)


class TestCallsEdgesNotInherits:
    def test_calls_kind_is_calls(self):
        result, _ = _make_repo({"a.py": """
            class Foo:
                def run(self):
                    self.helper()
                def helper(self): pass
        """})
        for e in result.edges:
            if e.kind == "calls":
                assert e.confidence_tier in ("EXTRACTED", "INFERRED")

    def test_no_calls_edge_to_self(self):
        result, _ = _make_repo({"a.py": "def foo():\n    foo()\n"})
        # Recursive call → same source and target → filtered out
        edges = [e for e in result.edges
                 if e.kind == "calls" and e.source_id == e.target_id]
        assert len(edges) == 0


# ---------------------------------------------------------------------------
# Relative import resolution
# ---------------------------------------------------------------------------

class TestRelativeImports:
    def test_same_package_from_dot_import(self):
        # from . import utils  →  resolves to sibling module
        result, _ = _make_repo({
            "pkg/__init__.py": "",
            "pkg/utils.py": "def helper(): pass\n",
            "pkg/main.py": "from . import utils\ndef caller():\n    utils.helper()\n",
        })
        c = _calls(result)
        assert any(tgt == "helper" and tier == "EXTRACTED" for _, tgt, tier in c)

    def test_same_package_from_dot_name_import(self):
        # from .utils import helper  →  resolves helper to pkg.utils::helper
        result, _ = _make_repo({
            "pkg/__init__.py": "",
            "pkg/utils.py": "def helper(): pass\n",
            "pkg/main.py": "from .utils import helper\ndef caller():\n    helper()\n",
        })
        c = _calls(result)
        assert any(src == "caller" and tgt == "helper" and tier == "EXTRACTED"
                   for src, tgt, tier in c)

    def test_parent_package_from_dotdot_import(self):
        # from .. import shared  →  resolves to parent package
        result, _ = _make_repo({
            "__init__.py": "",
            "shared.py": "def util(): pass\n",
            "sub/__init__.py": "",
            "sub/worker.py": "from .. import shared\ndef run():\n    shared.util()\n",
        })
        c = _calls(result)
        assert any(tgt == "util" for _, tgt, _ in c)

    def test_relative_import_unresolvable_skipped(self):
        # from ...too_far import x  →  can't go above root, no crash
        result, _ = _make_repo({
            "a.py": "from ...missing import gone\ndef f():\n    gone()\n",
        })
        # No crash; call to gone() simply not resolved
        c = _calls(result)
        assert not any(tgt == "gone" for _, tgt, _ in c)


# ---------------------------------------------------------------------------
# MRO inheritance chain lookup
# ---------------------------------------------------------------------------

class TestMROInheritance:
    def test_self_call_resolves_to_parent_method(self):
        # Child inherits run() from Parent; self.run() in Child::go → Parent::run
        result, _ = _make_repo({"a.py": """
            class Parent:
                def run(self): pass

            class Child(Parent):
                def go(self):
                    self.run()
        """})
        c = _calls(result)
        assert any(src == "go" and tgt == "run" and tier == "EXTRACTED"
                   for src, tgt, tier in c)

    def test_overridden_method_resolves_to_child(self):
        # Child defines its own run(); self.run() in Child::go → Child::run
        result, _ = _make_repo({"a.py": """
            class Parent:
                def run(self): pass

            class Child(Parent):
                def run(self): pass
                def go(self):
                    self.run()
        """})
        edges = [e for e in result.edges
                 if e.kind == "calls" and e.source_id.endswith("::go")]
        assert len(edges) == 1
        assert edges[0].target_id.endswith("::run")
        # Must resolve to Child::run, not Parent::run
        assert "Child" in edges[0].target_id

    def test_multi_level_mro(self):
        # GrandChild → Child → Parent; grandchild.go() calls self.helper() defined only in Parent
        result, _ = _make_repo({"a.py": """
            class Parent:
                def helper(self): pass

            class Child(Parent):
                pass

            class GrandChild(Child):
                def go(self):
                    self.helper()
        """})
        c = _calls(result)
        assert any(src == "go" and tgt == "helper" for src, tgt, _ in c)

    def test_type_annotated_call_uses_mro(self):
        # obj: Parent → obj.run() resolves via MRO even through subclassing
        result, _ = _make_repo({
            "base.py": "class Parent:\n    def run(self): pass\n",
            "worker.py": textwrap.dedent("""\
                from base import Parent
                class Child(Parent):
                    pass
                def caller(obj: Parent):
                    obj.run()
            """),
        })
        c = _calls(result)
        assert any(tgt == "run" and tier == "INFERRED" for _, tgt, tier in c)


# ---------------------------------------------------------------------------
# Execution flow detection
# ---------------------------------------------------------------------------

class TestExecutionFlows:
    def _flows(self, result) -> list:
        return [n for n in result.nodes if n.kind == "flow"]

    def _step_of_edges(self, result) -> list:
        return [e for e in result.edges if e.kind == "step_of"]

    def test_entry_point_produces_flow_node(self):
        # main() with ≥2 callees and 0 callers → entry point → flow node
        result, _ = _make_repo({"a.py": """
            def main():
                step_a()
                step_b()
                step_c()

            def step_a(): pass
            def step_b(): pass
            def step_c(): pass
        """})
        flows = self._flows(result)
        assert len(flows) >= 1

    def test_flow_node_kind_is_flow(self):
        result, _ = _make_repo({"a.py": """
            def main():
                step_a()
                step_b()
                step_c()

            def step_a(): pass
            def step_b(): pass
            def step_c(): pass
        """})
        for fn in self._flows(result):
            assert fn.kind == "flow"
            assert fn.namespace == "code"

    def test_step_of_edges_are_inferred(self):
        result, _ = _make_repo({"a.py": """
            def main():
                step_a()
                step_b()
                step_c()

            def step_a(): pass
            def step_b(): pass
            def step_c(): pass
        """})
        for e in self._step_of_edges(result):
            assert e.confidence_tier == "INFERRED"
            assert abs(e.confidence - 0.7) < 1e-9

    def test_short_chain_no_flow(self):
        # Only 1 callee → chain < min_flow_length=3 → no flow
        result, _ = _make_repo({"a.py": """
            def main():
                only_one()

            def only_one(): pass
        """})
        flows = self._flows(result)
        assert len(flows) == 0

    def test_dominated_flow_removed(self):
        # run() → [a, b, c, d]; start() → [a, b]; start's steps ⊂ run's steps
        result, _ = _make_repo({"a.py": """
            def run():
                a()
                b()
                c()
                d()

            def start():
                a()
                b()

            def a(): pass
            def b(): pass
            def c(): pass
            def d(): pass
        """})
        flows = self._flows(result)
        flow_labels = [fn.label for fn in flows]
        # run flow should survive; start flow (if it exists) should be dominated
        # At minimum: if both exist, start's steps must be a strict subset of run's steps
        if len(flows) > 1:
            step_sets = [set(fn.metadata["steps"]) for fn in flows]
            for i, si in enumerate(step_sets):
                for j, sj in enumerate(step_sets):
                    if i != j:
                        assert not (si < sj), "dominated flow was not removed"

    def test_flows_container_node_present(self):
        # When flows exist, a kind="flows" container is added
        result, _ = _make_repo({"a.py": """
            def main():
                step_a()
                step_b()
                step_c()

            def step_a(): pass
            def step_b(): pass
            def step_c(): pass
        """})
        flows = self._flows(result)
        if flows:  # flows detected
            container = [n for n in result.nodes if n.kind == "flows"]
            assert len(container) == 1
