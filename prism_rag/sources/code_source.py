"""Code source extractor — wraps the existing CodeParser for the sources API.

Adapts prism_rag.ingest.code_parser.CodeParser to the SourceExtractor protocol.
No parsing logic is reimplemented here.
"""

from __future__ import annotations

from pathlib import Path

from prism_rag.ingest.code_parser import CodeParser, _find_py_files
from prism_rag.ingest.parse_result import ParseResult
from prism_rag.sources.base import SourceExtractor, SourceKind


class CodeSourceExtractor(SourceExtractor):
    """Extracts code:: graph nodes from a Python source repository."""

    @property
    def kind(self) -> SourceKind:
        return "code"

    def discover(self, root: Path) -> list[Path]:
        root = root.expanduser().resolve()
        if root.is_file():
            return [root]
        return _find_py_files(root)

    def parse(self, root: Path) -> ParseResult:
        parser = CodeParser()
        return parser.parse(root)
