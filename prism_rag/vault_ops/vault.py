"""
Obsidian Vault MCP — Vault path management and security guardrails

Three-layer defense in depth:
  L1: Path sandbox — realpath check that the path is inside VAULT_BASE_DIR
  L2: Sensitive directory blocklist — .obsidian/, .git/, .trash/, node_modules/
  L3: Deletion protection — move to .trash/ instead of hard delete (implemented at the tool layer)

All file path operations must pass through resolve_path() for validation.
"""

import os
from pathlib import Path

from prism_rag.vault_ops.errors import VaultErrorCode, fail

# Sensitive directory blocklist (writes and deletes prohibited)
_BLOCKED_DIRS = frozenset({
    ".obsidian",
    ".git",
    ".trash",
    "node_modules",
    ".DS_Store",
})

# Allowed file extensions
_ALLOWED_EXTENSIONS = frozenset({
    ".md", ".markdown", ".txt", ".canvas", ".base",
})


class Vault:
    """Vault instance — bound to a base_dir, provides path resolution and security validation."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).resolve()
        if not self.base_dir.is_dir():
            raise ValueError(f"Vault directory does not exist: {self.base_dir}")

    def resolve_path(self, relative_path: str) -> Path | dict:
        """
        Resolve a Vault-relative path to an absolute path.
        Returns a Path on success, or a fail() error response on failure.

        Security validation:
          1. Resolved path must be inside base_dir (prevents path traversal)
          2. Must not be inside a blocklisted directory
        """
        # Sanitize input
        cleaned = relative_path.strip().lstrip("/").lstrip("\\")
        if not cleaned:
            return fail(VaultErrorCode.VALIDATION_ERROR, "Path must not be empty")

        # Resolve absolute path
        abs_path = (self.base_dir / cleaned).resolve()

        # L1: Path sandbox
        try:
            abs_path.relative_to(self.base_dir)
        except ValueError:
            return fail(
                VaultErrorCode.PATH_TRAVERSAL,
                f"Path traversal: {relative_path} resolves outside the Vault",
            )

        # L2: Sensitive directory blocklist
        rel_parts = abs_path.relative_to(self.base_dir).parts
        for part in rel_parts:
            if part in _BLOCKED_DIRS:
                return fail(
                    VaultErrorCode.PERMISSION_DENIED,
                    f"Access to sensitive directory is forbidden: {part}/",
                )

        return abs_path

    def resolve_dir(self, relative_path: str = "") -> Path | dict:
        """Resolve a directory path. Empty string = vault root directory."""
        if not relative_path or relative_path == ".":
            return self.base_dir

        abs_path = (self.base_dir / relative_path.strip().lstrip("/")).resolve()

        try:
            abs_path.relative_to(self.base_dir)
        except ValueError:
            return fail(
                VaultErrorCode.PATH_TRAVERSAL,
                f"Directory path traversal: {relative_path}",
            )

        # Blocklist check
        rel_parts = abs_path.relative_to(self.base_dir).parts
        for part in rel_parts:
            if part in _BLOCKED_DIRS:
                return fail(
                    VaultErrorCode.PERMISSION_DENIED,
                    f"Access to sensitive directory is forbidden: {part}/",
                )

        return abs_path

    def relative_path(self, abs_path: Path) -> str:
        """Convert an absolute path to a Vault-relative path string."""
        return str(abs_path.relative_to(self.base_dir))

    def is_note(self, path: Path) -> bool:
        """Return True if the path is a note file (determined by extension)."""
        return path.suffix.lower() in _ALLOWED_EXTENSIONS
