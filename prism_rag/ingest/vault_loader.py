"""Obsidian vault discovery and frontmatter parsing.

This module is responsible for Pass 1a of the pipeline:
1. Walk the vault recursively to find all .md files
2. Parse each file's YAML frontmatter
3. Return a VaultDocument wrapper with content + metadata

Symlinks, hidden directories, and Obsidian internals (.obsidian, .trash) are skipped.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter


_DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        ".git", ".obsidian", ".trash", ".DS_Store",
        "__pycache__", "node_modules",
        ".venv", "venv", ".env",           # Python virtual environments
        "logs",                             # runtime log directories
        ".pytest_cache", ".mypy_cache",    # tool caches
        "dist", "build",                   # build artifacts
        ".prismrag",                       # PrismRag output (avoid self-ingestion)
    }
)


@dataclass
class VaultDocument:
    """A single Obsidian markdown document."""

    path: Path
    vault_root: Path
    content: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    @classmethod
    def from_path(cls, path: Path, vault_root: Path) -> "VaultDocument":
        with path.open(encoding="utf-8") as f:
            post = frontmatter.load(f)
        content = post.content or ""
        meta = dict(post.metadata or {})
        hash_hex = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return cls(
            path=path,
            vault_root=vault_root,
            content=content,
            frontmatter=meta,
            content_hash=f"sha256:{hash_hex}",
        )

    @property
    def relative_path(self) -> Path:
        return self.path.relative_to(self.vault_root)

    @property
    def id(self) -> str:
        """Stable node ID.

        If frontmatter declares a knowledge_id (Phase 2 atomic node), use it.
        Otherwise fall back to relative path without .md extension, POSIX-style.
        """
        kid = self.frontmatter.get("knowledge_id")
        if kid:
            return str(kid)
        return self.relative_path.with_suffix("").as_posix()

    @property
    def label(self) -> str:
        """Human-readable label.

        For knowledge nodes (frontmatter has knowledge_id), applies the
        three-layer fallback: frontmatter title → clean_slug → stem.
        For regular notes, returns the filename stem unchanged.
        """
        if self.frontmatter.get("knowledge_id"):
            from prism_rag.ingest.label_resolver import resolve_knowledge_label
            return resolve_knowledge_label(self.frontmatter, self.path.stem)
        return self.path.stem

    @property
    def aliases(self) -> list[str]:
        raw = self.frontmatter.get("aliases", [])
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list):
            return [str(a) for a in raw]
        return []

    @property
    def frontmatter_tags(self) -> list[str]:
        """Tags declared in YAML frontmatter (as opposed to inline #tags)."""
        raw = self.frontmatter.get("tags", [])
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list):
            return [str(t) for t in raw]
        return []

    @property
    def category(self) -> str | None:
        cat = self.frontmatter.get("category")
        return str(cat) if cat else None


@dataclass
class VaultMedia:
    """A non-markdown vault file (PDF, image, audio)."""

    path: Path
    vault_root: Path
    kind: str  # "pdf" | "image" | "audio" | "unknown"
    content_hash: str = ""

    @classmethod
    def from_path(cls, path: Path, vault_root: Path) -> "VaultMedia":
        ext = path.suffix.lower()
        if ext == ".pdf":
            kind = "pdf"
        elif ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            kind = "image"
        elif ext in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
            kind = "audio"
        else:
            kind = "unknown"
        hash_hex = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(
            path=path,
            vault_root=vault_root,
            kind=kind,
            content_hash=f"sha256:{hash_hex}",
        )

    @property
    def relative_path(self) -> Path:
        return self.path.relative_to(self.vault_root)

    @property
    def id(self) -> str:
        """Stable ID: relative path without extension, POSIX-style."""
        return self.relative_path.with_suffix("").as_posix()

    @property
    def label(self) -> str:
        return self.path.stem


# Supported media extensions (MVP = PDF only; image/audio are stubs).
_MEDIA_EXTENSIONS: frozenset[str] = frozenset({".pdf"})


def discover_markdown_files(
    vault_root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Path]:
    """Recursively find all .md files under vault_root, skipping excluded dirs.

    Returns a sorted list for deterministic ordering.
    """
    if not vault_root.exists():
        raise FileNotFoundError(f"Vault root does not exist: {vault_root}")
    if not vault_root.is_dir():
        raise NotADirectoryError(f"Vault root is not a directory: {vault_root}")

    results: list[Path] = []
    for path in vault_root.rglob("*.md"):
        # Skip if any ancestor directory is excluded
        rel_parts = path.relative_to(vault_root).parts
        if any(part in exclude_dirs for part in rel_parts):
            continue
        results.append(path)
    return sorted(results)


def discover_vault_files(
    vault_root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Path]:
    """Recursively find .md and supported media files under vault_root.

    Like discover_markdown_files, but includes PDF (and in the future, images/audio).
    """
    if not vault_root.exists():
        raise FileNotFoundError(f"Vault root does not exist: {vault_root}")
    if not vault_root.is_dir():
        raise NotADirectoryError(f"Vault root is not a directory: {vault_root}")

    results: list[Path] = []
    all_extensions = {".md"} | _MEDIA_EXTENSIONS
    for path in vault_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in all_extensions:
            continue
        rel_parts = path.relative_to(vault_root).parts
        if any(part in exclude_dirs for part in rel_parts):
            continue
        results.append(path)
    return sorted(results)


def load_vault(
    vault_root: Path,
) -> tuple[list["VaultDocument"], set[tuple[str, str]]]:
    """Load all markdown files from the vault.

    Returns:
        (documents, live_sha_set) where live_sha_set = {(node_id, content_hash)}
        for use in embed_cache GC.
    """
    vault_root = vault_root.expanduser().resolve()
    paths = discover_markdown_files(vault_root)
    docs = [VaultDocument.from_path(p, vault_root) for p in paths]
    live_sha_set = {(doc.id, doc.content_hash) for doc in docs}
    return docs, live_sha_set
