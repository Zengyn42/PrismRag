"""Pass 2: media extraction — PDF → text, image/audio stubs."""
from __future__ import annotations

import logging
from pathlib import Path

from prism_rag.ingest.vault_loader import VaultMedia
from prism_rag.store.graph import KnowledgeGraph, Node

logger = logging.getLogger(__name__)

_MAX_CONTENT_CHARS = 30_000


def extract_pdf(path: Path) -> str:
    """Extract text from a PDF using pypdf.

    Pages are concatenated with "\\n\\n--- Page N ---\\n\\n" separators.
    Returns empty string if the PDF has no extractable text.
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError(
            "pypdf is required for Pass 2 PDF extraction. "
            "Install with: pip install prism-rag"
        ) from e

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        logger.warning(f"[media_extractor] failed to open PDF {path}: {e}")
        return ""

    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"[media_extractor] page {i} of {path} failed: {e}")
            text = ""
        text = text.strip()
        if text:
            pages.append(f"--- Page {i} ---\n\n{text}")

    if not pages:
        logger.warning(f"[media_extractor] {path} has no extractable text (scan-only?)")
        return ""

    full = "\n\n".join(pages)
    if len(full) > _MAX_CONTENT_CHARS:
        full = full[:_MAX_CONTENT_CHARS] + "\n\n...[truncated]"
    return full


def extract_image(path: Path) -> str:
    """Image extraction — NOT YET IMPLEMENTED."""
    raise NotImplementedError("Image extraction deferred to future step")


def extract_audio(path: Path) -> str:
    """Audio extraction — NOT YET IMPLEMENTED."""
    raise NotImplementedError("Audio extraction deferred to future step")


def add_media_nodes(
    graph: KnowledgeGraph,
    media_files: list[VaultMedia],
) -> int:
    """Add a Node per media file to the graph (PDFs only today).

    Returns the count of nodes added.
    """
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    added = 0
    for m in media_files:
        if m.kind == "pdf":
            text = extract_pdf(m.path)
            tokens = max(1, len(_enc.encode(text))) if text else 0
            node = Node(
                id=m.id,
                label=m.label,
                kind="pdf",
                source_file=str(m.relative_path),
                content=text,
                content_hash=m.content_hash,
                tokens=tokens,
            )
            graph.add_node(node)
            added += 1
        elif m.kind in ("image", "audio"):
            logger.info(f"[media_extractor] skipping {m.kind}: {m.path} (not implemented)")
        else:
            logger.info(f"[media_extractor] unknown kind '{m.kind}': {m.path}")
    return added
