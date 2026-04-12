"""Pass 3a: Compute embeddings for all text nodes using Gemini Embedding 2.

This module handles:
- Batch embedding computation (respects API rate limits)
- Matryoshka dimensionality control (default 768)
- Content truncation for documents exceeding 8192 token input limit
- Privacy tier enforcement (paid vs free)

Embeddings are computed INDEX-TIME ONLY. They are NOT used for query-time retrieval.
Their sole purpose is to feed into similarity_linker.py (Pass 3b) which generates
`semantically_similar_to` edges in the graph.

Usage:
    from prism_rag.ingest.embedder import compute_embeddings
    vectors = compute_embeddings(graph, settings)
    # vectors: dict[node_id, list[float]]
"""

from __future__ import annotations

import logging
import time
from typing import Any

from google import genai
from google.genai.types import EmbedContentConfig

from prism_rag.config import PrismRagSettings
from prism_rag.store.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Gemini Embedding 2 limits
_MODEL = "gemini-embedding-2-preview"
_MAX_INPUT_CHARS = 30_000  # ~8192 tokens; truncate beyond this
_BATCH_SIZE = 20  # documents per API call (avoid overwhelming the API)
_RATE_LIMIT_DELAY = 0.5  # seconds between batches (free tier is rate-limited)


def _truncate(text: str, max_chars: int = _MAX_INPUT_CHARS) -> str:
    """Truncate text to fit within Gemini's input limit."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _get_embeddable_nodes(graph: KnowledgeGraph) -> list[tuple[str, str]]:
    """Extract (node_id, content) pairs for all nodes worth embedding.

    Only embeds nodes with actual content (notes). Tags and categories
    are structural nodes with no semantic content to embed.
    """
    pairs: list[tuple[str, str]] = []
    for node_id, data in graph.g.nodes(data=True):
        kind = data.get("kind", "")
        content = data.get("content", "")
        if kind in ("note", "image", "pdf", "audio") and content.strip():
            pairs.append((node_id, content))
    return pairs


def compute_embeddings(
    graph: KnowledgeGraph,
    settings: PrismRagSettings,
    dimensionality: int = 768,
) -> dict[str, list[float]]:
    """Compute Gemini Embedding 2 vectors for all embeddable nodes.

    Args:
        graph: Knowledge graph with nodes populated (after Pass 1).
        settings: Must have gemini_api_key set.
        dimensionality: Output vector dimension (Matryoshka). Default 768.

    Returns:
        dict mapping node_id → embedding vector (list of floats).

    Raises:
        ValueError: If API key is not configured.
    """
    if not settings.gemini_api_key:
        raise ValueError(
            "PRISM_GEMINI_API_KEY is required for Pass 3 (embedding). "
            "Get one at https://aistudio.google.com/apikey"
        )

    if settings.privacy_tier == "free":
        logger.warning(
            "[embedder] privacy_tier=free: Gemini free tier may use your data "
            "for model training. Set PRISM_PRIVACY_TIER=paid for production use."
        )

    client = genai.Client(api_key=settings.gemini_api_key)
    config = EmbedContentConfig(output_dimensionality=dimensionality)

    nodes_to_embed = _get_embeddable_nodes(graph)
    if not nodes_to_embed:
        logger.info("[embedder] no embeddable nodes found")
        return {}

    logger.info(
        f"[embedder] computing embeddings for {len(nodes_to_embed)} nodes "
        f"(model={_MODEL}, dim={dimensionality})"
    )

    vectors: dict[str, list[float]] = {}
    total = len(nodes_to_embed)

    # embed_content with a list of strings concatenates them into ONE embedding.
    # We must call per-document to get separate vectors.
    for i, (node_id, content) in enumerate(nodes_to_embed):
        truncated = _truncate(content)
        try:
            result = client.models.embed_content(
                model=_MODEL,
                contents=truncated,
                config=config,
            )
            vectors[node_id] = result.embeddings[0].values
        except Exception as e:
            logger.error(f"[embedder] node {node_id} failed: {e}")
            continue

        # Log progress every 20 nodes
        if (i + 1) % 20 == 0 or i + 1 == total:
            logger.info(f"[embedder] progress: {i + 1}/{total}")

        # Rate limiting (free tier has ~1500 RPM, pace at ~2 req/sec)
        if i + 1 < total:
            time.sleep(_RATE_LIMIT_DELAY)

    logger.info(
        f"[embedder] done: {len(vectors)}/{total} nodes embedded successfully"
    )
    return vectors
