"""PrismRag CLI entrypoint (Typer).

Phase 1 MVP commands:
  prism-rag ingest    Run the AST-only pipeline on the configured vault
  prism-rag info      Print stats about an existing graph.json
  prism-rag version   Print the package version

Future commands (not yet implemented):
  prism-rag embed     Run Pass 3 (Gemini Embedding 2 + similarity edges)
  prism-rag media     Run Pass 2 (image/PDF/audio ingestion)
  prism-rag serve     Start the MCP Server
  prism-rag query     Ad-hoc graph traversal query
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from prism_rag import __version__
from prism_rag.cluster.leiden import run_leiden
from prism_rag.config import PrismRagSettings
from prism_rag.ingest.ast_extractor import extract_ast
from prism_rag.ingest.vault_loader import load_vault
from prism_rag.report.graph_report import generate_report
from prism_rag.store.graph import KnowledgeGraph

# Configure basic logging; users can override via LOGLEVEL env var if needed
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = typer.Typer(
    name="prism-rag",
    help="无垠智穹图优先 RAG 系统 (v4.0)",
    no_args_is_help=True,
)


def _resolve_settings(
    vault: Path | None,
    output: Path | None,
) -> PrismRagSettings:
    """Build a Settings object, with CLI options overriding env vars."""
    settings = PrismRagSettings()
    if vault is not None:
        settings.vault_path = vault.expanduser().resolve()
    if output is not None:
        settings.data_dir = output.expanduser().resolve()
    return settings


@app.command()
def ingest(
    vault: Path = typer.Option(  # noqa: B008
        None,
        "--vault",
        "-v",
        help="Vault path (overrides PRISM_VAULT_PATH)",
    ),
    output: Path = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Output dir (overrides PRISM_DATA_DIR)",
    ),
    skip_cluster: bool = typer.Option(
        False,
        "--skip-cluster",
        help="Skip Leiden community detection (useful for quick AST-only runs)",
    ),
    skip_report: bool = typer.Option(
        False,
        "--skip-report",
        help="Skip GRAPH_REPORT.md generation",
    ),
) -> None:
    """Build the knowledge graph from the vault (AST-only MVP).

    Pipeline passes executed:
      1a. Vault loading (discovery + frontmatter)
      1b. AST extraction (wikilinks, tags, frontmatter, category)
      4.  Leiden community detection (unless --skip-cluster)
      5.  JSON persistence + GRAPH_REPORT.md (unless --skip-report)

    NOT run in MVP:
      2.  Media (image/PDF/audio) — requires [media] extras
      3.  Embedding + similarity edges — requires Gemini API key
    """
    settings = _resolve_settings(vault, output)

    typer.secho(f"📂 Vault:  {settings.vault_path}", fg=typer.colors.CYAN)
    typer.secho(f"📁 Output: {settings.data_dir}", fg=typer.colors.CYAN)
    typer.echo("")

    if not settings.vault_path.exists():
        typer.secho(f"❌ Vault path does not exist: {settings.vault_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # ── Pass 1a: Load vault ──────────────────────────────────────────
    typer.secho("🔍 Pass 1a: Discovering markdown files...", fg=typer.colors.BLUE)
    docs = load_vault(settings.vault_path)
    typer.echo(f"   Found {len(docs)} markdown files")

    if not docs:
        typer.secho("⚠️  No markdown files found — nothing to do.", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    # ── Pass 1b: AST extraction ──────────────────────────────────────
    typer.secho("\n🧩 Pass 1b: Extracting wikilinks, tags, frontmatter...", fg=typer.colors.BLUE)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)
    typer.echo(f"   Nodes: {graph.node_count} · Edges: {graph.edge_count}")

    # ── Pass 4: Leiden clustering ────────────────────────────────────
    if skip_cluster:
        typer.secho("\n⏭  Pass 4: Leiden clustering (skipped by --skip-cluster)", fg=typer.colors.YELLOW)
    else:
        typer.secho("\n🧠 Pass 4: Leiden community detection...", fg=typer.colors.BLUE)
        n_communities = run_leiden(
            graph,
            resolution=settings.leiden_resolution,
            seed=settings.leiden_seed,
            god_nodes_per_community=settings.god_nodes_per_community,
        )
        typer.echo(f"   Communities: {n_communities}")

    # ── Pass 5: Persistence + report ─────────────────────────────────
    typer.secho("\n💾 Pass 5: Persisting graph...", fg=typer.colors.BLUE)
    graph.save(settings.graph_path)
    typer.echo(f"   → {settings.graph_path}")

    if skip_report:
        typer.secho("⏭  GRAPH_REPORT.md (skipped by --skip-report)", fg=typer.colors.YELLOW)
    else:
        typer.secho("\n📝 Pass 5: Generating GRAPH_REPORT.md...", fg=typer.colors.BLUE)
        generate_report(graph, settings.report_path, vault_root=settings.vault_path)
        typer.echo(f"   → {settings.report_path}")

    typer.secho("\n✅ Ingest complete.", fg=typer.colors.GREEN)


@app.command()
def info(
    graph_path: Path = typer.Option(  # noqa: B008
        None,
        "--graph",
        "-g",
        help="Path to graph.json (default: from settings)",
    ),
) -> None:
    """Print stats about an existing graph.json."""
    settings = PrismRagSettings()
    path = graph_path or settings.graph_path

    if not path.exists():
        typer.secho(f"❌ Graph not found: {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    graph = KnowledgeGraph.load(path)
    typer.secho(f"📊 Graph: {path}", fg=typer.colors.CYAN)
    typer.echo(f"   Nodes:       {graph.node_count}")
    typer.echo(f"   Edges:       {graph.edge_count}")
    typer.echo(f"   Communities: {len(graph.communities)}")

    if graph.communities:
        typer.echo("\n   Top 5 communities by size:")
        sorted_comms = sorted(
            graph.communities.values(),
            key=lambda c: c.member_count,
            reverse=True,
        )[:5]
        for comm in sorted_comms:
            typer.echo(
                f"   - {comm.id}: {comm.member_count} members ({comm.label})"
            )


@app.command()
def version() -> None:
    """Print PrismRag version."""
    typer.echo(f"PrismRag v{__version__}")


if __name__ == "__main__":
    app()
