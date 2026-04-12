"""PrismRag CLI entrypoint (Typer).

Commands:
  prism-rag ingest    Build the knowledge graph (Pass 1 + 3 + 4 + 5)
  prism-rag query     Query the graph via BFS/DFS traversal
  prism-rag info      Print stats about an existing graph.json
  prism-rag version   Print the package version

Future:
  prism-rag serve     Start the MCP Server
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = typer.Typer(
    name="prism-rag",
    help="无垠智穹图优先 RAG 系统 (v4.0)",
    no_args_is_help=True,
)


def _resolve_settings(vault: Path | None, output: Path | None) -> PrismRagSettings:
    settings = PrismRagSettings()
    if vault is not None:
        settings.vault_path = vault.expanduser().resolve()
    if output is not None:
        settings.data_dir = output.expanduser().resolve()
    return settings


@app.command()
def ingest(
    vault: Path = typer.Option(None, "--vault", "-v", help="Vault path"),
    output: Path = typer.Option(None, "--output", "-o", help="Output dir"),
    skip_cluster: bool = typer.Option(False, "--skip-cluster"),
    skip_report: bool = typer.Option(False, "--skip-report"),
    skip_embed: bool = typer.Option(False, "--skip-embed", help="Skip Pass 3 embedding + similarity edges"),
) -> None:
    """Build the knowledge graph from the vault.

    Pipeline: Pass 1 (AST) → Pass 3 (Embedding) → Pass 4 (Leiden) → Pass 5 (Persist + Report)
    """
    settings = _resolve_settings(vault, output)

    typer.secho(f"📂 Vault:  {settings.vault_path}", fg=typer.colors.CYAN)
    typer.secho(f"📁 Output: {settings.data_dir}", fg=typer.colors.CYAN)
    typer.echo("")

    if not settings.vault_path.exists():
        typer.secho(f"❌ Vault path does not exist: {settings.vault_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # ── Pass 1a: Load vault ──
    typer.secho("🔍 Pass 1a: Discovering markdown files...", fg=typer.colors.BLUE)
    docs = load_vault(settings.vault_path)
    typer.echo(f"   Found {len(docs)} markdown files")

    if not docs:
        typer.secho("⚠️  No markdown files found.", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    # ── Pass 1b: AST extraction ──
    typer.secho("\n🧩 Pass 1b: Extracting wikilinks, tags, frontmatter...", fg=typer.colors.BLUE)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)
    typer.echo(f"   Nodes: {graph.node_count} · Edges: {graph.edge_count}")

    # ── Pass 3: Embedding + similarity edges ──
    if skip_embed:
        typer.secho("\n⏭  Pass 3: Embedding (skipped by --skip-embed)", fg=typer.colors.YELLOW)
    elif not settings.gemini_api_key:
        typer.secho(
            "\n⏭  Pass 3: Embedding (skipped — no PRISM_GEMINI_API_KEY set)",
            fg=typer.colors.YELLOW,
        )
    else:
        from prism_rag.ingest.embedder import compute_embeddings
        from prism_rag.ingest.similarity_linker import link_similar_nodes

        typer.secho("\n🧬 Pass 3a: Computing embeddings (Gemini Embedding 2)...", fg=typer.colors.BLUE)
        vectors = compute_embeddings(graph, settings)
        typer.echo(f"   Embedded {len(vectors)} nodes (dim=768)")

        typer.secho("\n🔗 Pass 3b: Generating similarity edges...", fg=typer.colors.BLUE)
        n_new = link_similar_nodes(graph, vectors, settings)
        typer.echo(f"   New edges: {n_new} · Total edges: {graph.edge_count}")

    # ── Pass 4: Leiden clustering ──
    if skip_cluster:
        typer.secho("\n⏭  Pass 4: Leiden clustering (skipped)", fg=typer.colors.YELLOW)
    else:
        typer.secho("\n🧠 Pass 4: Leiden community detection...", fg=typer.colors.BLUE)
        n_communities = run_leiden(
            graph,
            resolution=settings.leiden_resolution,
            seed=settings.leiden_seed,
            god_nodes_per_community=settings.god_nodes_per_community,
        )
        typer.echo(f"   Communities: {n_communities}")

    # ── Pass 5: Persistence + report ──
    typer.secho("\n💾 Pass 5: Persisting graph...", fg=typer.colors.BLUE)
    graph.save(settings.graph_path)
    typer.echo(f"   → {settings.graph_path}")

    if not skip_report:
        typer.secho("\n📝 Pass 5: Generating GRAPH_REPORT.md...", fg=typer.colors.BLUE)
        generate_report(graph, settings.report_path, vault_root=settings.vault_path)
        typer.echo(f"   → {settings.report_path}")

    # ── Visualization (if pyvis is installed) ──
    try:
        from prism_rag.report.visualize import generate_html

        viz_path = settings.data_dir / "graph.html"
        typer.secho("\n🎨 Generating interactive visualization...", fg=typer.colors.BLUE)
        generate_html(graph, viz_path)
        typer.echo(f"   → {viz_path}")
    except ImportError:
        typer.secho(
            "\n⏭  Visualization skipped (install pyvis: pip install prism-rag[viz])",
            fg=typer.colors.YELLOW,
        )

    typer.secho("\n✅ Ingest complete.", fg=typer.colors.GREEN)


@app.command()
def add(
    file: Path = typer.Argument(..., help="Path to the .md file (absolute or vault-relative)"),
    skip_embed: bool = typer.Option(False, "--skip-embed", help="Skip embedding (faster)"),
) -> None:
    """Incrementally add or update a single file in the graph.

    Much faster than full ingest — only processes the one file,
    then re-runs Leiden and persists.
    """
    from prism_rag.ingest.incremental import ingest_file

    settings = PrismRagSettings()
    typer.secho(f"📄 Adding: {file}", fg=typer.colors.CYAN)

    try:
        result = ingest_file(file, settings=settings, skip_embed=skip_embed)
    except (FileNotFoundError, ValueError) as e:
        typer.secho(f"❌ {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.secho(f"\n✅ {result['action'].capitalize()}: {result['node_id']}", fg=typer.colors.GREEN)
    typer.echo(f"   AST edges: +{result['ast_edges']}")
    typer.echo(f"   Similarity edges: +{result['similarity_edges']}")
    typer.echo(f"   Graph: {result['total_nodes']} nodes · {result['total_edges']} edges · {result['communities']} communities")


@app.command()
def query(
    q: str = typer.Argument(..., help="Query string (node label, topic, or keyword)"),
    graph_path: Path = typer.Option(None, "--graph", "-g", help="Path to graph.json"),
    budget: int = typer.Option(4000, "--budget", "-b", help="Token budget"),
    mode: str = typer.Option("bfs", "--mode", "-m", help="Traversal mode: bfs or dfs"),
    show_content: bool = typer.Option(False, "--content", "-c", help="Show node content"),
) -> None:
    """Query the knowledge graph via BFS/DFS traversal."""
    from prism_rag.retrieve.bfs import bfs_traverse
    from prism_rag.retrieve.dfs import dfs_traverse
    from prism_rag.retrieve.entry import resolve_entry_point

    settings = PrismRagSettings()
    path = graph_path or settings.graph_path

    if not path.exists():
        typer.secho(f"❌ Graph not found: {path}", fg=typer.colors.RED, err=True)
        typer.secho("   Run 'prism-rag ingest' first.", err=True)
        raise typer.Exit(1)

    graph = KnowledgeGraph.load(path)

    # Resolve entry point
    entry = resolve_entry_point(graph, q)
    if entry is None:
        typer.secho(f"❌ No matching node for query: {q!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    entry_label = graph.g.nodes[entry].get("label", entry)
    typer.secho(f"🎯 Entry: {entry_label} ({entry})", fg=typer.colors.CYAN)

    # Traverse
    if mode == "dfs":
        nodes = dfs_traverse(graph, entry, budget=budget)
    else:
        nodes = bfs_traverse(graph, entry, budget=budget)

    typer.secho(
        f"📊 Traversed {len(nodes)} nodes · ~{sum(n.get('tokens', 0) for n in nodes):,} tokens",
        fg=typer.colors.CYAN,
    )
    typer.echo("")

    for i, node in enumerate(nodes):
        kind = node.get("kind", "?")
        label = node.get("label", node["id"])
        tokens = node.get("tokens", 0)
        community = node.get("community_id", "—")
        marker = "►" if i == 0 else " "

        typer.echo(f"  {marker} [{kind}] {label} ({tokens} tokens, {community})")

        if show_content and node.get("content"):
            content = node["content"][:300]
            if len(node.get("content", "")) > 300:
                content += "..."
            typer.secho(f"    {content}", fg=typer.colors.WHITE, dim=True)
            typer.echo("")


@app.command()
def info(
    graph_path: Path = typer.Option(None, "--graph", "-g", help="Path to graph.json"),
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

    # Edge type breakdown
    edge_types: dict[str, int] = {}
    for _, _, data in graph.g.edges(data=True):
        r = data.get("relation", "?")
        edge_types[r] = edge_types.get(r, 0) + 1
    if edge_types:
        typer.echo("\n   Edge types:")
        for r, count in sorted(edge_types.items(), key=lambda x: -x[1]):
            typer.echo(f"     {r}: {count}")

    if graph.communities:
        typer.echo("\n   Top 5 communities:")
        sorted_comms = sorted(
            graph.communities.values(), key=lambda c: c.member_count, reverse=True
        )[:5]
        for comm in sorted_comms:
            typer.echo(f"   - {comm.id}: {comm.member_count} members ({comm.label})")


@app.command()
def serve(
    transport: str = typer.Option("stdio", "--transport", "-t", help="MCP transport: stdio or sse"),
) -> None:
    """Start the PrismRag MCP Server.

    Exposes 5 tools for knowledge graph queries:
    search_knowledge, explain_node, trace_path, list_communities, explore_community.
    """
    from prism_rag.mcp_server.server import run_server

    settings = PrismRagSettings()
    if not settings.graph_path.exists():
        typer.secho(
            f"❌ Graph not found: {settings.graph_path}\n   Run 'prism-rag ingest' first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    typer.secho(f"🚀 Starting MCP Server (transport={transport})...", fg=typer.colors.GREEN)
    typer.echo(f"   Graph: {settings.graph_path}")
    run_server(transport=transport)


@app.command()
def version() -> None:
    """Print PrismRag version."""
    typer.echo(f"PrismRag v{__version__}")


if __name__ == "__main__":
    app()
