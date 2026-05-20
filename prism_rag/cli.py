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
from prism_rag.cli_atomize import app as atomize_app
from prism_rag.cluster.leiden import run_leiden
from prism_rag.config import PrismRagSettings
from prism_rag.ingest.ast_extractor import extract_ast
from prism_rag.ingest.embedder import detect_model_device, gc_embed_cache, _load_embed_cache
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
    namespace: str = typer.Option("", "--namespace", "-n", help="Namespace for this vault's graph"),
    skip_cluster: bool = typer.Option(False, "--skip-cluster"),
    skip_report: bool = typer.Option(False, "--skip-report"),
    skip_embed: bool = typer.Option(False, "--skip-embed", help="Skip Pass 3 embedding + similarity edges"),
    no_embedding: bool = typer.Option(
        False, "--no-embedding",
        help="Alias for --skip-embed; skip Pass 3 entirely (for offline testing)",
    ),
) -> None:
    """Build the knowledge graph from the vault.

    Pipeline: Pass 1 (AST) → Pass 3 (Embedding) → Pass 4 (Leiden) → Pass 5 (Persist + Report)
    """
    skip_embed = skip_embed or no_embedding

    if namespace and output is not None:
        output = output / namespace
    elif namespace:
        # Apply namespace subdirectory to the default output path
        settings_tmp = PrismRagSettings()
        output = settings_tmp.data_dir / namespace

    settings = _resolve_settings(vault, output)

    typer.secho(f"📂 Vault:  {settings.vault_path}", fg=typer.colors.CYAN)
    typer.secho(f"📁 Output: {settings.data_dir}", fg=typer.colors.CYAN)
    typer.echo("")

    if not settings.vault_path.exists():
        typer.secho(f"❌ Vault path does not exist: {settings.vault_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # ── Pass 1a: Load vault ──
    typer.secho("🔍 Pass 1a: Discovering markdown files...", fg=typer.colors.BLUE)
    docs, live_sha_set = load_vault(settings.vault_path)
    typer.echo(f"   Found {len(docs)} markdown files")

    if not docs:
        typer.secho("⚠️  No markdown files found.", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    # ── Pass 1b: AST extraction ──
    typer.secho("\n🧩 Pass 1b: Extracting wikilinks, tags, frontmatter...", fg=typer.colors.BLUE)
    graph = KnowledgeGraph()
    extract_ast(graph, docs)
    typer.echo(f"   Nodes: {graph.node_count} · Edges: {graph.edge_count}")

    # ── Pass 2: Media extraction ──
    from prism_rag.ingest.media_extractor import add_media_nodes
    from prism_rag.ingest.vault_loader import VaultMedia, discover_vault_files

    typer.secho("\n📄 Pass 2: Extracting PDF content...", fg=typer.colors.BLUE)
    media_paths = [
        p for p in discover_vault_files(settings.vault_path)
        if p.suffix.lower() == ".pdf"
    ]
    media = [VaultMedia.from_path(p, settings.vault_path) for p in media_paths]
    n_media = add_media_nodes(graph, media)
    typer.echo(f"   PDF nodes: {n_media}")

    # ── Pass 3: Embedding + similarity edges ──
    _embed_ready = (
        settings.embed_backend == "ollama"
        or (settings.embed_backend == "gemini" and settings.gemini_api_key)
    )
    if skip_embed:
        typer.secho("\n⏭  Pass 3: Embedding (skipped by --skip-embed)", fg=typer.colors.YELLOW)
    elif not _embed_ready:
        typer.secho(
            f"\n⏭  Pass 3: Embedding (skipped — "
            f"embed_backend={settings.embed_backend!r} requires PRISM_GEMINI_API_KEY)",
            fg=typer.colors.YELLOW,
        )
    else:
        from prism_rag.ingest.embedder import compute_embeddings, persist_embeddings
        from prism_rag.ingest.similarity_linker import link_similar_nodes

        _backend_label = (
            f"ollama/{settings.ollama_model}"
            if settings.embed_backend == "ollama"
            else f"gemini/{settings.gemini_embed_model}"
        )
        typer.secho(
            f"\n🧬 Pass 3a: Computing embeddings ({_backend_label}, dim={settings.embedding_dim})...",
            fg=typer.colors.BLUE,
        )
        cache_path = settings.data_dir / "embed_cache.jsonl"
        vectors = compute_embeddings(graph, settings, cache_path=cache_path)
        gc_embed_cache(cache_path, live_sha_set)
        typer.echo(f"   Embedded {len(vectors)} nodes")

        typer.secho("\n🔗 Pass 3b: Generating similarity edges...", fg=typer.colors.BLUE)
        n_new = link_similar_nodes(graph, vectors, settings)
        typer.echo(f"   New edges: {n_new} · Total edges: {graph.edge_count}")

        # Persist embeddings to LanceDB for serve-time bridge computation
        n_persisted = persist_embeddings(vectors, settings.embedding_cache_path, dim=settings.embedding_dim)
        if n_persisted:
            typer.echo(f"   Persisted {n_persisted} embeddings to LanceDB")

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

    # ── Hint: link-symbols if a code graph exists alongside ──
    from prism_rag.config import PrismRagSettings as _S
    _cfg = _S()
    _code_candidates = list(_cfg.data_dir.glob("*/graph.json")) + list(_cfg.data_dir.glob("code/graph.json"))
    _code_graphs = [p for p in _code_candidates if "nimbus" not in str(p) and p != settings.graph_path]
    if _code_graphs:
        typer.secho(
            "   💡 Code graph detected. Run to build cross-namespace links:\n"
            f"   prism-rag link-symbols --vault-graph {settings.graph_path} "
            f"--code-graph {_code_graphs[0]}",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.secho(
            "   💡 To link vault notes to code symbols, run after ingest-code:\n"
            "   prism-rag link-symbols --vault-graph <vault/graph.json> --code-graph <code/graph.json>",
            fg=typer.colors.YELLOW,
        )



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
    scope: str = typer.Option("", "--scope", "-s", help="Namespace to search (empty = all)"),
    budget: int = typer.Option(4000, "--budget", "-b", help="Token budget"),
    mode: str = typer.Option("bfs", "--mode", "-m", help="Traversal mode: bfs or dfs"),
    show_content: bool = typer.Option(False, "--content", "-c", help="Show node content"),
) -> None:
    """Query the knowledge graph via BFS/DFS traversal."""
    from prism_rag.retrieve.bfs import federated_bfs
    from prism_rag.retrieve.dfs import federated_dfs
    from prism_rag.retrieve.entry import resolve_entry_points
    from prism_rag.store.federated import FederatedGraph

    settings = PrismRagSettings()

    # Use federated layer (transparent in single-graph mode)
    resolved = settings.resolved_graphs

    # If --graph is specified, override with a single-graph source
    if graph_path is not None:
        if not graph_path.exists():
            typer.secho(f"❌ Graph not found: {graph_path}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        from prism_rag.config import GraphSource
        resolved = [GraphSource(namespace="default", vault_path=settings.vault_path, data_dir=graph_path.parent)]

    fg = FederatedGraph.load(resolved)

    if fg.node_count == 0:
        typer.secho("❌ No graphs loaded. Run 'prism-rag ingest' first.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Resolve entry points (federated)
    entries = resolve_entry_points(fg, q, scope=scope if scope else None)
    if not entries:
        typer.secho(f"❌ No matching node for query: {q!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Use the first (best) entry point
    entry_ns, entry_id = entries[0]
    entry_graph = fg.get_graph(entry_ns)
    entry_label = entry_graph.g.nodes[entry_id].get("label", entry_id) if entry_graph else entry_id
    typer.secho(f"🎯 Entry: {entry_label} ({entry_ns}::{entry_id})", fg=typer.colors.CYAN)

    # Traverse within the resolved namespace
    if mode == "dfs":
        nodes = federated_dfs(fg, entry_ns, entry_id, budget=budget)
    else:
        nodes = federated_bfs(fg, entry_ns, entry_id, budget=budget)

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
    port: int = typer.Option(8102, "--port", "-p", help="Port for SSE transport (ignored for stdio)"),
    vault_path: Path = typer.Option(None, "--vault", "-v", help="Override PRISM_VAULT_PATH for this server"),
    data_dir: Path = typer.Option(None, "--data-dir", "-d", help="Override PRISM_DATA_DIR for this server"),
) -> None:
    """Start the PrismRag MCP Server.

    Loads graph.json from disk (instant). If graph.json doesn't exist,
    run 'prism-rag ingest' first.

    Exposes tools: search_knowledge, explain_node, trace_path,
    list_communities, explore_community, and vault write tools.
    """
    import os

    from prism_rag.mcp_server.server import run_server
    from prism_rag.store.federated import FederatedGraph

    # Forward CLI overrides into env so PrismRagSettings picks them up;
    # this covers the case where an MCP manager launches us as a subprocess.
    if vault_path is not None:
        os.environ["PRISM_VAULT_PATH"] = str(vault_path.expanduser().resolve())
    if data_dir is not None:
        os.environ["PRISM_DATA_DIR"] = str(data_dir.expanduser().resolve())

    settings = PrismRagSettings()
    resolved = settings.resolved_graphs

    # Verify at least one graph exists before starting
    fg = FederatedGraph.load(resolved)
    if fg.node_count == 0:
        typer.secho(
            "❌ No graphs loaded. Run 'prism-rag ingest' first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    typer.secho(
        f"📊 Loaded {len(fg.namespaces)} graph(s): {fg.node_count} nodes · {fg.edge_count} edges",
        fg=typer.colors.GREEN,
        err=True,
    )
    typer.secho(f"🚀 Starting MCP Server (transport={transport})...", fg=typer.colors.GREEN, err=True)
    run_server(transport=transport, port=port)


@app.command(name="ingest-code")
def ingest_code(
    repo: Path = typer.Option(..., "--repo", "-r", help="Path to source code repository"),
    data_dir: Path = typer.Option(None, "--data-dir", "-d", help="Output directory (default: PRISM_DATA_DIR/code)"),
    namespace: str = typer.Option("code", "--namespace", "-n", help="Graph namespace"),
    skip_cluster: bool = typer.Option(False, "--skip-cluster", help="Skip Leiden clustering"),
    skip_embed: bool = typer.Option(False, "--skip-embed", help="Skip embedding computation"),
) -> None:
    """Parse a source code repository and build a code:: knowledge graph.

    Pipeline: Tree-sitter AST → ParseResult → KnowledgeGraph → Leiden → Embed → graph.json

    Example:
        prism-rag ingest-code --repo /home/kingy/Foundation/ZenithLoom
    """
    from prism_rag.ingest.code_parser import CodeParser
    from prism_rag.ingest.embedder import compute_embeddings, persist_embeddings
    from prism_rag.store.networkx_backend import NetworkXBackend

    repo = repo.expanduser().resolve()
    if not repo.exists():
        typer.secho(f"❌ Repo not found: {repo}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    settings = PrismRagSettings()
    out_dir = (data_dir or settings.data_dir / namespace).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "graph.json"

    typer.secho(f"📂 Repo:   {repo}", fg=typer.colors.CYAN)
    typer.secho(f"📁 Output: {out_dir}", fg=typer.colors.CYAN)
    typer.secho(f"🏷  Namespace: {namespace}", fg=typer.colors.CYAN)
    typer.echo("")

    # ── Parse ──
    typer.secho("🔍 Parsing Python files with Tree-sitter...", fg=typer.colors.BLUE)
    parser = CodeParser()
    result = parser.parse(repo)
    typer.echo(f"   Nodes: {len(result.nodes)} · Edges: {len(result.edges)}")

    # ── Write to KnowledgeGraph ──
    typer.secho("\n🧩 Building KnowledgeGraph...", fg=typer.colors.BLUE)
    graph = KnowledgeGraph()
    backend = NetworkXBackend(graph)
    backend.write_result(result)
    typer.echo(f"   Nodes: {graph.node_count} · Edges: {graph.edge_count}")

    # ── Leiden clustering ──
    if skip_cluster:
        typer.secho("\n⏭  Leiden clustering (skipped)", fg=typer.colors.YELLOW)
    else:
        typer.secho("\n🧠 Leiden community detection...", fg=typer.colors.BLUE)
        n_communities = run_leiden(
            graph,
            resolution=settings.leiden_resolution,
            seed=settings.leiden_seed,
            god_nodes_per_community=settings.god_nodes_per_community,
        )
        typer.echo(f"   Communities: {n_communities}")

    # ── Embedding ──
    if skip_embed:
        typer.secho("\n⏭  Embedding (skipped)", fg=typer.colors.YELLOW)
    else:
        typer.secho(
            f"\n🔢 Computing embeddings ({settings.embed_backend}/{settings.ollama_model if settings.embed_backend == 'ollama' else settings.gemini_embed_model}, dim={settings.embedding_dim})...",
            fg=typer.colors.BLUE,
        )
        try:
            vectors = compute_embeddings(graph, settings, cache_path=out_dir / "embed_cache.jsonl")
            lance_path = out_dir / "lance"
            n_persisted = persist_embeddings(vectors, lance_path, dim=settings.embedding_dim)
            typer.echo(f"   Embeddings: {n_persisted} → {lance_path}")
        except Exception as exc:
            typer.secho(f"   ⚠ Embedding failed (continuing): {exc}", fg=typer.colors.YELLOW, err=True)

    # ── Persist ──
    typer.secho("\n💾 Persisting graph...", fg=typer.colors.BLUE)
    graph.save(graph_path)
    typer.echo(f"   → {graph_path}")

    typer.secho("\n✅ ingest-code complete.", fg=typer.colors.GREEN)
    typer.echo(
        f"   To serve: set PRISM_GRAPHS env var to include {out_dir}, then 'prism-rag serve'"
    )

    # ── Stale-ref detection: mark vault docs that reference changed code nodes ──
    from prism_rag.config import PrismRagSettings as _S2
    _vcfg = _S2()
    _vault_graph_path = _vcfg.graph_path
    if _vault_graph_path.exists():
        try:
            from datetime import date as _date
            from prism_rag.ingest.symbol_linker import mark_stale_refs
            from prism_rag.store.graph import KnowledgeGraph as _KG

            _vault_g = _KG.load(_vault_graph_path)
            # Determine changed nodes: compare new graph hashes against vault graph
            # (vault graph has mentions_symbol edges pointing to code node IDs)
            _mentioned_ids = {
                t for _, t, d in _vault_g.g.edges(data=True)
                if d.get("relation") == "mentions_symbol"
            }
            if _mentioned_ids:
                _new_hashes = {
                    nid: data.get("content_hash", "")
                    for nid, data in graph.g.nodes(data=True)
                    if nid in _mentioned_ids
                }
                # All mentioned code nodes present in new graph are "changed" candidates;
                # vault graph has no prior hashes to compare against, so mark all
                # whose content_hash differs from any previously stored stale_refs date.
                # Conservative: only mark nodes that actually exist in new graph.
                _changed = {nid for nid in _mentioned_ids if nid in graph.g}
                if _changed:
                    _n = mark_stale_refs(_vault_g, _changed, str(_date.today()))
                    if _n:
                        _vault_g.save(_vault_graph_path)
                        typer.secho(
                            f"   ⚠  Marked {_n} vault docs as stale "
                            f"(stale_refs in graph nodes — run 'prism-rag link-symbols' to refresh links)",
                            fg=typer.colors.YELLOW,
                        )
        except Exception as _e:
            logger.debug(f"[ingest-code] stale-ref detection skipped: {_e}")


@app.command(name="link-symbols")
def link_symbols_cmd(
    vault_graph: Path = typer.Option(..., "--vault-graph", "-v", help="Path to vault graph.json"),
    code_graph: Path = typer.Option(..., "--code-graph", "-c", help="Path to code graph.json"),
) -> None:
    """Build mentions_symbol cross-namespace edges from vault notes to code symbols.

    Scans vault note content for code symbol names (wikilinks and word-boundary
    matches), then writes mentions_symbol edges into the vault graph.json.

    Run after both 'prism-rag ingest' and 'prism-rag ingest-code' complete.
    Safe to re-run (idempotent — removes stale edges before rebuilding).

    Example:
        prism-rag link-symbols \\
          --vault-graph /path/nimbus/graph.json \\
          --code-graph  /path/zenith_code/graph.json
    """
    from prism_rag.ingest.symbol_linker import run_link_symbols

    vault_graph = vault_graph.expanduser().resolve()
    code_graph = code_graph.expanduser().resolve()

    if not vault_graph.exists():
        typer.secho(f"❌ Vault graph not found: {vault_graph}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if not code_graph.exists():
        typer.secho(f"❌ Code graph not found: {code_graph}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.secho(f"📄 Vault graph: {vault_graph}", fg=typer.colors.CYAN)
    typer.secho(f"💻 Code graph:  {code_graph}", fg=typer.colors.CYAN)
    typer.echo("")
    typer.secho("🔗 Scanning for symbol mentions...", fg=typer.colors.BLUE)

    n_ext, n_inf, n_amb = run_link_symbols(vault_graph, code_graph)

    total = n_ext + n_inf + n_amb
    typer.secho(f"\n✅ link-symbols complete.", fg=typer.colors.GREEN)
    typer.echo(
        f"   Edges written: {total} total  "
        f"(EXTRACTED={n_ext}, INFERRED={n_inf}, AMBIGUOUS={n_amb})"
    )
    typer.echo(f"   → {vault_graph}")


@app.command()
def canvas(
    graph: Path = typer.Option(..., "--graph", "-g", help="Path to graph.json"),
    output: Path = typer.Option(..., "--out", "-o", help="Output .canvas file path"),
    prefix: str = typer.Option("framework/", "--prefix", "-p",
                               help="Source-file prefix to include (e.g. 'framework/')"),
    cols: int = typer.Option(6, "--cols", help="File columns per row in the grid layout"),
) -> None:
    """Export a code graph subgraph to an Obsidian Canvas file.

    Groups nodes by source file, stacks them in columns, draws call/inherits/imports edges.

    Example:
        prism-rag canvas -g data/code/graph.json -o ~/Foundation/NimbusVault/ZenithLoom.canvas
    """
    from prism_rag.report.canvas_export import generate_canvas

    if not graph.exists():
        typer.secho(f"graph.json not found: {graph}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho(f"Loading graph from {graph}…", fg=typer.colors.BLUE)
    kg = KnowledgeGraph.load(graph)
    typer.secho(f"  {kg.node_count} nodes, {kg.edge_count} edges", fg=typer.colors.GREEN)

    typer.secho(f"Generating canvas (prefix={prefix!r}, cols={cols})…", fg=typer.colors.BLUE)
    n_nodes, n_edges = generate_canvas(kg, output, filter_prefix=prefix, files_per_row=cols)

    typer.secho(
        f"  Wrote {n_nodes} cards, {n_edges} edges → {output}",
        fg=typer.colors.GREEN,
    )


@app.command(name="migrate-lifecycle")
def migrate_lifecycle(
    graph_path: Path = typer.Option(..., "--graph-path", help="Path to graph.json to migrate"),
) -> None:
    """One-shot migration: assign lifecycle_class to historical mentions_symbol edges.

    Hard-fail mode — no fallback guessing. If any mentions_symbol edge has an
    unknown source_pass (not in {ast, code, conv}), the entire migration rolls
    back and the user must manually classify the offending edges.

    Rules:
      - relation == "mentions_symbol" + source_pass in ("ast", "code")  → DETERMINISTIC
      - relation == "mentions_symbol" + source_pass == "conv"           → ANCHORED
      - other mentions_symbol edges                                     → BLOCK (rollback)
    Non-mentions_symbol edges are left alone (their default is PROBABILISTIC).
    """
    import json as _json
    from prism_rag.store.graph import LifecycleClass
    from prism_rag.utils.io import atomic_write

    if not graph_path.exists():
        typer.echo(f"graph file not found: {graph_path}", err=True)
        raise typer.Exit(code=1)

    payload = _json.loads(graph_path.read_text())
    migrated = 0
    skipped = 0
    blocked: list[dict] = []
    for edge in payload.get("edges", []):
        if edge.get("relation") != "mentions_symbol":
            continue
        if edge.get("lifecycle_class"):
            skipped += 1
            continue
        sp = edge.get("source_pass", "")
        if sp in ("ast", "code"):
            edge["lifecycle_class"] = LifecycleClass.DETERMINISTIC
            migrated += 1
        elif sp == "conv":
            edge["lifecycle_class"] = LifecycleClass.ANCHORED
            migrated += 1
        else:
            blocked.append({
                "source": edge.get("source"),
                "target": edge.get("target"),
                "source_pass": sp,
            })

    if blocked:
        typer.echo(
            f"BLOCKED: {len(blocked)} mentions_symbol edges have unknown source_pass "
            f"and cannot be auto-classified. Migration rolled back.",
            err=True,
        )
        for b in blocked[:10]:
            typer.echo(f"  - {b['source']} → {b['target']} (source_pass={b['source_pass']!r})", err=True)
        raise typer.Exit(code=2)

    atomic_write(graph_path, _json.dumps(payload, ensure_ascii=False, indent=2))
    typer.echo(f"migrate-lifecycle complete: {migrated} migrated, {skipped} already aligned, 0 blocked")


@app.command(name="classify-edges")
def classify_edges_cmd() -> None:
    """Run EdgeClassifier on all probe data; write Tier 1 edges + Tier 2 inbox entries."""
    import logging
    from prism_rag.config import PrismRagSettings, get_classifier_profile
    from prism_rag.inbox.store import InboxStore
    from prism_rag.ingest.edge_classifier import classify_and_route
    from prism_rag.store.cross_namespace_probe import CrossNamespaceProbe
    from prism_rag.store.federated import FederatedGraph

    logging.basicConfig(level=logging.INFO)
    settings = PrismRagSettings()
    fg = FederatedGraph.load(settings.resolved_graphs)
    nimbus_src = next((s for s in settings.resolved_graphs if s.namespace == "nimbus"), None)
    if nimbus_src is None:
        typer.echo("no 'nimbus' namespace configured; cannot run classifier", err=True)
        raise typer.Exit(code=1)
    log_path = settings.data_dir / "cross_namespace_log.jsonl"
    model_id = getattr(settings, "embedding_model", "default")
    probe = CrossNamespaceProbe(log_path=log_path, model_id=model_id)
    inbox = InboxStore(settings.data_dir / "inbox.jsonl")
    profile = get_classifier_profile(settings, model_id)
    report = classify_and_route(fg, probe, inbox, nimbus_src, profile)
    inbox.save_atomic()
    nimbus_graph = fg.get_graph("nimbus")
    if nimbus_graph is None:
        typer.echo("error: 'nimbus' graph failed to load from disk", err=True)
        raise typer.Exit(code=1)
    nimbus_graph.save(nimbus_src.graph_path)
    typer.echo(f"classify-edges: promoted={report.promoted} queued={report.queued} "
               f"rolled_back={report.rolled_back} discarded={report.discarded}")


@app.command()
def version() -> None:
    """Print PrismRag version."""
    typer.echo(f"PrismRag v{__version__}")


@app.command(name="embed-status")
def embed_status() -> None:
    """Show embedding progress per namespace (cached vs pending)."""
    settings = PrismRagSettings()
    model = settings.ollama_model
    host = settings.ollama_host
    device = detect_model_device(model, host)

    typer.echo(f"model={model}  device={device}")
    typer.echo("")

    for gs in settings.resolved_graphs:
        ns = gs.namespace or "default"
        data_dir = gs.data_dir
        graph_path = data_dir / "graph.json"
        cache_path = data_dir / "embed_cache.jsonl"

        if not graph_path.exists():
            typer.echo(f"  namespace={ns}  [no graph.json]")
            continue

        kg = KnowledgeGraph.load(graph_path)
        # Embeddable = nodes with content (kind doesn't matter for status purposes)
        embeddable_nodes = [
            (nid, data)
            for nid, data in kg.g.nodes(data=True)
            if data.get("content", "").strip()
        ]
        total = len(embeddable_nodes)

        cache = _load_embed_cache(cache_path) if cache_path.exists() else {}
        embedded = sum(
            1 for nid, data in embeddable_nodes
            if nid in cache and cache[nid][0] == data.get("content_hash", "")
        )
        pending = total - embedded

        warning = "  [WARNING: GPU not active]" if device == "cpu" else ""
        typer.echo(
            f"  namespace={ns:<12} nodes={total:<6} "
            f"embedded={embedded:<6} pending={pending:<6} "
            f"model={model}{warning}"
        )


@app.command()
def calibrate() -> None:
    """Calibrate dedup threshold by sampling node pairs (interactive annotation).

    NOT YET IMPLEMENTED. Falls back gracefully to the threshold set in config
    (default 0.90, override via PRISM_DEDUP='{"threshold": 0.85}').
    """
    typer.echo("prism calibrate: not yet implemented. Using fallback threshold from config.")
    settings = PrismRagSettings()
    typer.echo(
        f"  current threshold: {settings.dedup.threshold}  "
        f"min_nodes_for_calibration: {settings.dedup.min_nodes_for_calibration}"
    )
    raise typer.Exit(0)


@app.command(name="ingest-project")
def ingest_project(
    repo: Path = typer.Option(..., "--repo", "-r", help="Path to project root (code + docs)"),
    data_dir: Path = typer.Option(None, "--data-dir", "-d", help="Output directory (default: PRISM_DATA_DIR/<namespace>)"),
    namespace: str = typer.Option("", "--namespace", "-n", help="Graph namespace (default: repo directory name)"),
    vault_name: str = typer.Option("", "--vault-name", help="Obsidian vault name for deep-link URIs in visualization"),
    skip_cluster: bool = typer.Option(False, "--skip-cluster", help="Skip Leiden clustering"),
    skip_embed: bool = typer.Option(False, "--skip-embed", help="Skip embedding computation"),
) -> None:
    """Ingest a project's code AND docs into one unified knowledge graph + HTML visualization.

    Pipeline:
      Pass 1a — Tree-sitter AST (Python files)
      Pass 1b — Markdown vault loader + wikilink/tag extraction
      Pass 2  — Leiden community detection
      Pass 3a — Embeddings
      Pass 3b — Similarity edges (doc↔doc, code↔code, doc↔code)
      Pass 3c — Symbol links (mentions_symbol: doc → code)
      Pass 4  — Persist graph.json + graph.html

    Example:
        prism-rag ingest-project --repo /home/kingy/Projects/Pulsify --namespace pulsify
    """
    from prism_rag.ingest.ast_extractor import extract_ast
    from prism_rag.ingest.code_parser import CodeParser
    from prism_rag.ingest.embedder import compute_embeddings, persist_embeddings
    from prism_rag.ingest.similarity_linker import link_similar_nodes
    from prism_rag.ingest.symbol_linker import link_symbols
    from prism_rag.ingest.vault_loader import load_vault
    from prism_rag.store.networkx_backend import NetworkXBackend

    repo = repo.expanduser().resolve()
    if not repo.exists():
        typer.secho(f"❌ Repo not found: {repo}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    ns = namespace or repo.name
    settings = PrismRagSettings()
    out_dir = (data_dir or settings.data_dir / ns).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.secho(f"📂 Repo:      {repo}", fg=typer.colors.CYAN)
    typer.secho(f"📁 Output:    {out_dir}", fg=typer.colors.CYAN)
    typer.secho(f"🏷  Namespace: {ns}", fg=typer.colors.CYAN)
    typer.echo("")

    graph = KnowledgeGraph()

    # ── Pass 1a: Python code (Tree-sitter) ──
    typer.secho("🔍 Pass 1a: Parsing Python files with Tree-sitter...", fg=typer.colors.BLUE)
    parser = CodeParser()
    result = parser.parse(repo)
    backend = NetworkXBackend(graph)
    backend.write_result(result)
    typer.echo(f"   Code nodes: {graph.node_count} · Code edges: {graph.edge_count}")

    # ── Pass 1b: Markdown docs ──
    typer.secho("\n📄 Pass 1b: Discovering markdown files...", fg=typer.colors.BLUE)
    docs, live_sha_set = load_vault(repo)
    if docs:
        typer.echo(f"   Found {len(docs)} markdown files")
        before_nodes = graph.node_count
        extract_ast(graph, docs)
        typer.echo(f"   Doc nodes added: {graph.node_count - before_nodes} · Total nodes: {graph.node_count}")
    else:
        typer.secho("   No markdown files found, skipping.", fg=typer.colors.YELLOW)

    # ── Pass 2: Leiden clustering ──
    if skip_cluster:
        typer.secho("\n⏭  Pass 2: Leiden clustering (skipped)", fg=typer.colors.YELLOW)
    else:
        typer.secho("\n🧠 Pass 2: Leiden community detection...", fg=typer.colors.BLUE)
        n_communities = run_leiden(
            graph,
            resolution=settings.leiden_resolution,
            seed=settings.leiden_seed,
            god_nodes_per_community=settings.god_nodes_per_community,
        )
        typer.echo(f"   Communities: {n_communities}")

    # ── Pass 3a: Embedding ──
    vectors: dict = {}
    if skip_embed:
        typer.secho("\n⏭  Pass 3a: Embedding (skipped)", fg=typer.colors.YELLOW)
    else:
        _backend_label = (
            f"ollama/{settings.ollama_model}"
            if settings.embed_backend == "ollama"
            else f"gemini/{settings.gemini_embed_model}"
        )
        typer.secho(f"\n🔢 Pass 3a: Computing embeddings ({_backend_label}, dim={settings.embedding_dim})...", fg=typer.colors.BLUE)
        try:
            vectors = compute_embeddings(graph, settings, cache_path=out_dir / "embed_cache.jsonl")
            lance_path = out_dir / "lance"
            n_persisted = persist_embeddings(vectors, lance_path, dim=settings.embedding_dim)
            typer.echo(f"   Embeddings: {n_persisted} → {lance_path}")
        except Exception as exc:
            typer.secho(f"   ⚠ Embedding failed (continuing): {exc}", fg=typer.colors.YELLOW, err=True)

    # ── Pass 3b: Similarity edges (doc↔code cross-type included) ──
    if vectors:
        typer.secho("\n🔗 Pass 3b: Generating similarity edges (doc↔code)...", fg=typer.colors.BLUE)
        n_sim = link_similar_nodes(graph, vectors, settings)
        typer.echo(f"   Similarity edges added: {n_sim} · Total edges: {graph.edge_count}")
    else:
        typer.secho("\n⏭  Pass 3b: Similarity edges (skipped — no vectors)", fg=typer.colors.YELLOW)

    # ── Pass 3c: Symbol links (mentions_symbol: doc → code) ──
    typer.secho("\n🔗 Pass 3c: Linking doc mentions → code symbols...", fg=typer.colors.BLUE)
    try:
        n_ext, n_inf, n_amb = link_symbols(graph, graph)
        total_sym = n_ext + n_inf + n_amb
        typer.echo(f"   Symbol edges: {total_sym} (EXTRACTED={n_ext}, INFERRED={n_inf}, AMBIGUOUS={n_amb})")
    except Exception as exc:
        typer.secho(f"   ⚠ Symbol linking failed (continuing): {exc}", fg=typer.colors.YELLOW, err=True)

    # ── Pass 4a: Persist graph.json ──
    typer.secho("\n💾 Pass 4a: Persisting graph...", fg=typer.colors.BLUE)
    graph_path = out_dir / "graph.json"
    graph.save(graph_path)
    typer.echo(f"   → {graph_path}")

    # ── Pass 4b: HTML visualization ──
    typer.secho("\n🎨 Pass 4b: Generating HTML visualization...", fg=typer.colors.BLUE)
    try:
        from prism_rag.report.visualize import generate_html
        html_path = out_dir / "graph.html"
        generate_html(graph, html_path, vault_name=vault_name or None)
        typer.echo(f"   → {html_path}")
    except ImportError:
        typer.secho("   ⏭  Visualization skipped (install pyvis: pip install prism-rag[viz])", fg=typer.colors.YELLOW)

    typer.secho("\n✅ ingest-project complete.", fg=typer.colors.GREEN)
    typer.echo(f"   To serve: add namespace={ns!r} data_dir={out_dir!r} to PRISM_GRAPHS, then 'prism-rag serve'")


@app.command()
def visualize(
    namespace: str = typer.Option("", "--namespace", "-n", help="Namespace to visualize (default: first namespace)"),
    output: Path = typer.Option(None, "--output", "-o", help="Output HTML path (default: data_dir/graph.html)"),
    vault: str = typer.Option("", "--vault", "-v", help="Obsidian vault name for deep-link URIs"),
    min_degree: int = typer.Option(0, "--min-degree", "-d", help="Only show nodes with degree >= N (0 = all). Use 10-20 for large graphs."),
) -> None:
    """Generate an interactive HTML knowledge graph visualization.

    Produces a graph.html in the data directory.
    Use --vault to enable Obsidian deep-link click-to-open on note nodes.

    Examples:
        prism visualize
        prism visualize --namespace nimbus --vault NimbusVault
    """
    from prism_rag.config import PrismRagSettings
    from prism_rag.store.graph import KnowledgeGraph

    settings = PrismRagSettings()
    resolved = settings.resolved_graphs

    if not resolved:
        typer.secho("No graphs configured.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    vault_name = vault or None

    # ── Single-namespace graph ────────────────────────────────────────
    try:
        from prism_rag.report.visualize import generate_html
    except ImportError as exc:
        typer.secho(
            f"Visualization unavailable (install pyvis: pip install prism-rag[viz]): {exc}",
            fg=typer.colors.YELLOW, err=True,
        )
        raise typer.Exit(1)

    # Select namespace
    if namespace:
        src = next((s for s in resolved if s.namespace == namespace), None)
        if src is None:
            typer.secho(f"Namespace {namespace!r} not found. Available: {[s.namespace for s in resolved]}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
    else:
        src = resolved[0]

    graph_path = src.data_dir / "graph.json"
    if not graph_path.exists():
        typer.secho(f"Graph not found at {graph_path}. Run 'prism ingest' first.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    graph = KnowledgeGraph.load(graph_path)
    out = output or (src.data_dir / "graph.html")
    generate_html(graph, out, vault_name=vault_name, min_degree=min_degree)
    typer.secho(f"✅ Graph visualization → {out}", fg=typer.colors.GREEN)


# ── Atomize sub-application ─────────────────────────────────────────────────

app.add_typer(atomize_app, name="atomize")

# ── Inbox sub-application ───────────────────────────────────────────────────

inbox_app = typer.Typer(help="Inbox management for cross-namespace edge review")
app.add_typer(inbox_app, name="inbox")


@inbox_app.callback(invoke_without_command=True)
def inbox_default(
    ctx: typer.Context,
    status: str = typer.Option("pending", "--status",
        help="Filter by status: pending|approved|rejected|auto_promoted|discarded|all"),
    top: int = typer.Option(20, "--top", help="Max entries to show"),
) -> None:
    """List inbox entries (default: pending)."""
    if ctx.invoked_subcommand is not None:
        return
    from prism_rag.config import PrismRagSettings
    from prism_rag.inbox.store import InboxStore
    settings = PrismRagSettings()
    inbox = InboxStore(settings.data_dir / "inbox.jsonl")
    if status == "all":
        rows = inbox.list_all(status=None, top_n=top)
    elif status == "pending":
        rows = inbox.list_pending(top_n=top)
    else:
        rows = inbox.list_all(status=status, top_n=top)
    if not rows:
        typer.echo(f"(no entries with status={status})")
        return
    typer.echo(f"{'[id]':<40} {'[source]':<40} {'[target]':<40} {'[conf]':<6} {'[status]'}")
    for e in rows:
        typer.echo(
            f"{e['id'][:38]:<40} {e['source'][:38]:<40} "
            f"{e['target'][:38]:<40} {e['confidence']:<6.2f} {e['status']}"
        )


@inbox_app.command(name="show")
def inbox_show(edge_id: str) -> None:
    """Show full details of one inbox entry."""
    from prism_rag.config import PrismRagSettings
    from prism_rag.inbox.store import InboxStore
    settings = PrismRagSettings()
    inbox = InboxStore(settings.data_dir / "inbox.jsonl")
    entry = inbox.get(edge_id)
    if entry is None:
        typer.echo(f"unknown edge_id: {edge_id}", err=True)
        raise typer.Exit(code=1)
    import json as _json
    typer.echo(_json.dumps(entry, ensure_ascii=False, indent=2))


def _apply_decision_for_cli(edge_id: str, decision: str, note: str) -> None:
    from prism_rag.config import PrismRagSettings
    from prism_rag.inbox.store import InboxStore, StatusTransitionError

    settings = PrismRagSettings()
    inbox = InboxStore(settings.data_dir / "inbox.jsonl")
    if inbox.get(edge_id) is None:
        typer.echo(f"unknown edge_id: {edge_id}", err=True)
        raise typer.Exit(code=1)

    # Locate nimbus source — may be absent in environments with no graphs configured.
    nimbus_src = next(
        (s for s in settings.resolved_graphs if s.namespace == "nimbus"), None
    )

    try:
        if decision == "approve" and nimbus_src is not None:
            # Full approval path: write ANCHORED edge into the vault graph.
            from prism_rag.inbox.approval import apply_decision
            from prism_rag.store.federated import FederatedGraph

            fg = FederatedGraph.load(settings.resolved_graphs)
            apply_decision(
                edge_id, decision, note,
                inbox=inbox, fg=fg, src=nimbus_src,
                decided_by="user_via_cli",
            )
            inbox.save_atomic()
            nimbus = fg.get_graph("nimbus")
            if nimbus is not None:
                nimbus.save(nimbus_src.graph_path)
        else:
            # Lightweight path: update inbox status only (no graph write).
            # Used when: decision == "reject" (no graph write needed), or when
            # no nimbus namespace is configured (approve without graph write).
            status = "approved" if decision == "approve" else "rejected"
            inbox.set_status(
                edge_id, status,
                decided_by="user_via_cli",
                decision_note=note,
            )
            inbox.save_atomic()
    except StatusTransitionError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)

    typer.echo(f"{decision}d: {edge_id}")


@inbox_app.command(name="approve")
def inbox_approve(
    edge_id: str,
    note: str = typer.Option("", "--note"),
) -> None:
    """Approve a pending edge."""
    _apply_decision_for_cli(edge_id, "approve", note)


@inbox_app.command(name="reject")
def inbox_reject(
    edge_id: str,
    note: str = typer.Option("", "--note"),
) -> None:
    """Reject a pending edge."""
    _apply_decision_for_cli(edge_id, "reject", note)


@inbox_app.command(name="approve-all")
def inbox_approve_all(
    min_conf: float = typer.Option(0.85, "--min-conf"),
    top_n: int = typer.Option(0, "--top-n", help="If > 0, only top-N by confidence"),
    yes: bool = typer.Option(False, "--yes", help="Skip interactive confirmation"),
) -> None:
    """Approve all pending edges above a confidence threshold."""
    from prism_rag.config import PrismRagSettings
    from prism_rag.inbox.store import InboxStore
    settings = PrismRagSettings()
    inbox = InboxStore(settings.data_dir / "inbox.jsonl")
    pending = [e for e in inbox.list_pending(top_n=999) if e["confidence"] >= min_conf]
    if top_n > 0:
        pending = pending[:top_n]
    if not pending:
        typer.echo("nothing to approve")
        return
    if not yes:
        typer.echo(f"about to approve {len(pending)} entries; rerun with --yes to proceed")
        return
    for e in pending:
        _apply_decision_for_cli(e["id"], "approve", f"approve-all min_conf={min_conf}")


@inbox_app.command(name="review")
def inbox_review() -> None:
    """Launch interactive TUI to review pending edges."""
    from prism_rag.config import PrismRagSettings
    from prism_rag.inbox.tui import InboxReviewApp
    settings = PrismRagSettings()
    app_tui = InboxReviewApp(settings.data_dir / "inbox.jsonl", settings)
    app_tui.run()


if __name__ == "__main__":
    app()
