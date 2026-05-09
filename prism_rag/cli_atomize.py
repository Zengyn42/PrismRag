"""CLI group for offline atomize proposal management.

Commands:
  prism-rag atomize list    List all pending proposals
  prism-rag atomize show    Show details of a specific proposal
  prism-rag atomize apply   Apply a pending proposal offline
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(name="atomize", help="Manage atomize_document proposals", no_args_is_help=True)


def _pending_dir(data_dir: Path) -> Path:
    return data_dir / "atomize-proposals" / "pending"


def _applied_dir(data_dir: Path) -> Path:
    return data_dir / "atomize-proposals" / "applied"


@app.command(name="list")
def list_proposals(
    output: Path = typer.Option(None, "--output", "-o", help="Output dir (default: from settings)"),
) -> None:
    """List all pending atomize proposals."""
    from prism_rag.config import PrismRagSettings
    settings = PrismRagSettings()
    data_dir = output.expanduser().resolve() if output else settings.data_dir
    pending = _pending_dir(data_dir)

    if not pending.exists() or not any(pending.glob("*.json")):
        typer.echo("No pending proposals.")
        return

    for f in sorted(pending.glob("*.json")):
        proposal = json.loads(f.read_text())
        claim_count = len(proposal.get("claims", []))
        typer.echo(f"  {proposal['proposal_id'][:12]}…  doc={proposal.get('doc_path', '?')}  claims={claim_count}")


@app.command(name="show")
def show_proposal(
    proposal_id: str = typer.Argument(..., help="Proposal ID (full or prefix)"),
    output: Path = typer.Option(None, "--output", "-o"),
) -> None:
    """Show details of a pending proposal."""
    from prism_rag.config import PrismRagSettings
    settings = PrismRagSettings()
    data_dir = output.expanduser().resolve() if output else settings.data_dir
    pending = _pending_dir(data_dir)

    # Allow prefix match
    matches = [f for f in pending.glob("*.json") if f.stem.startswith(proposal_id)]
    if not matches:
        typer.secho(f"Proposal not found: {proposal_id!r}", fg=typer.colors.RED)
        raise typer.Exit(1)
    if len(matches) > 1:
        typer.secho(f"Ambiguous prefix {proposal_id!r} — {len(matches)} matches", fg=typer.colors.RED)
        raise typer.Exit(1)

    proposal = json.loads(matches[0].read_text())
    typer.echo(f"Proposal: {proposal['proposal_id']}")
    typer.echo(f"Doc:      {proposal.get('doc_path', '?')}")
    typer.echo(f"Doc SHA:  {proposal.get('doc_sha', '?')}")
    typer.echo(f"Claims:   {len(proposal.get('claims', []))}")
    for c in proposal.get("claims", []):
        status = c.get("claim_status", "?")
        typer.echo(f"  [{status:8}] {c.get('knowledge_id', '?')} — {c.get('title', '?')}")


@app.command(name="apply")
def apply_proposal(
    proposal_id: str = typer.Argument(..., help="Proposal ID to apply"),
    vault: Path = typer.Option(None, "--vault", "-v", help="Vault root"),
    output: Path = typer.Option(None, "--output", "-o", help="Data dir"),
) -> None:
    """Apply a pending atomize proposal offline (creates knowledge files, patches source doc)."""
    from prism_rag.config import PrismRagSettings
    from prism_rag.ingest.atomize import atomize_apply_impl, StaleDocError
    settings = PrismRagSettings()
    data_dir = output.expanduser().resolve() if output else settings.data_dir
    vault_root = vault.expanduser().resolve() if vault else settings.resolved_graphs[0].vault_path

    pending = _pending_dir(data_dir)
    applied = _applied_dir(data_dir)

    # Prefix match
    matches = [f for f in pending.glob("*.json") if f.stem.startswith(proposal_id)]
    if not matches:
        typer.secho(f"Proposal not found: {proposal_id!r}", fg=typer.colors.RED)
        raise typer.Exit(1)
    full_id = matches[0].stem

    try:
        result = atomize_apply_impl(
            proposal_id=full_id,
            vault_root=vault_root,
            pending_dir=pending,
            applied_dir=applied,
        )
        typer.secho(f"Applied: {result['applied_count']} claims", fg=typer.colors.GREEN)
        for kf in result.get("knowledge_files", []):
            typer.echo(f"  Created: {kf}")
    except StaleDocError as e:
        typer.secho(f"Stale document: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)
