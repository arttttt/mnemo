"""CLI controller: human-facing commands and operational tooling over the use cases."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from typing import Optional

import typer

from mnemo.domain.memory_type import MemoryType
from mnemo.infrastructure.composition import build_container

_TYPES = ", ".join(member.value for member in MemoryType)

app = typer.Typer(
    help="mnemo - local memory for AI coding agents. Store and search typed memories locally.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def store(
    content: str = typer.Argument(
        ..., help="The memory text (problem, solution, reasoning)."
    ),
    type: str = typer.Option(
        "working-notes", "--type", "-t", help=f"Memory type. One of: {_TYPES}."
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        "-p",
        help="Project slug (kebab-case). Omit and use --scope global for cross-project memory.",
    ),
    scope: str = typer.Option(
        "project",
        "--scope",
        "-s",
        help="'project' (belongs to one project) or 'global' (applies everywhere).",
    ),
    topic_key: Optional[str] = typer.Option(
        None, "--topic-key", "-k", help="Stable key to evolve one memory instead of duplicating."
    ),
    tags: Optional[list[str]] = typer.Option(
        None, "--tag", help="Tag for later filtering (repeatable)."
    ),
    related_files: Optional[list[str]] = typer.Option(
        None, "--file", help="File path this memory concerns (repeatable)."
    ),
) -> None:
    """Store a memory. No LLM runs on write; prints {id, dedup, superseded}."""
    container = build_container()
    try:
        result = container.remember.execute(
            content=content,
            type=type,
            project=project,
            scope=scope,
            topic_key=topic_key,
            tags=tags,
            related_files=related_files,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(json.dumps(asdict(result)))


@app.command()
def search(
    query: str = typer.Argument(..., help="What to look for (natural language)."),
    scope: str = typer.Option(
        "project",
        "--scope",
        "-s",
        help="'project' (the given project + global; requires --project), 'global', or 'all' (cross-project).",
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Project slug to scope to. Required with --scope project (the default), omitted for global/all."
    ),
    type: Optional[str] = typer.Option(
        None, "--type", "-t", help=f"Restrict to one type: {_TYPES}."
    ),
    tags: Optional[list[str]] = typer.Option(
        None, "--tag", help="Keep only memories carrying ALL of these tags (repeatable)."
    ),
    related_files: Optional[list[str]] = typer.Option(
        None, "--file", help="Keep only memories referencing ANY of these files (repeatable)."
    ),
    recency_days: Optional[int] = typer.Option(
        None, "--recency-days", min=1, help="Only memories created within the last N days."
    ),
    limit: int = typer.Option(
        10, "--limit", "-l", min=1, max=100, help="Maximum number of hits."
    ),
) -> None:
    """Search memories by meaning; prints ranked hits as JSON."""
    container = build_container()
    try:
        results = container.search.execute(
            query=query,
            scope=scope,
            project=project,
            type=type,
            tags=tags,
            related_files=related_files,
            recency_days=recency_days,
            limit=limit,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))


@app.command()
def stats() -> None:
    """Show how many memories are stored, broken down by type."""
    container = build_container()
    memories = container.repository.list_all()
    by_type = Counter(memory.type.value for memory in memories)
    typer.echo(json.dumps({"total": len(memories), "by_type": dict(by_type)}, indent=2))


@app.command()
def reindex(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be re-embedded; change nothing."
    ),
) -> None:
    """Re-embed every memory with the current embedder (run after switching embedders).

    Rebuilds the store at the new dimension if it changed; content, metadata and links
    are preserved. A no-op when the embedder/dimension is unchanged. Stops the shared
    service so it respawns on demand with the new dimension instead of the stale one.
    """
    from mnemo.adapters.mcp.service_control import stop_service
    from mnemo.application.use_cases.reindex_memories import ReindexMemories
    from mnemo.infrastructure.config import Config

    config = Config.from_env()
    container = build_container(config)
    target_dim = container.embedder.dim
    if dry_run:
        typer.echo(json.dumps(
            {"memories": len(container.repository.list_all()),
             "target_dim": target_dim, "dry_run": True}
        ))
        return
    # A running shared service holds the old store/dimension in memory: stop it before the
    # rebuild (so its open connection can't obstruct the table swap) and again after (in
    # case a live connector respawned it mid-run). It respawns on demand with the new state.
    stopped_before = stop_service(config)
    count = ReindexMemories(
        container.repository, container.embedder, container.scheduler
    ).execute()
    stopped_after = stop_service(config)
    typer.echo(json.dumps(
        {"reindexed": count, "dim": target_dim,
         "service_restarted": stopped_before or stopped_after}
    ))


@app.command()
def delete(
    ids: list[str] = typer.Argument(..., help="Ids of the memories to delete."),
) -> None:
    """Permanently delete specific memories."""
    result = build_container().delete.delete(ids)
    typer.echo(json.dumps(asdict(result)))


@app.command()
def clear(
    project: str = typer.Argument(..., help="Project whose memories to delete."),
) -> None:
    """Permanently delete all memories of one project."""
    result = build_container().delete.clear(project)
    typer.echo(json.dumps(asdict(result)))


@app.command()
def purge() -> None:
    """Permanently delete ALL memories."""
    result = build_container().delete.purge()
    typer.echo(json.dumps(asdict(result)))


@app.command()
def setup(
    client: Optional[str] = typer.Argument(
        None,
        help="Client to wire: claude-code, codex, kimi-code, cursor, windsurf, opencode. "
        "Omit to detect installed clients and pick from a list.",
    ),
    all_clients: bool = typer.Option(
        False, "--all", help="Wire every detected client without prompting."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done; write nothing."
    ),
) -> None:
    """Wire an MCP client to mnemo, or detect installed clients and offer to."""
    from mnemo.adapters.setup.client_registry import build_installers
    from mnemo.adapters.setup.selection import parse_selection

    installers = build_installers()
    by_name = {installer.name: installer for installer in installers}

    if client is not None:
        target = by_name.get(client)
        if target is None:
            raise typer.BadParameter(f"unknown client '{client}'. Known: {', '.join(by_name)}")
        _apply([target], dry_run)
        return

    detected = [installer for installer in installers if installer.detect()]
    if not detected:
        typer.echo(
            "No supported MCP clients detected. Wire one explicitly: "
            f"mnemo setup <{'|'.join(by_name)}>"
        )
        return

    typer.echo("Detected clients:")
    for number, installer in enumerate(detected, start=1):
        typer.echo(f"  {number}) {installer.name} — {installer.describe()}")
    if dry_run:
        return

    if all_clients:
        chosen = detected
    else:
        answer = typer.prompt("Select clients to wire (e.g. 1,3 or 'all')", default="all")
        chosen = [detected[index] for index in parse_selection(answer, len(detected))]
    _apply(chosen, dry_run=False)


def _apply(installers, dry_run: bool) -> None:
    if not installers:
        typer.echo("Nothing to do.")
        return
    for installer in installers:
        if dry_run:
            typer.echo(f"[dry-run] {installer.name}: {installer.describe()}")
            continue
        result = installer.install()
        mark = "✓" if result.status == "ok" else "✗"
        suffix = f"  ({result.message})" if result.message else ""
        typer.echo(f"{mark} {result.client}: {result.target}{suffix}")


def main() -> None:
    app()
