"""CLI controller: human-facing commands and operational tooling over the use cases."""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version as package_version
from typing import Optional

import typer

from mnemo.adapters.cli.confirmation import confirm_or_abort
from mnemo.application.project_gate import UnknownProject
from mnemo.domain.constants import DEFAULT_RECALL_LIMIT
from mnemo.domain.memory_type import MemoryType
from mnemo.infrastructure.composition import build_container
from mnemo.infrastructure.container import Container
from mnemo.infrastructure.dimension_guard import verify_store_dimension
from mnemo.infrastructure.logging_config import configure_logging

_TYPES = ", ".join(member.value for member in MemoryType)
_PACKAGE_NAME = "mnemo"

app = typer.Typer(
    help="mnemo - local memory for AI coding agents. Store and search typed memories locally.",
    no_args_is_help=True,
    add_completion=False,
)


def _installed_version() -> str:
    return package_version(_PACKAGE_NAME)


def _echo_version() -> None:
    typer.echo(_installed_version())


def _guarded_container() -> Container:
    """Build the container, then fail fast (cleanly) if the configured embedder's dimension
    disagrees with the store's. NOT used by `reindex`, which must open a mismatched store to
    repair it via set_dimension."""
    container = build_container()
    try:
        verify_store_dimension(container.embedding_queue, container.embedder.dim)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    return container


@app.command("version")
def show_version() -> None:
    """Show the installed mnemo version."""
    _echo_version()


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
    """Store a memory. No LLM runs on write; prints {id, status}."""
    container = _guarded_container()
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
    created_after: Optional[str] = typer.Option(
        None, "--created-after", help="Keep only memories created at or after this ISO-8601 instant (e.g. 2026-06-01)."
    ),
    limit: int = typer.Option(
        10, "--limit", "-l", min=1, max=100, help="Maximum number of hits."
    ),
) -> None:
    """Search memories by meaning; prints ranked hits as JSON."""
    container = _guarded_container()
    try:
        results = container.search.execute(
            query=query,
            scope=scope,
            project=project,
            type=type,
            tags=tags,
            related_files=related_files,
            created_after=created_after,
            limit=limit,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))


@app.command()
def browse(
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
    created_after: Optional[str] = typer.Option(
        None, "--created-after", help="Keep only memories created at or after this ISO-8601 instant (e.g. 2026-06-01)."
    ),
    status: str = typer.Option(
        "active", "--status", help="Which versions: 'active' (default), 'superseded', or 'all'."
    ),
    limit: int = typer.Option(
        10, "--limit", "-l", min=1, max=100, help="Maximum number of memories."
    ),
) -> None:
    """List memories by filter, newest first (no query, no ranking); prints JSON."""
    container = build_container()
    try:
        results = container.browse.execute(
            scope=scope,
            project=project,
            type=type,
            tags=tags,
            related_files=related_files,
            created_after=created_after,
            status=status,
            limit=limit,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))


@app.command()
def get(
    id: Optional[str] = typer.Option(
        None, "--id", help="Exact memory id (global). Mutually exclusive with --topic-key."
    ),
    topic_key: Optional[str] = typer.Option(
        None, "--topic-key", "-k", help="Resolve by topic_key (active head + chain). Needs --project or --scope global."
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Project the topic_key lives in (required for --scope project)."
    ),
    scope: str = typer.Option(
        "project", "--scope", "-s", help="'project' (default) or 'global' — where the topic_key lives. Ignored with --id."
    ),
    chain_limit: int = typer.Option(
        10, "--chain-limit", min=1, max=100, help="Max chain versions (newest first)."
    ),
    chain_after: Optional[str] = typer.Option(
        None, "--chain-after", help="Chain cursor: id of the last (oldest) entry you saw."
    ),
) -> None:
    """Dereference one memory by id or topic_key — full record + supersede chain; prints JSON."""
    try:
        result = build_container().get.execute(
            id=id,
            topic_key=topic_key,
            project=project,
            scope=scope,
            chain_limit=chain_limit,
            chain_after=chain_after,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(json.dumps(asdict(result), indent=2, ensure_ascii=False))


@app.command()
def recall(
    project: str = typer.Argument(..., help="Project whose memory to recall."),
    query: str = typer.Argument(..., help="What to recall about — a question or topic."),
    limit: int = typer.Option(
        DEFAULT_RECALL_LIMIT, "--limit", "-l", min=1, max=200,
        help="Number of the most query-relevant memories to ground the answer on.",
    ),
) -> None:
    """Recall a project's memory as a query-focused answer (the CLI view of the `recall` MCP tool).

    MNEMO_GENERATOR synthesizes a grounded summary from the gathered memories (refusing when none
    are relevant); with the generator off it is the structured grouping (MNEMO_RERANKER can order
    it by the query). Model timing/RAM go to the logs (MNEMO_LOG_LEVEL); the bundle prints as JSON.
    """
    configure_logging()
    started = time.monotonic()
    try:
        bundle = _guarded_container().recall.execute(project=project, query=query, limit=limit)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    except RuntimeError as exc:
        # A configured model runtime failed to initialise. Surface its actionable
        # message without leaking a traceback through the CLI boundary.
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    logging.getLogger("mnemo.recall").info(
        "recall project=%s total=%d in %.2fs",
        bundle.project, bundle.total, time.monotonic() - started,
    )
    payload = {
        "project": bundle.project,
        "query": query,
        "total": bundle.total,
        "summary": bundle.summary,
        "sources": [
            {"id": memory.id, "type": section.type}
            for section in bundle.sections
            for memory in section.memories
        ],
    }
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command()
def stats() -> None:
    """Show how many memories are stored, by type, and how many await a vector.

    `pending` = memories with no embedding yet — lexically searchable but absent
    from dense/semantic search until the background worker embeds them (a
    permanently-failed encode stays pending too, and is retried on the next start).
    """
    container = build_container()
    memories = container.repository.list_all()
    by_type = Counter(memory.type.value for memory in memories)
    typer.echo(json.dumps(
        {
            "total": len(memories),
            "pending": container.embedding_queue.pending_count(),
            "by_type": dict(by_type),
        },
        indent=2,
    ))


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
        container.embedding_queue, container.embedder, container.scheduler
    ).execute()
    stopped_after = stop_service(config)
    typer.echo(json.dumps(
        {"reindexed": count, "dim": target_dim,
         "service_restarted": stopped_before or stopped_after}
    ))


@app.command()
def delete(
    ids: list[str] = typer.Argument(..., help="Ids of the memories to delete."),
    cascade: bool = typer.Option(
        False,
        "--cascade",
        help="Also delete every older memory each id supersedes, down to the chain root (the whole lineage).",
    ),
) -> None:
    """Permanently delete specific memories (with --cascade, their whole older lineage too)."""
    result = build_container().delete.delete(ids, cascade=cascade)
    typer.echo(json.dumps(asdict(result)))


@app.command()
def create_project(
    name: str = typer.Argument(..., help="Project slug (the id; reused on every memory)."),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="What this project is (optional)."
    ),
) -> None:
    """Register a new project. Writing to an unregistered project is rejected."""
    try:
        project = build_container().create_project.execute(name, description)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(json.dumps(asdict(project)))


@app.command()
def delete_project(
    name: str = typer.Argument(..., help="Project slug to delete, with all its memories."),
) -> None:
    """Permanently delete a project and all its memories (and their links)."""
    try:
        project = build_container().delete_project.execute(name)
    except UnknownProject as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(json.dumps(asdict(project)))


@app.command()
def update_project(
    name: str = typer.Argument(..., help="Project slug to update."),
    description: str = typer.Argument(..., help="New description for the project."),
) -> None:
    """Set or change a project's description."""
    try:
        project = build_container().update_project.execute(name, description)
    except UnknownProject as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(json.dumps(asdict(project)))


@app.command()
def list_projects() -> None:
    """List the registered projects (newest first)."""
    projects = build_container().list_projects.execute()
    typer.echo(json.dumps([asdict(p) for p in projects], indent=2, ensure_ascii=False))


@app.command()
def purge(
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt (for non-interactive use)."
    ),
) -> None:
    """Permanently delete ALL memories and the project registry. Prompts to confirm."""
    confirm_or_abort(
        "This permanently deletes ALL memories and the project registry. Continue?",
        assume_yes=yes,
    )
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
