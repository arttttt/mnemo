"""CLI controller: human-facing commands and operational tooling over the use cases."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from typing import Optional

import typer

from mnemo.domain.memory import MemoryType
from mnemo.infrastructure.container import build_container

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
    importance: float = typer.Option(
        0.5,
        "--importance",
        "-i",
        min=0.0,
        max=1.0,
        help="0.0-1.0 (0.9 critical, 0.5 normal, 0.3 minor). Defaults to 0.5.",
    ),
) -> None:
    """Store a memory. No LLM runs on write; prints {id, dedup, score}."""
    container = build_container()
    try:
        result = container.remember.execute(
            content=content,
            type=type,
            project=project,
            scope=scope,
            importance=importance,
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
        help="'project' (current project + global), 'global', or 'all' (cross-project).",
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Project slug to scope to (when --scope project)."
    ),
    type: Optional[str] = typer.Option(
        None, "--type", "-t", help=f"Restrict to one type: {_TYPES}."
    ),
    limit: int = typer.Option(
        10, "--limit", "-l", min=1, max=100, help="Maximum number of hits."
    ),
) -> None:
    """Search memories by meaning; prints ranked hits as JSON."""
    container = build_container()
    try:
        results = container.search.execute(
            query=query, scope=scope, project=project, type=type, limit=limit
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


def main() -> None:
    app()
