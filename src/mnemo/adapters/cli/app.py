"""CLI controller: human-facing commands and operational tooling over the use cases."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from typing import Optional

import typer

from mnemo.infrastructure.container import build_container

app = typer.Typer(
    help="mnemo - local memory for AI coding agents",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def store(
    content: str,
    type: str = "working-notes",
    project: Optional[str] = None,
    scope: str = "project",
) -> None:
    """Store a memory."""
    container = build_container()
    result = container.remember.execute(
        content=content, type=type, project=project, scope=scope
    )
    typer.echo(json.dumps(asdict(result)))


@app.command()
def search(
    query: str,
    scope: str = "project",
    project: Optional[str] = None,
    type: Optional[str] = None,
    limit: int = 10,
) -> None:
    """Search memory (scope: project | global | all)."""
    container = build_container()
    results = container.search.execute(
        query=query, scope=scope, project=project, type=type, limit=limit
    )
    typer.echo(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))


@app.command()
def stats() -> None:
    """Show memory counts."""
    container = build_container()
    memories = container.repository.list_all()
    by_type = Counter(memory.type.value for memory in memories)
    typer.echo(json.dumps({"total": len(memories), "by_type": dict(by_type)}, indent=2))


def main() -> None:
    app()
