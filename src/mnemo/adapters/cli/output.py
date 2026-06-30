"""CLI presentation layer: choose between machine-readable JSON (``--json``, for agents
and scripts) and compact human-readable text (the default, for a terminal).

Each command computes its result once, then hands it here. Keeping the JSON
serialization and the human renderers together keeps the command functions in app.py
thin controllers and the two output modes provably consistent — one source of data,
two views. The ``format_*`` renderers are pure (data in, ``str`` out), so they unit-test
directly without a CLI runner.
"""
from __future__ import annotations

import json
from typing import Any

import typer

# A memory body collapses to this many characters in a list view before it is elided;
# the full text is one `get` (or `--json`) away, so a list stays scannable.
_SNIPPET_WIDTH = 100


def json_option() -> Any:
    """The shared ``--json`` flag.

    A factory (not a module-level constant) so each command gets its own typer
    ``OptionInfo`` rather than sharing one mutable instance. Default OFF: the
    terminal-friendly view is the default and an agent/script opts into the stable JSON
    contract with ``--json``.
    """
    return typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON instead of human-readable text (for agents/scripts).",
    )


def render(json_payload: Any, human_text: str, *, as_json: bool) -> None:
    """Echo one command's result: JSON when ``--json``, else the prepared human text.

    The JSON branch is the BARE payload (no envelope) at a single consistent format, so
    ``--json`` is the stable, parseable contract across every command.
    """
    if as_json:
        typer.echo(json.dumps(json_payload, indent=2, ensure_ascii=False))
    else:
        typer.echo(human_text)


def _one_line(text: str, width: int = _SNIPPET_WIDTH) -> str:
    """Collapse a memory body to one trimmed line for a list view, eliding the overflow."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: width - 1].rstrip() + "…"


def _date(created_at: str) -> str:
    """The date part (YYYY-MM-DD) of an ISO-8601 timestamp; the raw value if too short."""
    return created_at[:10] if len(created_at) >= 10 else created_at


def _handle(memory: Any) -> str:
    """The actionable handle to show in a list: the durable topic_key when present, else
    a short id prefix (a one-off memory's only handle; the full id is in --json)."""
    return memory.topic_key or f"id:{memory.id[:12]}"


def _count(singular: str, n: int, plural: str | None = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _hit_block(index: int, hit: Any) -> str:
    """One numbered entry for search/browse: a header line plus an indented snippet.

    Works on both SearchResult and BrowseResult — it reads only their shared fields
    (id, type, content, created_at, topic_key).
    """
    header = f"{index}. [{hit.type}] {_handle(hit)}  ·  {_date(hit.created_at)}"
    return f"{header}\n   {_one_line(hit.content)}"


def format_hits(hits: list[Any]) -> str:
    """Render a ranked search list or a browse list (newest-first); empty -> a clear note."""
    if not hits:
        return "No memories found."
    body = "\n\n".join(_hit_block(i, hit) for i, hit in enumerate(hits, start=1))
    return f"{_count('hit', len(hits))}\n\n{body}"


def format_remember(result: Any) -> str:
    """The outcome of a store: status verb + the id it applies to (the new head, the
    existing duplicate, or the freshly created row)."""
    return f"{result.status} — {result.id}"


def format_get(result: Any) -> str:
    """The full dereferenced record (untruncated content) plus its supersede chain."""
    lines = [
        f"[{result.type}] {result.topic_key or '(no topic_key)'}"
        f"  ·  {result.status}  ·  {_date(result.created_at)}",
        f"id: {result.id}",
    ]
    if result.project:
        lines.append(f"project: {result.project}")
    if result.related_files:
        lines.append(f"files: {', '.join(result.related_files)}")
    if result.supersedes:
        lines.append(f"supersedes: {result.supersedes}")
    lines += ["", result.content, "", f"chain ({len(result.chain)} of {result.chain_total}):"]
    for entry in result.chain:
        lines.append(f"  • {entry.status:<10} {_date(entry.created_at)}  {entry.id}")
    return "\n".join(lines)


def format_recall(payload: dict) -> str:
    """Render the recall bundle the CLI assembles: the header, the synthesized summary
    (or a note when the generator is off), then the light source references."""
    parts = [
        f"recall: {payload['project']}  ·  query \"{payload['query']}\"  "
        f"·  {_count('memory', payload['total'], 'memories')}",
        "",
        payload["summary"] or "(no generated summary — generator off; sources below)",
    ]
    sources = payload.get("sources") or []
    if sources:
        parts.append("")
        parts.append("sources:")
        parts += [f"  • [{source['type']}] {source['id']}" for source in sources]
    return "\n".join(parts)


def format_stats(stats: dict) -> str:
    """Totals plus a per-type breakdown (aligned), reading the stats payload."""
    lines = [f"memories: {stats['total']}  (pending: {stats['pending']})"]
    by_type = stats.get("by_type") or {}
    if by_type:
        width = max(len(name) for name in by_type)
        lines.append("by type:")
        lines += [f"  {name.ljust(width)}  {count}" for name, count in sorted(by_type.items())]
    return "\n".join(lines)


def format_reindex(payload: dict) -> str:
    """Render either the dry-run plan or the completed re-embed, from the reindex payload."""
    if payload.get("dry_run"):
        return (
            f"would re-embed {payload['memories']} memories at dim "
            f"{payload['target_dim']} (dry run; nothing changed)"
        )
    restarted = "yes" if payload["service_restarted"] else "no"
    return (
        f"re-embedded {payload['reindexed']} memories at dim {payload['dim']}; "
        f"service restarted: {restarted}"
    )


def format_deletion(result: Any, *, purged: bool = False) -> str:
    """Count of removed memories; the purge variant also notes the registry reset."""
    summary = f"{_count('memory', result.deleted, 'memories')} deleted"
    return f"{summary}; project registry reset" if purged else summary


def format_project_created(project: Any) -> str:
    suffix = f" — {project.description}" if project.description else ""
    return f"created project '{project.slug}'{suffix}"


def format_project_deleted(project: Any) -> str:
    return f"deleted project '{project.slug}' and all its memories"


def format_project_updated(project: Any) -> str:
    return f"updated project '{project.slug}': {project.description}"


def format_projects(projects: list[Any]) -> str:
    if not projects:
        return "No projects registered."
    width = max(len(project.slug) for project in projects)
    lines = [_count("project", len(projects)) + ":"]
    lines += [
        f"  {project.slug.ljust(width)}  {project.description or '(no description)'}"
        for project in projects
    ]
    return "\n".join(lines)
