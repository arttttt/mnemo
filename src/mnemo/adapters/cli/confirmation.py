"""Reusable interactive confirmation for destructive CLI commands.

One place defines how a command asks "are you sure?": print the question with a y/n
prompt, BLOCK until the user gives a clear answer (re-asking on anything else), and abort
the command (a clean non-zero exit, no traceback) unless they confirm. A command's `--yes`
flag is passed through as `assume_yes` to skip the prompt for non-interactive use.

Reuse `confirm_or_abort` for any command that irreversibly destroys data (e.g. `purge`),
so every such command asks the same way.
"""
from __future__ import annotations

import typer

_YES = {"y", "yes"}
_NO = {"n", "no"}


def confirm_or_abort(question: str, *, assume_yes: bool = False) -> None:
    """Ask `question` as a y/n prompt and wait for the answer: return on yes, raise
    ``typer.Abort`` on no. ``assume_yes=True`` (a command's ``--yes`` flag) confirms without
    asking. Any answer other than yes/no re-asks, so the command blocks until the choice is
    explicit (or stdin closes, which typer turns into an abort)."""
    if assume_yes:
        return
    while True:
        answer = typer.prompt(f"{question} [y/n]", default="", show_default=False).strip().lower()
        if answer in _YES:
            return
        if answer in _NO:
            raise typer.Abort()
        typer.echo("Please answer 'y' or 'n'.")
