"""Port: write mnemo's permission policy into one client's own config file.

Each client keeps its permission rules in its own format. An adapter writes POLICY in
that format, keeps the rest of the file as it was, and leaves the file untouched when
the rules are already there. When the file cannot take the rules safely, it writes
nothing and raises RulesNotWritten with the lines to add by hand.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mnemo.adapters.setup.json_config import load_json

ADDED = "permission rules added"
PRESENT = "permission rules already in place"


class RulesNotWritten(Exception):
    def __init__(self, reason: str, snippet: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.snippet = snippet


class PermissionRules(Protocol):
    @property
    def target(self) -> Path:
        """The config file the rules go into."""
        ...

    def describe(self) -> str:
        """A one-line description of the rules and where they go (for --dry-run)."""
        ...

    def apply(self) -> str:
        """Write the rules unless present; return ADDED or PRESENT, or a note on why
        nothing was needed. Raises RulesNotWritten."""
        ...


def describe_rules(rules: list[tuple[str, str]], target: Path) -> str:
    listed = ", ".join(f"{decision} {pattern}" for decision, pattern in rules)
    return f"add permission rules ({listed}) to {target}"


def load_json_object(path: Path, snippet: str) -> dict:
    """A JSON config file's top-level object ({} when absent); RulesNotWritten otherwise."""
    try:
        data = load_json(path)
    except ValueError as error:
        raise RulesNotWritten(f"{path} is not valid JSON ({error})", snippet) from error
    if not isinstance(data, dict):
        raise RulesNotWritten(f"{path} does not hold a JSON object", snippet)
    return data
