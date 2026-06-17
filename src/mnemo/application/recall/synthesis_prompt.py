"""Build the prompt that asks the generator for a faithful, concise recall digest.

Keeps the generator on the one thing it is for — synthesis — and instructs it to use
only the supplied memories (faithfulness is a prompt-and-gate concern; the gate is a
later addition). The memories are laid out grouped by type so the model sees structure.
"""
from __future__ import annotations

from mnemo.application.recall.bundle import RecallBundle


def build_synthesis_prompt(bundle: RecallBundle, query: str | None = None) -> str:
    lines = [
        f"Summarize the current state of project '{bundle.project}' from the memories below.",
    ]
    if query:
        lines.append(f"Focus the summary on: {query}")
    lines.append("Use only what is given — do not invent facts. Be concise: a few sentences.")
    lines.append("")
    for section in bundle.sections:
        lines.append(f"## {section.type}")
        lines.extend(f"- {memory.content}" for memory in section.memories)
        lines.append("")
    lines.append("Summary:")
    return "\n".join(lines)
