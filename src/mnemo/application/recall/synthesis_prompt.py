"""Build the prompt that asks the generator for a faithful recall answer.

Keeps the generator on its one job — synthesis — and grounds it: use only the supplied
memories, never outside knowledge, and refuse (a fixed phrase) when none are relevant.
The refusal here is the prompt-level guard (a model/decoding/verifier gate is a later,
advanced addition); it measurably reduces confabulation on irrelevant retrievals without
over-refusing. Memories are laid out grouped by type so the model sees structure, and the
question is restated after them so attention re-anchors on it.
"""
from __future__ import annotations

from mnemo.application.recall.bundle import RecallBundle

REFUSAL = "No relevant memories found."


def build_synthesis_prompt(bundle: RecallBundle, query: str | None = None) -> str:
    lines = [
        f"Answer using ONLY the memories about project '{bundle.project}' below. Use only the "
        "memories — do not use outside knowledge and do not invent facts. Give a clear, "
        f'sufficiently complete answer. If none of the memories are relevant, reply exactly: "{REFUSAL}"',
        "",
    ]
    if query:
        lines += [f"Question: {query}", ""]
    lines.append("Memories:")
    for section in bundle.sections:
        lines.append(f"## {section.type}")
        lines.extend(f"- {memory.content}" for memory in section.memories)
        lines.append("")
    if query:
        lines.append(
            f'Now answer the question — "{query}" — using only the memories above. '
            f'If none are relevant, reply exactly: "{REFUSAL}"'
        )
    else:
        lines.append("Answer:")
    return "\n".join(lines)
