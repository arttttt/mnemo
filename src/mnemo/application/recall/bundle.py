"""The output of a recall run — a project's memory grouped by type for a digest.

Always carries the structured grouping; ``summary`` is the optional LLM-synthesized
prose, filled only when a generator stage ran (otherwise the grouping and order carry
the structure on their own).
"""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.application.pipeline.slot import Slot
from mnemo.domain.memory import Memory


@dataclass(frozen=True)
class RecallSection:
    type: str
    memories: tuple[Memory, ...]


@dataclass(frozen=True)
class RecallBundle:
    project: str
    sections: tuple[RecallSection, ...]
    summary: str | None = None

    @property
    def total(self) -> int:
        return sum(len(section.memories) for section in self.sections)


RECALL: Slot[RecallBundle] = Slot("recall")
