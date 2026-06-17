"""The output of a recall run — a project's memory grouped by type for a digest.

This is the model-free recall: a structured bundle, no LLM synthesis yet. A later
generation stage will compress it into prose; until then the grouping and order carry
the structure.
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

    @property
    def total(self) -> int:
        return sum(len(section.memories) for section in self.sections)


RECALL: Slot[RecallBundle] = Slot("recall")
