"""The plan's operation vocabulary — what the executor may do to the store.

A cohesive, sealed group of three. ``Supersede`` covers both a merge and a
summarization (N originals collapse into one new record); ``Insert`` adds a brand-new
record (e.g. an extracted insight) and leaves the sources untouched; ``Flag`` records a
contradiction edge without changing any content. The executor applies them by id, so
re-running a plan is idempotent.
"""
from __future__ import annotations

from dataclasses import dataclass

from mnemo.application.pipeline.proposed_memory import ProposedMemory


@dataclass(frozen=True)
class Supersede:
    source_ids: tuple[str, ...]
    replacement: ProposedMemory


@dataclass(frozen=True)
class Insert:
    record: ProposedMemory


@dataclass(frozen=True)
class Flag:
    left_id: str
    right_id: str
    kind: str = "contradiction"


Operation = Supersede | Insert | Flag
