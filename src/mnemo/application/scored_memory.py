"""A memory paired with its similarity score."""
from dataclasses import dataclass

from mnemo.domain.memory import Memory


@dataclass(frozen=True)
class ScoredMemory:
    memory: Memory
    score: float
