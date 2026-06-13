"""Result of a deletion operation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DeletionResult:
    deleted: int
