"""Shared application type aliases."""
from typing import Callable

from mnemo.domain.memory import Memory

Vector = list[float]
MemoryPredicate = Callable[[Memory], bool]
