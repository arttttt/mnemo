"""Residency policy: how long a loaded model is kept in memory.

Chosen in code when a capability is built (not from the environment): ``Resident`` holds
the model until shutdown; ``Transient`` frees it as soon as nothing is using it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Resident:
    """Load lazily on first use and keep until shutdown."""


@dataclass(frozen=True)
class Transient:
    """Free the model as soon as nothing is using it."""


Residency = Resident | Transient
