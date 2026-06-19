"""A runtime that can load and free its model on demand — what a ResidencyManager drives."""
from __future__ import annotations

from typing import Protocol


class Loadable(Protocol):
    def load(self) -> None:
        """Load the model into memory. Idempotent: a second call while loaded is a no-op."""
        ...

    def unload(self) -> None:
        """Free the model. Idempotent: safe to call when nothing is loaded."""
        ...
