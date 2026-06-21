"""A runtime that can load and free its model on demand — what a ResidencyManager drives."""
from __future__ import annotations

from typing import Protocol


class Loadable(Protocol):
    def load(self) -> None:
        """Load the model into memory. Idempotent: a second call while loaded is a no-op."""
        ...

    def unload(self) -> None:
        """Free the model. Idempotent (safe when nothing is loaded) and ALWAYS ends
        unloaded: on return the runtime is in the not-loaded state and a later load()
        fully re-initialises — even if freeing the underlying resource raised. An
        implementation MUST null its handle in a finally around any fallible close()."""
        ...
