"""Owns a runtime's load/unload by a residency policy.

A capability runs its work inside ``use()``: the runtime is loaded for the body and,
under a ``Transient`` policy, freed once nothing else is using it (ref-counted, so a
concurrent use never frees a model mid-call). A ``Resident`` policy loads lazily on the
first use and keeps it until shutdown.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Generic, TypeVar

from llmkit.lifecycle.loadable import Loadable
from llmkit.lifecycle.residency import Residency, Transient

T = TypeVar("T", bound=Loadable)


class ResidencyManager(Generic[T]):
    def __init__(self, runtime: T, residency: Residency) -> None:
        self._runtime = runtime
        self._residency = residency
        self._lock = threading.Lock()
        self._users = 0
        self._loaded = False

    @contextmanager
    def use(self) -> Iterator[T]:
        self._acquire()
        try:
            yield self._runtime
        finally:
            self._release()

    def _acquire(self) -> None:
        with self._lock:
            if not self._loaded:
                self._runtime.load()
                self._loaded = True
            self._users += 1

    def _release(self) -> None:
        with self._lock:
            self._users -= 1
            if self._users == 0 and isinstance(self._residency, Transient):
                self._runtime.unload()
                self._loaded = False
