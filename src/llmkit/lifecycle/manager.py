"""A bounded POOL of independent runtime instances, governed by a residency policy.

Each concurrent caller LEASES its own instance (its own session) through ``use()``, runs
inference with NO lock held, and returns it — so N callers run truly in parallel and
safely, with no shared mutable runtime and no thread-safety assumption. At most ``size``
instances exist (a bounded semaphore caps concurrent leases); a caller waits when all are
busy.

A ``Transient`` instance is unloaded as soon as it is returned (``size=1`` reduces to the
old load-on-use / free-on-idle behaviour); a ``Resident`` instance is kept warm in the idle
set until ``close()``. Loading happens OUTSIDE the lock, so loading one instance never
blocks leasing another; a load failure frees the permit and abandons the slot (the caller
sees the error, the pool is not shrunk). ``unload`` always ends a slot unloaded.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable, Generic, TypeVar

from llmkit.lifecycle.loadable import Loadable
from llmkit.lifecycle.residency import Residency, Transient

_log = logging.getLogger("llmkit.lifecycle")

T = TypeVar("T", bound=Loadable)


class _Slot(Generic[T]):
    __slots__ = ("runtime", "loaded")

    def __init__(self, runtime: T) -> None:
        self.runtime = runtime
        self.loaded = False


class ResidencyManager(Generic[T]):
    def __init__(
        self, factory: Callable[[], T], residency: Residency, *, size: int = 1
    ) -> None:
        self._factory = factory
        self._residency = residency
        self._size = max(1, size)
        self._sem = threading.BoundedSemaphore(self._size)
        self._lock = threading.Lock()       # guards _idle + _closed only (never held over load/unload)
        self._idle: list[_Slot[T]] = []
        self._closed = False

    @contextmanager
    def use(self) -> Iterator[T]:
        slot = self._checkout()
        try:
            yield slot.runtime
        finally:
            self._checkin(slot)

    def close(self) -> None:
        """Unload every idle instance and refuse new leases; in-flight leases unload
        themselves on return. Idempotent."""
        with self._lock:
            self._closed = True
            idle, self._idle = self._idle, []
        for slot in idle:
            self._unload(slot)

    def _checkout(self) -> _Slot[T]:
        self._sem.acquire()  # blocks when all `size` instances are leased
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("residency pool is closed")
                slot = self._idle.pop() if self._idle else _Slot(self._factory())
            if not slot.loaded:  # a freshly minted slot; a warm idle slot skips load
                slot.runtime.load()
                slot.loaded = True
            return slot
        except BaseException:
            self._sem.release()  # never strand the permit if mint/load failed (pool not shrunk)
            raise

    def _checkin(self, slot: _Slot[T]) -> None:
        try:
            if isinstance(self._residency, Transient):
                self._unload(slot)  # free as soon as nothing is using it
                return
            with self._lock:
                closed = self._closed
                if not closed:
                    self._idle.append(slot)  # Resident: keep warm for the next lease
            if closed:
                self._unload(slot)  # raced close() → free it instead of re-idling (outside the lock)
        finally:
            self._sem.release()

    def _unload(self, slot: _Slot[T]) -> None:
        try:
            slot.runtime.unload()
        except Exception:  # noqa: BLE001 — a checkin/close must never fail on a bad unload
            _log.warning("residency unload failed; dropping the instance anyway", exc_info=True)
        finally:
            slot.loaded = False
