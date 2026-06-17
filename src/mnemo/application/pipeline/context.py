"""The immutable state threaded through a pipeline — a typed slot map.

It knows nothing about any specific pipeline: it is a generic ``name -> value`` map,
read and written through ``Slot[T]`` keys (so a value's type is recovered statically).
Each ``set`` returns a new context (copy-on-write); stages never mutate in place. A
present slot — even an empty tuple — counts as produced; absent means a stage has not
run yet.
"""
from __future__ import annotations

from typing import TypeVar, cast

from mnemo.application.pipeline.errors import PipelineError
from mnemo.application.pipeline.slot import Slot

T = TypeVar("T")


class PipelineContext:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    @classmethod
    def empty(cls) -> "PipelineContext":
        return cls({})

    def get(self, slot: Slot[T]) -> T:
        if slot.name not in self._data:
            raise PipelineError(f"slot '{slot.name}' was not produced")
        return cast(T, self._data[slot.name])

    def has(self, slot: Slot[T]) -> bool:
        return slot.name in self._data

    def set(self, slot: Slot[T], value: T) -> "PipelineContext":
        return PipelineContext({**self._data, slot.name: value})

    @property
    def filled(self) -> frozenset[str]:
        return frozenset(self._data)
