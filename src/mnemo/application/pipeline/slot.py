"""A typed key into the pipeline context.

A stage reads and writes context values through ``Slot[T]`` keys, not attributes, so
the context carries no per-pipeline fields. The slot's ``name`` identifies it in a
stage's ``requires``/``provides`` contract; ``T`` is phantom — it only recovers the
value type statically at ``get``/``set``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Slot(Generic[T]):
    name: str
