"""The kind of a memory — shapes how it is stored and retrieved, and how long it may be.

A proper enum (NOT a bare ``str``): a ``MemoryType`` is its own type, never interchangeable with
a string — convert explicitly via ``.value`` (out) and ``MemoryType(...)`` (in). Each member also
carries its own ``max_tokens`` content cap; most are 512, but a ``rule`` is held far tighter — a
rule must be terse and prescriptive, not a write-up. The embedder window is a separate ceiling,
enforced alongside this (the effective limit is the stricter of the two).
"""
from enum import Enum


class MemoryType(Enum):
    max_tokens: int  # per-member content cap, in tokens (declared here for typing; set in __new__)

    DECISION = ("decision", 512)
    PROGRESS = ("progress", 512)
    RESEARCH = ("research", 512)
    RULE = ("rule", 128)
    LEARNING = ("learning", 512)
    WORKING_NOTES = ("working-notes", 512)

    def __new__(cls, value: str, max_tokens: int) -> "MemoryType":
        obj = object.__new__(cls)
        obj._value_ = value           # MemoryType(value) and .value keep working
        obj.max_tokens = max_tokens
        return obj
