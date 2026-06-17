"""Port: a small instruct LLM — the only stage that needs text synthesis.

Used for generation only (a faithful, concise summary / merged record), never for
routing or classification. Heavy to load, so it runs inside a ``session()`` that loads
it on entry and frees it on exit (load → generate → unload) — the same on-demand
lifecycle the reranker uses, keeping it off resident RAM.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol


class LoadedGenerator(Protocol):
    def generate(self, prompt: str, *, max_tokens: int) -> str:
        """Complete `prompt` into text, bounded by `max_tokens` (deterministic)."""
        ...


class GeneratorPort(Protocol):
    def session(self) -> AbstractContextManager[LoadedGenerator]:
        """Load the model for the duration of the context, freeing it on exit."""
        ...
