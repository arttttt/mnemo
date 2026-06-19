"""Port: text generation with a small instruct LLM.

Operation-level: one call completes a prompt; residency is hidden behind it.
"""
from __future__ import annotations

from typing import Protocol


class Generator(Protocol):
    def generate(self, prompt: str, *, max_tokens: int) -> str: ...
