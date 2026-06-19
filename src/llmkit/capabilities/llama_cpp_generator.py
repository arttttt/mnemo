"""Generator capability over the llama.cpp runtime — no head, the runtime emits text.

The model is loaded only for the call, through the residency manager.
"""
from __future__ import annotations

from llmkit.lifecycle.manager import ResidencyManager
from llmkit.runtime.llama_cpp import LlamaCppRuntime


class LlamaCppGenerator:
    def __init__(self, manager: ResidencyManager[LlamaCppRuntime]) -> None:
        self._manager = manager

    def generate(self, prompt: str, *, max_tokens: int) -> str:
        with self._manager.use() as runtime:
            return runtime.complete(prompt, max_tokens=max_tokens)
