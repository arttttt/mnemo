"""LlamaCppGenerator — passes the prompt and cap through, frees the model after the call."""
from __future__ import annotations

from llmkit.capabilities.llama_cpp_generator import LlamaCppGenerator
from llmkit.lifecycle.manager import ResidencyManager
from llmkit.lifecycle.residency import Transient


class _FakeGeneratorRuntime:
    def __init__(self, text: str) -> None:
        self._text = text
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        assert self.loaded  # the manager must have loaded it before the call
        return f"{self._text}|{prompt}|{max_tokens}"


def test_generator_passes_prompt_and_cap_then_frees_the_model():
    runtime = _FakeGeneratorRuntime("S")
    generator = LlamaCppGenerator(ResidencyManager(lambda: runtime, Transient()))

    out = generator.generate("hello", max_tokens=42)

    assert out == "S|hello|42"
    assert not runtime.loaded  # transient → freed after the call
