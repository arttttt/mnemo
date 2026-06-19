"""LlamaCppRuntime + LlamaCppGenerator against a real GGUF (needs llama-cpp, downloads weights)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.heavy


def test_generator_loads_a_real_gguf_and_completes():
    from llmkit.capabilities.llama_cpp_generator import LlamaCppGenerator
    from llmkit.lifecycle.manager import ResidencyManager
    from llmkit.lifecycle.residency import Transient
    from llmkit.runtime.llama_cpp import GgufSource, LlamaCppRuntime

    runtime = LlamaCppRuntime(GgufSource(model="Qwen/Qwen2.5-3B-Instruct-GGUF"))
    generator = LlamaCppGenerator(ResidencyManager(runtime, Transient()))

    text = generator.generate("Reply with a single word.", max_tokens=8)

    assert isinstance(text, str) and text.strip()
