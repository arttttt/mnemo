"""LlamaCppRuntime.unload always ends unloaded — even if the model's close() raises."""
from __future__ import annotations

import logging

from llmkit.runtime.llama_cpp import GgufSource, LlamaCppRuntime


class _RaisingModel:
    def close(self) -> None:
        raise RuntimeError("native close blew up")


def test_unload_drops_the_handle_even_when_close_raises(caplog):
    runtime = LlamaCppRuntime(GgufSource(model="/nonexistent.gguf"))
    runtime._llama = _RaisingModel()

    with caplog.at_level(logging.WARNING, logger="llmkit.llama"):
        runtime.unload()  # must not raise

    assert runtime._llama is None  # always ends unloaded → a later load() re-initialises
    assert any("close() failed" in record.getMessage() for record in caplog.records)


def test_unload_is_idempotent_when_nothing_is_loaded():
    runtime = LlamaCppRuntime(GgufSource(model="/nonexistent.gguf"))
    runtime.unload()  # no-op, must not raise
    assert runtime._llama is None
