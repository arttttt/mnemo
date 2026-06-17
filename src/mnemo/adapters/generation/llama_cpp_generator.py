"""Generator adapter: a small instruct LLM via llama.cpp (GGUF, Metal/CPU).

Loads the model inside ``session()`` and frees it on exit, logging load/generate timing
and peak RSS so the heavy stage's cost is visible. ``MNEMO_GENERATOR`` points at a local
GGUF file; llama-cpp-python is an optional dependency, imported lazily here so the rest
of mnemo runs without it.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from mnemo.adapters.runtime_stats import peak_rss_mb
from mnemo.application.ports.generator import LoadedGenerator

_log = logging.getLogger("mnemo.generator")


class LlamaCppGenerator:
    def __init__(self, model_path: str, *, context_tokens: int = 4096) -> None:
        self._model_path = model_path
        self._context_tokens = context_tokens

    @contextmanager
    def session(self) -> Iterator[LoadedGenerator]:
        from llama_cpp import Llama

        start = time.monotonic()
        llama = Llama(
            model_path=self._model_path,
            n_ctx=self._context_tokens,
            n_gpu_layers=-1,  # offload to Metal when present, else CPU
            verbose=False,
        )
        _log.info(
            "generator loaded model=%s load=%.2fs peak_rss=%.0fMB",
            self._model_path, time.monotonic() - start, peak_rss_mb(),
        )
        try:
            yield _LoadedLlamaCppGenerator(llama)
        finally:
            if hasattr(llama, "close"):
                llama.close()
            del llama
            _log.info("generator released model=%s peak_rss=%.0fMB", self._model_path, peak_rss_mb())


class _LoadedLlamaCppGenerator:
    def __init__(self, llama) -> None:
        self._llama = llama

    def generate(self, prompt: str, *, max_tokens: int) -> str:
        start = time.monotonic()
        completion = self._llama.create_completion(
            prompt, max_tokens=max_tokens, temperature=0.0
        )
        _log.info(
            "generated max_tokens=%d in %.2fs peak_rss=%.0fMB",
            max_tokens, time.monotonic() - start, peak_rss_mb(),
        )
        return completion["choices"][0]["text"]
