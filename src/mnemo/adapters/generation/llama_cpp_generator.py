"""Generator adapter: a small instruct LLM via llama.cpp (GGUF, Metal/CPU).

Loads the model inside ``session()`` and frees it on exit, logging load/generate timing
and peak RSS so the heavy stage's cost is visible. ``MNEMO_GENERATOR`` points at a local
GGUF file; llama-cpp-python is an optional dependency, imported lazily here so the rest
of mnemo runs without it.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from mnemo.adapters.runtime_stats import peak_rss_mb
from mnemo.application.ports.generator import LoadedGenerator

_log = logging.getLogger("mnemo.generator")


class LlamaCppGenerator:
    def __init__(
        self, model: str, *, filename: str = "*q4_k_m.gguf", context_tokens: int = 4096
    ) -> None:
        self._model = model
        self._filename = filename
        self._context_tokens = context_tokens

    @contextmanager
    def session(self) -> Iterator[LoadedGenerator]:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "the recall generator needs llama-cpp-python — install it "
                "(pip install llama-cpp-python) or set MNEMO_GENERATOR=off"
            ) from exc

        start = time.monotonic()
        # a local GGUF path loads directly; anything else is treated as a HF repo to fetch
        if os.path.exists(self._model):
            llama = Llama(
                model_path=self._model, n_ctx=self._context_tokens,
                n_gpu_layers=-1, verbose=False,  # offload to Metal when present, else CPU
            )
        else:
            llama = Llama.from_pretrained(
                repo_id=self._model, filename=self._filename,
                n_ctx=self._context_tokens, n_gpu_layers=-1, verbose=False,
            )
        _log.info(
            "generator loaded model=%s load=%.2fs peak_rss=%.0fMB",
            self._model, time.monotonic() - start, peak_rss_mb(),
        )
        try:
            yield _LoadedLlamaCppGenerator(llama)
        finally:
            if hasattr(llama, "close"):
                llama.close()
            del llama
            _log.info("generator released model=%s peak_rss=%.0fMB", self._model, peak_rss_mb())


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
