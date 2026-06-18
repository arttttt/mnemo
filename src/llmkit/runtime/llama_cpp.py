"""llama.cpp runtime for a small instruct LLM (GGUF, Metal/CPU).

Loads a GGUF — a local path or a Hugging Face repo (downloaded by filename glob) — into a
mmap-backed, ``n_ctx``-bounded llama.cpp model on load, and frees it on unload. ``llama_cpp``
is imported lazily with an actionable error, so importing this module is cheap and the rest
of llmkit runs without it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GgufSource:
    model: str                      # a local GGUF path, or a HF repo id to download from
    filename: str = "*q4_k_m.gguf"  # glob within the repo (ignored for a local path)
    context_tokens: int = 4096


class LlamaCppRuntime:
    def __init__(self, source: GgufSource, *, cache_dir: str | None = None) -> None:
        self._source = source
        self._cache_dir = cache_dir
        self._llama: Any = None

    def load(self) -> None:
        if self._llama is not None:
            return  # idempotent
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "the generator needs llama-cpp-python — install it (pip install llama-cpp-python)"
            ) from exc

        src = self._source
        if os.path.exists(src.model):
            self._llama = Llama(
                model_path=src.model, n_ctx=src.context_tokens,
                n_gpu_layers=-1, verbose=False,  # offload to Metal when present, else CPU
            )
        else:
            self._llama = Llama.from_pretrained(
                repo_id=src.model, filename=src.filename, cache_dir=self._cache_dir,
                n_ctx=src.context_tokens, n_gpu_layers=-1, verbose=False,
            )

    def unload(self) -> None:
        if self._llama is not None and hasattr(self._llama, "close"):
            self._llama.close()
        self._llama = None

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        completion = self._llama.create_completion(prompt, max_tokens=max_tokens, temperature=0.0)
        return completion["choices"][0]["text"]
