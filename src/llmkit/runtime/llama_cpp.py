"""llama.cpp runtime for a small instruct LLM (GGUF, Metal/CPU).

Loads a GGUF — a local path or a Hugging Face repo (downloaded by filename glob) — into a
mmap-backed, ``n_ctx``-bounded llama.cpp model on load, and frees it on unload. ``llama_cpp``
is imported lazily with an actionable error, so importing this module is cheap and the rest
of llmkit runs without it.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmkit.runtime._stats import current_rss_mb, peak_rss_mb

_log = logging.getLogger("llmkit.llama")


@dataclass(frozen=True)
class GgufSource:
    model: str                      # a local GGUF path, or a HF repo id to download from
    filename: str = "*q4_k_m.gguf"  # glob within the repo (ignored for a local path)
    revision: str | None = None     # immutable HF revision; ignored for a local path
    context_tokens: int = 4096
    chat: bool = False              # apply the model's embedded chat template (instruct models
    #                                 are trained with turn tokens; a raw prompt makes them ramble)
    temperature: float = 0.0        # 0.0 = greedy (the raw-completion default)
    top_p: float = 1.0
    top_k: int = 0                  # 0 = disabled
    min_p: float = 0.0


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
                "llama-cpp-python is a required mnemo dependency but is not importable — "
                "reinstall mnemo"
            ) from exc

        src = self._source
        started = time.monotonic()
        if os.path.exists(src.model):
            model_path = src.model
        else:
            from llmkit.runtime.hf_cache import resolve_snapshot

            local = resolve_snapshot(
                src.model,
                revision=src.revision,
                cache_dir=self._cache_dir,
                allow_patterns=[src.filename],
            )
            matches = sorted(Path(local).glob(src.filename))
            if not matches:
                raise RuntimeError(f"no file matching {src.filename!r} in {src.model}")
            model_path = str(matches[0])
        self._llama = Llama(
            model_path=model_path, n_ctx=src.context_tokens,
            n_gpu_layers=-1, verbose=False,  # offload to Metal when present, else CPU
        )
        _log.info(
            "generator loaded model=%s load=%.2fs rss=%.0fMB peak=%.0fMB",
            src.model, time.monotonic() - started, current_rss_mb(), peak_rss_mb(),
        )

    def unload(self) -> None:
        if self._llama is None:
            return
        try:
            if hasattr(self._llama, "close"):
                self._llama.close()
        except Exception:  # noqa: BLE001 — never keep a half-closed handle; drop it regardless
            _log.warning(
                "generator close() failed model=%s; dropping the handle anyway",
                self._source.model, exc_info=True,
            )
        finally:
            self._llama = None  # ALWAYS end unloaded — a later load() re-initialises
        _log.info(
            "generator freed model=%s rss=%.0fMB peak=%.0fMB",
            self._source.model, current_rss_mb(), peak_rss_mb(),
        )

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        src = self._source
        started = time.monotonic()
        if src.chat:
            # Instruct models need their chat template (turn tokens); llama.cpp applies the
            # GGUF's embedded template here. A raw prompt to these models rambles/confabulates.
            out = self._llama.create_chat_completion(
                messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens,
                temperature=src.temperature, top_p=src.top_p, top_k=src.top_k, min_p=src.min_p,
            )
            text = out["choices"][0]["message"]["content"]
        else:
            out = self._llama.create_completion(
                prompt, max_tokens=max_tokens, temperature=src.temperature,
            )
            text = out["choices"][0]["text"]
        _log.info(
            "generated chat=%s max_tokens=%d in %.2fs",
            src.chat, max_tokens, time.monotonic() - started,
        )
        return text
