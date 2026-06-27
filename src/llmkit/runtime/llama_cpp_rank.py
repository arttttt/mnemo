"""llama.cpp runtime for a cross-encoder RERANKER (GGUF, Metal/CPU).

A cross-encoder reranker is a different llama.cpp mode from text generation: the model is
loaded with ``embedding=True`` and ``pooling_type=RANK`` so ``create_embedding`` returns a
single relevance logit per pre-joined ``query<sep>doc`` pair (higher = more relevant) instead
of generating tokens. It reuses :class:`GgufSource` (``context_tokens`` is the n_ctx, sized
to also drive n_batch/n_ubatch so a whole shortlist encodes in one batch).

``llama_cpp`` is imported lazily inside ``load`` (with an actionable error and the
``LLAMA_POOLING_TYPE_RANK`` constant read from it there), so importing this module is cheap
and the rest of llmkit runs without it.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llmkit.runtime._stats import current_rss_mb, peak_rss_mb
from llmkit.runtime.llama_cpp import GgufSource

_log = logging.getLogger("llmkit.llama.rank")


class LlamaCppRerankRuntime:
    """Loads a GGUF cross-encoder in llama.cpp's RANK mode and scores pre-joined pairs.

    The proven algorithm (verified for bge / jina XLM-R cross-encoders): build the model with
    ``embedding=True, pooling_type=RANK``, then ``create_embedding`` over ``query<sep>doc``
    strings yields one relevance logit each (``embedding[0]``).
    """

    def __init__(self, source: GgufSource, *, cache_dir: str | None = None) -> None:
        self._source = source
        self._cache_dir = cache_dir
        self._llama: Any = None

    def load(self) -> None:
        if self._llama is not None:
            return  # idempotent
        try:
            import llama_cpp
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
        # n_batch/n_ubatch = n_ctx so a whole shortlist encodes in one batch; RANK pooling
        # makes create_embedding return the relevance logit rather than a hidden state.
        self._llama = Llama(
            model_path=model_path, n_ctx=src.context_tokens,
            n_gpu_layers=-1, embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_RANK,
            n_batch=src.context_tokens, n_ubatch=src.context_tokens, verbose=False,
        )
        _log.info(
            "reranker loaded model=%s load=%.2fs rss=%.0fMB peak=%.0fMB",
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
                "reranker close() failed model=%s; dropping the handle anyway",
                self._source.model, exc_info=True,
            )
        finally:
            self._llama = None  # ALWAYS end unloaded — a later load() re-initialises
        _log.info(
            "reranker freed model=%s rss=%.0fMB peak=%.0fMB",
            self._source.model, current_rss_mb(), peak_rss_mb(),
        )

    def rank(self, pairs: Sequence[str]) -> list[float]:
        """A relevance logit per pre-joined ``query<sep>doc`` pair, aligned to ``pairs``."""
        pairs = list(pairs)
        if not pairs:
            return []
        started = time.monotonic()
        data = self._llama.create_embedding(input=pairs)["data"]
        scores = [float(item["embedding"][0]) for item in data]
        _log.info("ranked pairs=%d in %.2fs", len(pairs), time.monotonic() - started)
        return scores
