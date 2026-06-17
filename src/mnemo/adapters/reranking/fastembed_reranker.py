"""Reranker adapter: a cross-encoder via fastembed (ONNX runtime, no torch).

Loads the model inside ``session()`` and frees it on exit, logging load/rank timing and
peak RSS so a real run's cost is visible. fastembed downloads the ONNX model on first
use and runs it on CPU — the same stack as the embedder. fastembed is imported lazily so
the rest of mnemo runs without it.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from mnemo.adapters.runtime_stats import peak_rss_mb
from mnemo.application.ports.reranker import LoadedReranker

_log = logging.getLogger("mnemo.reranker")


class FastEmbedReranker:
    def __init__(self, model_name: str, *, cache_dir: str | None = None) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir

    @contextmanager
    def session(self) -> Iterator[LoadedReranker]:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        start = time.monotonic()
        encoder = TextCrossEncoder(model_name=self._model_name, cache_dir=self._cache_dir)
        _log.info(
            "reranker loaded model=%s load=%.2fs peak_rss=%.0fMB",
            self._model_name, time.monotonic() - start, peak_rss_mb(),
        )
        try:
            yield _LoadedFastEmbedReranker(encoder)
        finally:
            del encoder
            _log.info("reranker released model=%s peak_rss=%.0fMB", self._model_name, peak_rss_mb())


class _LoadedFastEmbedReranker:
    def __init__(self, encoder) -> None:
        self._encoder = encoder

    def rank(self, query: str, documents: Sequence[str]) -> list[float]:
        start = time.monotonic()
        scores = list(self._encoder.rerank(query, list(documents)))
        _log.info(
            "reranked n=%d in %.3fs peak_rss=%.0fMB",
            len(documents), time.monotonic() - start, peak_rss_mb(),
        )
        return scores
