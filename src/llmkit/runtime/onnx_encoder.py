"""Shared ONNX runtime for transformer encoders (embedder / reranker / NLI).

It does the mechanics common to all three — tokenize (a single text or a text pair) →
run an ONNX Runtime session → return the raw output — and nothing about how that output
is interpreted (that is the capability's "head"). Loaded on demand and freed on unload,
with a lean (non-arena) session so it does not pre-grab memory. ``onnxruntime`` /
``tokenizers`` / ``huggingface-hub`` are imported lazily, so importing this module is cheap.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmkit.runtime._stats import peak_rss_mb

_log = logging.getLogger("llmkit.onnx")


@dataclass(frozen=True)
class OnnxSource:
    repo: str                              # HF repo id holding the ONNX export + tokenizer
    onnx_file: str = "onnx/model.onnx"
    tokenizer_file: str = "tokenizer.json"
    revision: str | None = None
    max_input: int = 512                   # truncation cap (a cross-encoder window is ~512)


class OnnxEncoderRuntime:
    def __init__(self, source: OnnxSource, *, cache_dir: str | None = None) -> None:
        self._source = source
        self._cache_dir = cache_dir
        self._np: Any = None
        self._tokenizer: Any = None
        self._session: Any = None
        self._inputs: frozenset[str] = frozenset()

    def load(self) -> None:
        if self._session is not None:
            return  # idempotent
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        src = self._source
        started = time.monotonic()
        local = snapshot_download(
            src.repo,
            revision=src.revision,
            cache_dir=self._cache_dir,
            allow_patterns=[f"{src.onnx_file}*", src.tokenizer_file],
        )
        tokenizer = Tokenizer.from_file(str(Path(local) / src.tokenizer_file))
        tokenizer.enable_truncation(max_length=src.max_input)
        tokenizer.enable_padding()  # pad to the longest in each batch
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False  # no greedy pre-allocation; freed on unload
        session = ort.InferenceSession(
            str(Path(local) / src.onnx_file),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._np = np
        self._tokenizer = tokenizer
        self._session = session
        self._inputs = frozenset(spec.name for spec in session.get_inputs())
        _log.info(
            "encoder loaded model=%s load=%.2fs peak_rss=%.0fMB",
            src.repo, time.monotonic() - started, peak_rss_mb(),
        )

    def unload(self) -> None:
        if self._session is None:
            return
        self._session = None
        self._tokenizer = None
        self._inputs = frozenset()
        _log.info("encoder freed model=%s peak_rss=%.0fMB", self._source.repo, peak_rss_mb())

    def forward(self, texts: Sequence[str]) -> Any:
        """Raw model output for single texts (e.g. token embeddings to pool)."""
        return self._run(list(texts))

    def forward_pairs(self, pairs: Sequence[tuple[str, str]]) -> Any:
        """Raw model output for text pairs (e.g. cross-encoder logits)."""
        return self._run([(left, right) for left, right in pairs])

    def _run(self, inputs: list[Any]) -> Any:
        started = time.monotonic()
        np = self._np
        encoded = self._tokenizer.encode_batch(inputs)
        feed = {"input_ids": np.array([e.ids for e in encoded], dtype=np.int64)}
        if "attention_mask" in self._inputs:
            feed["attention_mask"] = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)
        output = self._session.run(None, feed)[0]
        _log.info(
            "encoder ran n=%d in %.3fs peak_rss=%.0fMB",
            len(inputs), time.monotonic() - started, peak_rss_mb(),
        )
        return output
