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


@dataclass(frozen=True)
class EncoderOutput:
    output: Any           # raw model output: logits (classifier) or last_hidden_state (embedder)
    attention_mask: Any   # the padding mask used — heads that pool (the embedder) need it


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

    def count_tokens(self, text: str) -> int:
        """Untruncated token count — the write path rejects over-window content with it."""
        return len(self._tokenizer.encode(text).ids)

    def _run(self, inputs: list[Any]) -> EncoderOutput:
        started = time.monotonic()
        np = self._np
        cap = self._source.max_input
        # Truncate each sequence to the model's window, then pad the batch with zeros.
        # The attention mask is 0 over the padding, so the pad id itself is irrelevant.
        rows = [
            (e.ids[:cap], e.attention_mask[:cap], e.type_ids[:cap])
            for e in self._tokenizer.encode_batch(inputs)
        ]
        width = max((len(ids) for ids, _, _ in rows), default=0)

        def stack(index: int) -> Any:
            return np.array(
                [list(row[index]) + [0] * (width - len(row[index])) for row in rows],
                dtype=np.int64,
            )

        feed = {"input_ids": stack(0)}
        mask = stack(1)
        if "attention_mask" in self._inputs:
            feed["attention_mask"] = mask
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = stack(2)
        output = self._session.run(None, feed)[0]
        _log.info(
            "encoder ran n=%d in %.3fs peak_rss=%.0fMB",
            len(inputs), time.monotonic() - started, peak_rss_mb(),
        )
        return EncoderOutput(output=output, attention_mask=mask)
