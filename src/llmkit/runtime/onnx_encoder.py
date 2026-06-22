"""Shared ONNX runtime for transformer encoders (embedder / reranker / NLI).

Runs PRE-TOKENIZED inputs through an ONNX Runtime session and returns the raw output —
nothing about how that output is interpreted (the capability's "head"), and nothing about
tokenization (that is the injected Tokenizer; see ports/tokenizer.py). Loaded on demand and
freed on unload, with a lean (non-arena) session so it does not pre-grab memory.
``onnxruntime`` / ``huggingface-hub`` are imported lazily, so importing this module is cheap.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmkit.ports.tokenizer import Encoding
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
        self._session: Any = None
        self._inputs: frozenset[str] = frozenset()

    def load(self) -> None:
        if self._session is not None:
            return  # idempotent
        import numpy as np
        import onnxruntime as ort

        from llmkit.runtime.hf_cache import resolve_snapshot

        src = self._source
        started = time.monotonic()
        local = resolve_snapshot(
            src.repo,
            revision=src.revision,
            cache_dir=self._cache_dir,
            allow_patterns=[f"{src.onnx_file}*"],  # the session weights only — tokenizer is separate
        )
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False  # no greedy pre-allocation; freed on unload
        session = ort.InferenceSession(
            str(Path(local) / src.onnx_file),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._np = np
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
        self._inputs = frozenset()
        _log.info("encoder freed model=%s peak_rss=%.0fMB", self._source.repo, peak_rss_mb())

    def run(self, encodings: Sequence[Encoding]) -> EncoderOutput:
        """Run a batch of PRE-TOKENIZED inputs through the session. Each encoding is
        truncated to the model's window, then the batch is zero-padded (the attention mask
        is 0 over padding, so the pad id is irrelevant)."""
        started = time.monotonic()
        np = self._np
        cap = self._source.max_input
        rows = [
            (e.ids[:cap], e.attention_mask[:cap], e.type_ids[:cap]) for e in encodings
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
            len(rows), time.monotonic() - started, peak_rss_mb(),
        )
        return EncoderOutput(output=output, attention_mask=mask)
