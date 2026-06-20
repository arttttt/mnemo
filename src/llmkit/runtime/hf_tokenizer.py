"""HuggingFace tokenizer over a model's tokenizer.json — the concrete Tokenizer port.

Loads ONLY the tokenizer file (a few MB), never the ONNX session, lazily on first use.
So counting tokens / preparing inputs is cheap and independent of the heavy session pool.
"""
from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llmkit.ports.tokenizer import Encoding, TokenizerInput
from llmkit.runtime.onnx_encoder import OnnxSource


class HfTokenizer:
    def __init__(self, source: OnnxSource, *, cache_dir: str | None = None) -> None:
        self._source = source
        self._cache_dir = cache_dir
        self._tokenizer: Any = None
        self._lock = threading.Lock()

    def _ensure(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        with self._lock:
            if self._tokenizer is None:  # double-checked: only one thread downloads/parses
                from huggingface_hub import snapshot_download
                from tokenizers import Tokenizer as HfTok

                src = self._source
                local = snapshot_download(
                    src.repo,
                    revision=src.revision,
                    cache_dir=self._cache_dir,
                    allow_patterns=[src.tokenizer_file],  # tokenizer ONLY — no ONNX weights
                )
                self._tokenizer = HfTok.from_file(str(Path(local) / src.tokenizer_file))
        return self._tokenizer

    def count(self, text: str) -> int:
        return len(self._ensure().encode(text).ids)

    def encode_batch(self, inputs: Sequence[TokenizerInput]) -> list[Encoding]:
        encoded = self._ensure().encode_batch(list(inputs))
        return [
            Encoding(
                ids=list(e.ids),
                attention_mask=list(e.attention_mask),
                type_ids=list(e.type_ids),
            )
            for e in encoded
        ]
