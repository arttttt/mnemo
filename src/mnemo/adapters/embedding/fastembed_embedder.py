"""Local ONNX embeddings via fastembed (default BAAI/bge-small-en-v1.5).

`fastembed` is imported lazily so the core and offline tests don't require it.
The model is configurable via `model_name` (wired from MNEMO_EMBED_MODEL).
"""
from __future__ import annotations

from mnemo.application.types import Vector

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MAX_INPUT = 512  # bge-small window; used only if the tokenizer can't be read


class FastEmbedEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding  # lazy, heavy/optional dependency

        self._model = TextEmbedding(model_name=model_name)
        self._tokenizer = self._resolve_tokenizer()
        self._max_input = self._resolve_max_input()
        self._dim = len(self.encode("dimension probe"))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_input(self) -> int:
        return self._max_input

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text).ids)
        return len(text.split())  # fallback if fastembed internals are unavailable

    def encode(self, text: str) -> Vector:
        embedding = next(iter(self._model.embed([text])))
        return [float(value) for value in embedding]

    def _resolve_tokenizer(self):
        # fastembed keeps the HF `tokenizers.Tokenizer` on the loaded worker model;
        # the path is internal, so reach for it defensively.
        model = getattr(self._model, "model", None)
        return getattr(model, "tokenizer", None)

    def _resolve_max_input(self) -> int:
        truncation = getattr(self._tokenizer, "truncation", None)
        if isinstance(truncation, dict) and truncation.get("max_length"):
            return int(truncation["max_length"])
        return DEFAULT_MAX_INPUT
