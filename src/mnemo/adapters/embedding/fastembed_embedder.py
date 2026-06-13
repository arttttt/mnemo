"""Local ONNX embeddings via fastembed (default BAAI/bge-small-en-v1.5).

`fastembed` is imported lazily so the core and offline tests don't require it.
"""
from __future__ import annotations

from mnemo.application.types import Vector

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding  # lazy, heavy/optional dependency

        self._model = TextEmbedding(model_name=model_name)
        self._dim = len(self.encode("dimension probe"))

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> Vector:
        embedding = next(iter(self._model.embed([text])))
        return [float(value) for value in embedding]
