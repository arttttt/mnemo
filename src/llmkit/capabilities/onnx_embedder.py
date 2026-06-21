"""Embedder capability over the shared ONNX encoder runtime.

The head: a masked mean-pool of the token embeddings, then optional L2-normalize — a
single sentence vector. (A model that already emits a 2D pooled output is used as-is.)
``dim`` and ``max_input`` come from config; the model loads and frees behind the
residency manager (an embedder is often resident).
"""
from __future__ import annotations

from llmkit.lifecycle.manager import ResidencyManager
from llmkit.ports.tokenizer import Tokenizer
from llmkit.runtime.onnx_encoder import EncoderOutput, OnnxEncoderRuntime
from llmkit.types import Vector


class OnnxEmbedder:
    def __init__(
        self,
        manager: ResidencyManager[OnnxEncoderRuntime],
        tokenizer: Tokenizer,
        *,
        dim: int,
        max_input: int,
        normalize: bool = True,
    ) -> None:
        self._manager = manager
        self._tokenizer = tokenizer
        self._dim = dim
        self._max_input = max_input
        self._normalize = normalize

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_input(self) -> int:
        return self._max_input

    def close(self) -> None:
        """Unload the pooled session instances (a Resident embedder keeps them warm until
        the service shuts down). Tokenization is unaffected — the tokenizer is separate."""
        self._manager.close()

    def count_tokens(self, text: str) -> int:
        return self._tokenizer.count(text)  # tokenizer-only — never leases a session

    def encode(self, text: str) -> Vector:
        encodings = self._tokenizer.encode_batch([text])
        with self._manager.use() as runtime:
            result = runtime.run(encodings)
        return self._pool(result)

    def _pool(self, result: EncoderOutput) -> Vector:
        import numpy as np

        output = result.output
        if output.ndim == 3:  # token embeddings [1, seq, dim] → masked mean
            weights = result.attention_mask[:, :, None].astype(np.float32)
            pooled = (output * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        else:  # already a sentence embedding [1, dim]
            pooled = output
        vector = pooled[0]
        if not self._normalize:
            return [float(value) for value in vector]
        norm = float(np.linalg.norm(vector)) or 1.0
        return [float(value) / norm for value in vector]
