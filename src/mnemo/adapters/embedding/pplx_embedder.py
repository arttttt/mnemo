"""pplx-embed-v1-0.6b (int8 ONNX) on CPU — the default semantic embedder.

Raw onnxruntime, no torch / sentence-transformers: tokenize → ONNX graph →
mean-pool over the attention mask → L2-normalize. The int8 weights + tokenizer are
fetched once from a pinned HF revision into ``~/.mnemo/models/pplx`` and then run
fully offline. dim 1024, mean pooling, no instruction prefix.

`onnxruntime`/`tokenizers`/`huggingface_hub` are the optional ``pplx`` extra, imported
lazily so the core and offline tests don't require them.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO = "perplexity-ai/pplx-embed-v1-0.6b"
_REVISION = "2c4d510dd4a732063c31a0f70193e35067b51fd8"  # pinned: switching it = a reindex
_ONNX_FILE = "onnx/model_quantized.onnx"  # int8
_DIM = 1024
DEFAULT_MAX_TOKENS = 2048  # the operational window cap (<= the model's 32K), keeps writes cheap


class PplxEmbedder:
    def __init__(self, max_input: int = DEFAULT_MAX_TOKENS, models_dir: str | None = None) -> None:
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        self._np = np
        self._max_input = max_input
        base = models_dir or os.path.expanduser(
            os.environ.get("MNEMO_MODELS_DIR", "~/.mnemo/models")
        )
        target = Path(base) / "pplx"
        target.mkdir(parents=True, exist_ok=True)
        # One-time fetch (then offline): the int8 graph + its weight sidecar + tokenizer.
        local = snapshot_download(
            _REPO,
            revision=_REVISION,
            local_dir=str(target),
            allow_patterns=[f"{_ONNX_FILE}*", "tokenizer.json"],
        )
        self._tokenizer = Tokenizer.from_file(str(Path(local) / "tokenizer.json"))
        self._session = ort.InferenceSession(
            str(Path(local) / _ONNX_FILE), providers=["CPUExecutionProvider"]
        )
        self._output = self._session.get_outputs()[0].name

    @property
    def dim(self) -> int:
        return _DIM

    @property
    def max_input(self) -> int:
        return self._max_input

    def count_tokens(self, text: str) -> int:
        # Full (untruncated) count — the write use case rejects oversize content with it.
        return len(self._tokenizer.encode(text).ids)

    def encode(self, text: str) -> list[float]:
        np = self._np
        encoded = self._tokenizer.encode(text)
        ids = encoded.ids[: self._max_input]  # truncate here guards the query path (search)
        mask = encoded.attention_mask[: self._max_input]
        input_ids = np.array([ids], dtype=np.int64)
        attention = np.array([mask], dtype=np.int64)
        out = self._session.run(
            [self._output], {"input_ids": input_ids, "attention_mask": attention}
        )[0]
        if out.ndim == 3:  # token embeddings [1, seq, dim] → masked mean
            weights = attention[:, :, None].astype(np.float32)
            pooled = (out * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        else:  # already a sentence embedding [1, dim]
            pooled = out
        vector = pooled[0]
        norm = float(np.linalg.norm(vector)) or 1.0
        return [float(value) / norm for value in vector]
