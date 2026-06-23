"""In-process reranker backends for the A/B harness.

- ONNX cross-encoder via llmkit's OnnxReranker (CPU). Works for XLM-R encoders (bge / jina-v2)
  and any ONNX classification-head reranker (pass --onnx-file for a quantized variant).
- GGUF cross-encoder via llama.cpp on Metal (GPU). The pair is pre-joined as `query<sep>doc`
  (XLM-R = `</s></s>`, verified for jina/bge); pooling_type=RANK returns the relevance logit.
  (Ref: the lilbee compute_rerank_scores convention.)

Both expose `.rank(query, documents) -> list[float]`. Out-of-process / LLM rerankers (jina-v3
MLX, Qwen3-Reranker) live under scorers/ — they read dumped candidates, not this interface.
"""
from __future__ import annotations

from collections.abc import Sequence

_RERANK_SEP = "</s></s>"


def build_onnx_reranker(repo: str, models_dir: str, max_tokens: int, onnx_file: str = "onnx/model.onnx"):
    """A cross-encoder reranker over llmkit's ONNX runtime (CPU). Resident = loaded once for the
    whole A/B (a bench convenience; the product wires rerankers Transient)."""
    from llmkit.build import build_reranker
    from llmkit.config import ModelConfig
    from llmkit.lifecycle.residency import Resident
    from llmkit.runtime.onnx_encoder import OnnxSource

    return build_reranker(ModelConfig(
        source=OnnxSource(repo=repo, onnx_file=onnx_file, max_input=max_tokens),
        residency=Resident(),
        cache_dir=models_dir or None,
    ))


class GgufReranker:
    """GGUF cross-encoder reranker via llama.cpp on Metal. Resident (loaded once) is a bench
    convenience; the product would wire it Transient (load-on-demand, unload after each use)."""

    def __init__(self, repo: str, gguf_file: str, models_dir: str, *, sep: str = _RERANK_SEP, n_ctx: int = 2048):
        import llama_cpp
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        path = hf_hub_download(repo, gguf_file, cache_dir=models_dir or None)
        self._sep = sep
        self._llm = Llama(
            model_path=path, embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_RANK,
            n_gpu_layers=-1, n_ctx=n_ctx, n_batch=n_ctx, n_ubatch=n_ctx, verbose=False,
        )

    def rank(self, query: str, documents: Sequence[str]) -> list[float]:
        documents = list(documents)
        if not documents:
            return []
        data = self._llm.create_embedding(input=[f"{query}{self._sep}{d}" for d in documents])["data"]
        return [float(item["embedding"][0]) for item in data]


def sanity_check(reranker) -> bool:
    """A known relevant doc must outscore irrelevant ones, non-degenerate. Catches a wrong
    separator / broken conversion before the A/B. Returns True if OK."""
    q = "When did Caroline go to the LGBTQ support group?"
    docs = ["Caroline (3 May 2023): I went to the LGBTQ support group on 7 May 2023.",
            "Melanie: I painted a sunrise this morning, the colors were beautiful.",
            "Caroline: The weather is really nice today, going for a run."]
    s = reranker.rank(q, docs)
    ok = s[0] == max(s) and max(abs(x) for x in s) > 1e-3
    print(f"  sanity: relevant={s[0]:+.4f} irrel={s[1]:+.4f},{s[2]:+.4f} "
          f"max|s|={max(abs(x) for x in s):.2e} -> {'OK' if ok else 'SUSPECT (wrong sep / broken?)'}")
    return ok
