"""Config-driven reranker registry for the eval: a model is a SPEC, built behind a uniform .rank.

mnemo's chosen production reranker is bge-reranker-v2-m3 (the LoCoMo + MIRACL A/B winner; see the
bench/reranker-selection memory) and is the only entry we run. ``build_reranker(name)`` keeps the
RUNTIME abstracted (GGUF on Metal here) behind ``.rank(query, docs) -> scores``, so testing another
model later is a single spec line — no runner changes, no new plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RerankerSpec:
    name: str
    runtime: str                  # "gguf" | "onnx" — selects the abstracted backend
    repo: str
    file: str | None = None       # GGUF filename (gguf) / ONNX weights path (onnx)
    window: int = 2048
    sep: str = "</s></s>"         # XLM-R cross-encoder pair separator
    note: str = ""


# The chosen production reranker. Add a spec here to evaluate another model — nothing else changes.
RERANKERS: dict[str, RerankerSpec] = {
    "bge": RerankerSpec(
        name="bge", runtime="gguf",
        repo="gpustack/bge-reranker-v2-m3-GGUF", file="bge-reranker-v2-m3-Q8_0.gguf", window=2048,
        note="prod reranker — bge-reranker-v2-m3, XLM-R cross-encoder, Q8 GGUF on Metal",
    ),
}


def build_reranker(name: str, models_dir: str):
    """Build the named reranker behind a uniform ``.rank(query, docs) -> scores``, dispatching on
    its runtime so the runner stays runtime-agnostic. Raises on an unknown name or runtime."""
    spec = RERANKERS.get(name)
    if spec is None:
        raise SystemExit(f"unknown reranker {name!r}; known: {sorted(RERANKERS)}")
    from tools.eval import rerankers
    if spec.runtime == "gguf":
        return rerankers.GgufReranker(spec.repo, spec.file, models_dir, sep=spec.sep, n_ctx=spec.window)
    if spec.runtime == "onnx":
        return rerankers.build_onnx_reranker(spec.repo, models_dir, spec.window, spec.file or "onnx/model.onnx")
    raise SystemExit(f"unknown runtime {spec.runtime!r} for reranker {name!r}")
