#!/usr/bin/env python3
"""Reranker top-1 A/B over dumped candidates — in-process backends (ONNX CPU / GGUF Metal).

Reads candidates.json from dump_candidates and reports baseline@1 vs reranked@1 (delta,
win/loss). Out-of-process / LLM rerankers (jina-v3 MLX, Qwen3) live under scorers/ and read
the SAME candidates.json, so every reranker is scored identically.

    python -m tools.eval.dump_candidates --store-dir tools/.locomo_store
    python -m tools.eval.rerank_ab --backend gguf \
        --gguf-repo gpustack/bge-reranker-v2-m3-GGUF --gguf-file bge-reranker-v2-m3-Q8_0.gguf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.eval import core, rerankers


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", type=Path, default=core.RESULTS_DIR / "candidates.json")
    p.add_argument("--backend", default="gguf", choices=["onnx", "gguf"])
    p.add_argument("--reranker", help="ONNX cross-encoder repo (onnx backend)")
    p.add_argument("--onnx-file", default="onnx/model.onnx", help="ONNX weights path (e.g. onnx/model_int8.onnx)")
    p.add_argument("--gguf-repo", help="GGUF repo (gguf backend)")
    p.add_argument("--gguf-file", help="GGUF filename (gguf backend)")
    p.add_argument("--sep", default="</s></s>", help="query/doc separator for gguf (XLM-R = </s></s>)")
    p.add_argument("--reranker-max-tokens", type=int, default=512)
    p.add_argument("--models-dir", default=str(Path("~/.mnemo/models").expanduser()))
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    if args.backend == "gguf":
        if not (args.gguf_repo and args.gguf_file):
            raise SystemExit("--backend gguf needs --gguf-repo and --gguf-file")
        label = f"{args.gguf_repo.split('/')[-1]}/{args.gguf_file}"
        reranker = rerankers.GgufReranker(args.gguf_repo, args.gguf_file, args.models_dir, sep=args.sep)
    else:
        if not args.reranker:
            raise SystemExit("--backend onnx needs --reranker")
        label = args.reranker
        reranker = rerankers.build_onnx_reranker(
            args.reranker, args.models_dir, args.reranker_max_tokens, args.onnx_file)
    print(f"backend={args.backend}  model={label}")
    rerankers.sanity_check(reranker)

    cands = json.loads(args.candidates.read_text())
    overall, per_cat, single, ms = core.score_candidates(cands, reranker.rank)
    res = core.report_ab(label, overall, per_cat, single, ms, core.CATEGORY_LABELS)
    if args.out:
        args.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
