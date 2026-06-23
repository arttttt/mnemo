#!/usr/bin/env python3
"""Score dumped candidates with jina-reranker-v3 (MLX / Metal GPU).

Runs in the ISOLATED mlx venv — imports only stdlib + Jina's bundled `rerank.py` (mlx-lm) +
tools.eval.metrics (pure, no mnemo). Reports the same top-1 A/B as the in-process backends.

    <mlxvenv>/bin/python -m tools.eval.scorers.jina_v3_mlx \
        --model-dir <jina-v3-mlx> --candidates tools/results/candidates.json
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="the jina-reranker-v3-MLX checkout dir")
    p.add_argument("--candidates", required=True)
    p.add_argument("--out")
    args = p.parse_args()

    # Resolve paths and the repo root BEFORE chdir (rerank.py wants cwd = model dir).
    repo_root = str(Path(__file__).resolve().parents[3])
    candidates = Path(args.candidates).resolve()
    out = Path(args.out).resolve() if args.out else None
    sys.path.insert(0, repo_root)
    from tools.eval.metrics import report_ab, score_candidates

    os.chdir(args.model_dir)
    sys.path.insert(0, args.model_dir)
    from rerank import MLXReranker

    reranker = MLXReranker()
    print("loaded jina-v3 MLXReranker on MLX/Metal")

    def rank_fn(query, docs):
        scores = [0.0] * len(docs)
        for item in reranker.rerank(query, docs):  # sorted; item has 'index' + 'relevance_score'
            scores[item["index"]] = item["relevance_score"]
        return scores

    cands = json.loads(candidates.read_text())
    result = report_ab("jina-reranker-v3 (MLX/Metal)", score_candidates(cands, rank_fn))
    if out:
        out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
