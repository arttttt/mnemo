#!/usr/bin/env python3
"""Score dumped candidates with Qwen3-Reranker-0.6B (stock GGUF Q8, llama.cpp / Metal).

Qwen3-Reranker is an LLM yes/no reranker: each (query, doc) pair is wrapped in its instruction
template and pooling_type=RANK returns P(yes) (verified non-degenerate on the official
ggml-org GGUF). Runs in the project venv (llama-cpp-python). Same top-1 A/B as the others.

    python -m tools.eval.scorers.qwen3 --gguf <q8.gguf> --candidates tools/results/candidates.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.eval.metrics import report_ab, score_candidates  # noqa: E402

import llama_cpp  # noqa: E402
from llama_cpp import Llama  # noqa: E402

INSTR = "Given a web search query, retrieve relevant passages that answer the query"
PRE = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the '
       'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
       '<|im_end|>\n<|im_start|>user\n')
SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def fmt(q: str, d: str) -> str:
    return f"{PRE}<Instruct>: {INSTR}\n<Query>: {q}\n<Document>: {d}{SUF}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--out")
    args = p.parse_args()

    llm = Llama(model_path=args.gguf, embedding=True, pooling_type=llama_cpp.LLAMA_POOLING_TYPE_RANK,
                n_gpu_layers=-1, n_ctx=2048, n_batch=2048, n_ubatch=2048, verbose=False)

    def rank_fn(query, docs):
        scores = []
        for i in range(0, len(docs), 8):  # chunk to stay under n_batch
            data = llm.create_embedding(input=[fmt(query, d) for d in docs[i:i + 8]])["data"]
            scores.extend(float(it["embedding"][0]) for it in data)
        return scores

    cands = json.loads(open(args.candidates).read())
    overall, per_cat, single, ms = score_candidates(cands, rank_fn)
    res = report_ab("Qwen3-Reranker-0.6B (GGUF Q8/Metal)", overall, per_cat, single, ms)
    if args.out:
        open(args.out, "w").write(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
