#!/usr/bin/env python3
"""Strict prod-state eval — score mnemo's REAL search CLI on a COPY of the live bank.

Unlike eval.domain (which re-ingests a synthetic fixture into a fresh store, re-embedding
each memory), this measures the SHIPPED product on the REAL bank exactly as it stands:
  - the corpus is a file copy of the live store (authorized snapshot) — no re-ingest, no
    recompaction, no dropping of over-cap notes; the stored vectors are the real ones;
  - retrieval goes through the actual `mnemo search` CLI (the product boundary), parsed
    from its JSON output — NO raw embedder/reranker pokes;
  - hits are matched to gold by `topic_key` (stable across supersede), so the question set
    survives id churn.
Scores answerable/evolution slices by Recall@k / Any@k / Complete@k / MRR. The `irrelevant`
slice has no gold: the product has no refusal gate, so we just show what search returns
(the "search never abstains" readout). Reuses tools.eval.metrics.Bucket. Run:

    python -m tools.eval.prod_snapshot --store-dir <snapshot>/data --questions q.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from tools.eval.metrics import Bucket


def search_cli(store_dir: Path, embedder: str, question: str, project: str, limit: int) -> list[dict]:
    """Run the real `mnemo search` CLI against the snapshot store; return its JSON hits."""
    env = {**os.environ, "MNEMO_DATA_DIR": str(store_dir), "MNEMO_EMBEDDER": embedder}
    proc = subprocess.run(
        ["mnemo", "search", question, "--project", project, "--limit", str(limit), "--json"],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"mnemo search failed for {question!r}: {proc.stderr.strip()[-400:]}")
    return json.loads(proc.stdout)


def evaluate(store_dir: Path, embedder: str, questions: list[dict], k_list: list[int]) -> dict:
    topk = max(k_list)
    buckets: dict[str, Bucket] = defaultdict(Bucket)
    rows = []
    for q in questions:
        hits = search_cli(store_dir, embedder, q["question"], q["project"], topk)
        keys = [h.get("topic_key") for h in hits]          # ranked topic_keys (None if absent)
        gold = set(q.get("gold_keys", []))
        rank = next((i for i, k in enumerate(keys) if k in gold), None)
        if gold:
            buckets[q["slice"]].add([{k} if k else set() for k in keys], gold, k_list)
        rows.append({"id": q["id"], "slice": q["slice"], "has_gold": bool(gold),
                     "gold_rank": rank, "top3": keys[:3]})
    return {"slices": {name: b.summary(k_list) for name, b in buckets.items()}, "per_question": rows}


def print_report(report: dict, meta: dict, k_list: list[int]) -> None:
    bar = "=" * 72
    print(f"\n{bar}\nStrict prod-state eval — real bank via `mnemo search` CLI\n{bar}")
    print(f"embedder={meta['embedder']}  store={meta['store']}  questions={meta['n_questions']}")
    for name, summary in report["slices"].items():
        if not summary.get("n"):
            continue
        head = "  ".join(f"@{k}".rjust(6) for k in k_list)
        recall = "  ".join(f"{summary['recall_at_k'][k]:.3f}".rjust(6) for k in k_list)
        anyk = "  ".join(f"{summary['any_at_k'][k]:.3f}".rjust(6) for k in k_list)
        print(f"\n{name.upper()} (n={summary['n']})   MRR={summary['mrr']:.3f}")
        print("  k     " + head)
        complete = "  ".join(f"{summary['complete_at_k'][k]:.3f}".rjust(6) for k in k_list)
        print("  recall" + recall)
        print("  any   " + anyk)
        print("  compl " + complete)
    miss = [r for r in report["per_question"] if r["has_gold"] and r["gold_rank"] is None]
    if miss:
        print(f"\nGOLD NOT FOUND in top-{max(k_list)} ({len(miss)}): " + ", ".join(r["id"] for r in miss))
    irr = [r for r in report["per_question"] if not r["has_gold"]]
    if irr:
        print(f"\nIRRELEVANT (no gold; product returns hits anyway — no refusal): n={len(irr)}")
    print(bar)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store-dir", type=Path, required=True, help="snapshot data dir (copy of the live store)")
    p.add_argument("--questions", type=Path, required=True, help="authored question set (JSON)")
    p.add_argument("--embedder", default="pplx", help="MUST match the embedder the bank was stored with")
    p.add_argument("--k", default="1,3,5,10")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    k_list = sorted({int(x) for x in args.k.split(",")})
    spec = json.loads(args.questions.read_text())
    questions = spec["questions"]
    print(f"store={args.store_dir}  embedder={args.embedder}  questions={len(questions)}")

    report = evaluate(args.store_dir, args.embedder, questions, k_list)
    meta = {"embedder": args.embedder, "store": str(args.store_dir),
            "n_questions": len(questions), "k": k_list, "questions_file": str(args.questions)}
    print_report(report, meta, k_list)

    if args.out:
        args.out.write_text(json.dumps({"meta": meta, "report": report}, indent=2, ensure_ascii=False))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
