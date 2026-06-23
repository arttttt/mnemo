#!/usr/bin/env python3
"""LoCoMo Tier-1 retrieval benchmark — LLM-free, the SEARCH path only (no recall).

Validates the harness machinery against a public standard; NOT a measure of mnemo on its
real domain (project facts). Read the per-category breakdown, never the aggregate.

Per conversation: register a project, ingest each dialog turn as a memory (tagged
dialog:<dia_id>), then per QA run `search` scoped to the conversation, map returned memory
ids back to dia_ids, and score Recall@k / MRR@k — plus AnyEvidence@k / CompleteEvidence@k,
which split multi-hop into found-one (coverage) vs found-none (relevance). Category 5
(adversarial) is the should-refuse slice, reported apart; --abstention adds the input-gate
curve on raw cosine.

    python -m tools.eval.locomo --embedder pplx --store-dir tools/.locomo_store
    python -m tools.eval.locomo --embedder hash --conversations 1   # machinery smoke
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from mnemo.application.results.search_result import SearchResult
from mnemo.application.use_cases.create_project import ProjectAlreadyExists

from tools.eval import core


def ingest(container, data, conversations, max_turns):
    id_to_dia: dict[str, set[str]] = defaultdict(set)
    n_turns = 0
    convs = data if conversations is None else data[:conversations]
    for sample in convs:
        slug = sample["sample_id"]
        try:
            container.create_project.execute(slug)
        except ProjectAlreadyExists:
            pass
        per_conv = 0
        for _, date, turn in core.iter_turns(sample["conversation"]):
            if max_turns is not None and per_conv >= max_turns:
                break
            result = container.remember.execute(
                content=core.turn_content(turn.get("speaker", ""), date, turn),
                type="discussion", scope="project", project=slug, tags=[f"dialog:{turn['dia_id']}"],
            )
            id_to_dia[result.id].add(turn["dia_id"])
            per_conv += 1
            n_turns += 1
        print(f"  ingested {slug}: {per_conv} turns", flush=True)
    return id_to_dia, n_turns


def evaluate(container, data, id_to_dia, k_list, conversations, abstention):
    limit = max(k_list)
    per_category: dict[int, core.Bucket] = defaultdict(core.Bucket)
    answerable = core.Bucket()
    adversarial = core.Bucket()
    pos, neg = [], []

    total = len(core.eligible_questions(data, conversations, answerable_only=False))
    print(f"querying {total} questions…", flush=True)
    n = 0
    convs = data if conversations is None else data[:conversations]
    for sample in convs:
        slug = sample["sample_id"]
        for qa in sample["qa"]:
            evidence = set(qa.get("evidence") or [])
            if not evidence:
                continue
            hits: list[SearchResult] = container.search.execute(
                query=qa["question"], scope="project", project=slug, limit=limit
            )
            ranked = [id_to_dia.get(h.id, set()) for h in hits]
            n += 1
            if n % 200 == 0:
                print(f"  queried {n}/{total}", flush=True)
            if qa.get("category") == core.ADVERSARIAL:
                adversarial.add(ranked, evidence, k_list)
            else:
                per_category[qa["category"]].add(ranked, evidence, k_list)
                answerable.add(ranked, evidence, k_list)
            if abstention and hits:
                top1 = core.cosine(container.embedder.encode(qa["question"]),
                                   container.embedder.encode(hits[0].content))
                (neg if qa.get("category") == core.ADVERSARIAL else pos).append(top1)

    report = {
        "n_questions": n,
        "answerable": answerable.summary(k_list),
        "by_category": {str(c): {"label": core.CATEGORY_LABELS.get(c, "?"), **per_category[c].summary(k_list)}
                        for c in sorted(per_category)},
        "adversarial_retrieval": adversarial.summary(k_list),
    }
    if abstention:
        report["abstention"] = core.abstention_curve(pos, neg)
    return report


def print_report(report, meta, k_list):
    bar = "=" * 78
    print(f"\n{bar}\nLoCoMo Tier-1 retrieval (mnemo search, no recall)\n{bar}")
    print(f"embedder={meta['embedder']} conversations={meta['conversations']} "
          f"turns={meta['n_turns']} questions={report['n_questions']}")
    header = "slice".ljust(22) + "n".rjust(5) + "  " + "  ".join(f"R@{k}".rjust(7) for k in k_list) + "   MRR".rjust(9)
    print("\n" + header + "\n" + "-" * len(header))

    def row(name, s):
        if not s.get("n"):
            return
        cells = "  ".join(f"{s['recall_at_k'][k]:.4f}".rjust(7) for k in k_list)
        print(name.ljust(22) + str(s["n"]).rjust(5) + "  " + cells + f"{s['mrr']:.4f}".rjust(9))

    row("ANSWERABLE (1-4)", report["answerable"])
    for cat, s in report["by_category"].items():
        row(f"  {cat} {s['label']}", s)
    row("adversarial (5)*", report["adversarial_retrieval"])
    print("\n* cat 5 is the should-REFUSE slice; retrieval shown for visibility, not credit.")

    # Any-vs-Complete diagnostic (multi-hop): found-one vs found-all.
    kk = max(k_list)
    print(f"\nmulti-hop diagnostic @ k={kk}  (Any = >=1 hop, Complete = all hops)")
    for cat, s in report["by_category"].items():
        if s.get("n"):
            print(f"  {cat} {s['label']:<12} Any={s['any_at_k'][kk]:.3f}  Complete={s['complete_at_k'][kk]:.3f}")

    if "abstention" in report and "curve" in report["abstention"]:
        ab = report["abstention"]
        print(f"\nAbstention (input-gate, raw cosine) mean: ans={ab['mean_cosine_answerable']} adv={ab['mean_cosine_adversarial']}")
        for pt in ab["curve"]:
            if pt["T"] in (0.4, 0.5, 0.6):
                print(f"  T={pt['T']}  false-refuse(ans)={pt['false_refusal_on_answerable']:.3f}  true-refuse(adv)={pt['true_refusal_on_adversarial']:.3f}")
    print(bar)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedder", default="pplx", choices=["pplx", "hash"])
    p.add_argument("--data", type=Path, default=core.DEFAULT_DATA)
    p.add_argument("--store-dir", type=Path, default=None)
    p.add_argument("--models-dir", default=os.path.expanduser("~/.mnemo/models"))
    p.add_argument("--conversations", type=int, default=None)
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--k", default="1,3,5,10,20")
    p.add_argument("--abstention", action="store_true")
    p.add_argument("--skip-ingest", action="store_true")
    args = p.parse_args()

    k_list = sorted({int(x) for x in args.k.split(",")})
    data = core.load_dataset(args.data)

    tmp = None
    if args.store_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="mnemo-locomo-")
        store_dir = Path(tmp.name)
    else:
        store_dir = args.store_dir

    print(f"isolated store: {store_dir}  embedder={args.embedder}")
    container = core.isolated_container(store_dir, args.embedder, args.models_dir)

    started = time.time()
    if args.skip_ingest:
        id_to_dia = core.load_manifest(store_dir, args.embedder)
        n_turns = sum(len(v) for v in id_to_dia.values())
    else:
        print("ingesting (turns embed inline via the sync scheduler):")
        id_to_dia, n_turns = ingest(container, data, args.conversations, args.max_turns)
        core.save_manifest(store_dir, args.embedder, id_to_dia, n_turns)
    print(f"ingest/bridge done in {time.time() - started:.1f}s; querying…")

    report = evaluate(container, data, id_to_dia, k_list, args.conversations, args.abstention)
    meta = {"embedder": args.embedder, "conversations": args.conversations or len(data),
            "n_turns": n_turns, "k": k_list, "elapsed_seconds": round(time.time() - started, 1)}
    print_report(report, meta, k_list)

    core.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = core.RESULTS_DIR / f"locomo_{args.embedder}.json"
    out.write_text(json.dumps({"meta": meta, "report": report}, indent=2))
    print(f"wrote {out}")
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
