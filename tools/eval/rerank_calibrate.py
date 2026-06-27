#!/usr/bin/env python3
"""Rerank-gate calibration — measure WHERE bge helps vs hurts, and which signal predicts it.

For each scorable question: build the fused candidate pool + the per-query SIGNALS, then score
the gold's rank with NO rerank vs with bge reranking the WHOLE pool. Reranking everything (the
gate OFF) decouples bge's effect from any gate, so the gate (when to fire bge) can be designed
offline from the (signals -> help/hurt) relationship. In-process (reads the fuser + signals
directly) — a research tool, NOT the product search path.

    python -m tools.eval.rerank_calibrate --store-dir <snapshot>/data --questions q.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from mnemo.application.fusion.fuser import Fuser
from mnemo.application.retrieval import Retrieval
from mnemo.application.search_criteria import SearchCriteria

from tools.eval import core
from tools.eval.models import build_reranker


def _gold_rank(keys: list, gold: set) -> int | None:
    return next((i for i, k in enumerate(keys) if k in gold), None)


def _effect(off: int | None, bge: int | None) -> str:
    if off is None or bge is None:
        return "na"
    if bge < off:
        return "help"
    if bge > off:
        return "hurt"
    return "same"


def collect(container, fuser, reranker, questions, pool):
    rows = []
    for q in questions:
        gold = set(q.get("gold_keys") or [])
        if not gold:
            continue  # irrelevant slice — no gold to rank
        retrieval = Retrieval(
            criteria=SearchCriteria(scope="project", project=q["project"]),
            limit=pool, text=q["question"],
            vector=container.embedder.encode(q["question"]),
        )
        fused = fuser.fuse(container.repository.retrieve_channels(retrieval), pool)
        items = fused.pool
        if not items:
            continue
        off_order = [h.memory.topic_key for h in items]
        scores = reranker.rank(q["question"], [h.memory.content for h in items])
        order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
        bge_order = [items[i].memory.topic_key for i in order]

        def _rank_in(order: list, key) -> int | None:
            return order.index(key) if key in order else None

        # Per-gold (per-hop) ranks: None = the hop is OUTSIDE the pool, so reranking can't reach it.
        golds = [{"key": gk, "off": _rank_in(off_order, gk), "bge": _rank_in(bge_order, gk)}
                 for gk in q["gold_keys"]]
        present_off = [g["off"] for g in golds if g["off"] is not None]
        present_bge = [g["bge"] for g in golds if g["bge"] is not None]
        off_rank = min(present_off) if present_off else None  # rank of the FIRST gold reached
        bge_rank = min(present_bge) if present_bge else None
        s = fused.signals
        rows.append({
            "id": q["id"], "slice": q["slice"],
            "dense_top1": round(s.dense_top1, 4), "dense_margin": round(s.dense_margin, 4),
            "agree": s.agree, "overlap": s.overlap,
            "off_rank": off_rank, "bge_rank": bge_rank, "golds": golds,
            "effect": _effect(off_rank, bge_rank),
        })
    return rows


def _gate_fires(row, floor: float) -> bool:
    """The current policy shape: rerank when the top dense hit is weak OR the legs disagree."""
    return row["dense_top1"] < floor or not row["agree"]


def report(rows: list[dict]) -> None:
    bar = "=" * 92
    print(f"\n{bar}\nRerank-gate calibration — bge over the whole pool (gate OFF), per question\n{bar}")
    print(f"{'id':24}{'slice':11}{'d_top1':>8}{'margin':>8}{'agree':>7}{'overlap':>8}"
          f"{'off→bge':>10}{'effect':>7}")
    for r in sorted(rows, key=lambda r: (r["effect"], r["id"])):
        off = "M" if r["off_rank"] is None else r["off_rank"] + 1
        bge = "M" if r["bge_rank"] is None else r["bge_rank"] + 1
        print(f"{r['id']:24}{r['slice']:11}{r['dense_top1']:>8.3f}{r['dense_margin']:>8.3f}"
              f"{str(r['agree']):>7}{r['overlap']:>8.2f}{f'{off}→{bge}':>10}{r['effect']:>7}")

    counts = defaultdict(int)
    for r in rows:
        counts[r["effect"]] += 1
    print(f"\nEFFECT on the FIRST gold (reranking EVERY query): help={counts['help']} "
          f"hurt={counts['hurt']} same={counts['same']} na={counts['na']}  (n={len(rows)})")

    # The first-gold effect MISLEADS multi-gold rows — score the metric each slice actually needs:
    # R@1 (first gold at rank 1) AND Complete@k (ALL golds in top-k, the multi-hop coverage metric).
    print("\nPER-SLICE off→bge  (R@1 = first gold@1; C@3/C@5 = ALL golds in top-k)")
    by_slice = defaultdict(list)
    for r in rows:
        by_slice[r["slice"]].append(r)

    def _r1(rs, which):
        return sum(1 for r in rs if r[f"{which}_rank"] == 0) / len(rs)

    def _complete(rs, which, k):
        return sum(1 for r in rs
                   if all(g[which] is not None and g[which] < k for g in r["golds"])) / len(rs)

    for sl in sorted(by_slice):
        rs = by_slice[sl]
        print(f"  {sl:11} n={len(rs):2}  R@1 {_r1(rs, 'off'):.2f}→{_r1(rs, 'bge'):.2f}"
              f"   C@3 {_complete(rs, 'off', 3):.2f}→{_complete(rs, 'bge', 3):.2f}"
              f"   C@5 {_complete(rs, 'off', 5):.2f}→{_complete(rs, 'bge', 5):.2f}")

    # Where does bge help vs hurt, by the candidate signals?
    for eff in ("help", "hurt"):
        sub = [r for r in rows if r["effect"] == eff]
        if not sub:
            continue
        d = [r["dense_top1"] for r in sub]
        dis = sum(1 for r in sub if not r["agree"])
        print(f"  {eff:4}: dense_top1 min/mean/max = {min(d):.3f}/{sum(d)/len(d):.3f}/{max(d):.3f}"
              f"  disagree={dis}/{len(sub)}")

    # Multi-gold (multi-hop) needs COVERAGE of every hop, not just the first — show each hop's
    # off->bge rank; '·' = the hop is outside the pool (a retrieval miss, not a rerank problem).
    multi = [r for r in rows if len(r["golds"]) > 1]
    if multi:
        print("\nMULTI-GOLD per-hop (off→bge rank in the pool; '·' = NOT retrieved into the pool)")
        for r in multi:
            hops = "   ".join(
                f"{g['key'].split('/')[-1][:20]} "
                f"{('·' if g['off'] is None else g['off'] + 1)}→{('·' if g['bge'] is None else g['bge'] + 1)}"
                for g in r["golds"]
            )
            print(f"  {r['id']:22}[{r['slice']:9}] d_top1={r['dense_top1']:.2f} "
                  f"agree={str(r['agree']):5} {hops}")

    # Confusion of the CURRENT gate (does it fire on helps, skip hurts?) over a floor sweep.
    print("\nGATE SWEEP  (rerank if dense_top1<floor OR disagree) — want: fire on help, skip hurt")
    print(f"  {'floor':>6}{'help_fired':>12}{'hurt_fired':>12}{'same_fired':>12}{'net(help-hurt)':>16}")
    helps = [r for r in rows if r["effect"] == "help"]
    hurts = [r for r in rows if r["effect"] == "hurt"]
    sames = [r for r in rows if r["effect"] == "same"]
    for floor in (0.0, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 1.01):
        hf = sum(_gate_fires(r, floor) for r in helps)
        uf = sum(_gate_fires(r, floor) for r in hurts)
        sf = sum(_gate_fires(r, floor) for r in sames)
        print(f"  {floor:>6.2f}{hf:>7}/{len(helps):<4}{uf:>7}/{len(hurts):<4}{sf:>7}/{len(sames):<4}{hf - uf:>16}")
    print(bar)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--embedder", default="pplx")
    p.add_argument("--reranker", default="bge")
    p.add_argument("--pool", type=int, default=50)
    p.add_argument("--models-dir", default=os.path.expanduser("~/.mnemo/models"))
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    container = core.isolated_container(args.store_dir, args.embedder, args.models_dir)
    fuser = Fuser()
    reranker = build_reranker(args.reranker, args.models_dir)
    questions = json.loads(args.questions.read_text())["questions"]
    print(f"store={args.store_dir}  embedder={args.embedder}  reranker={args.reranker}  pool={args.pool}")

    rows = collect(container, fuser, reranker, questions, args.pool)
    report(rows)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
