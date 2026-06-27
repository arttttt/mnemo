#!/usr/bin/env python3
"""Weighted-RRF fusion sweep — find the dense↔lexical balance between plain RRF and pure
dense-primary that keeps the semantic / multi-hop wins WITHOUT the exact-term regressions.

The fresh-snapshot A/B ([[bench/dense-primary-ab-result]]) showed pure dense-primary helps
semantic + multi-hop queries but buries exact-term lookups the BM25 leg ranked #1. This sweeps a
single weight: score(id) = lam/(k+rank_dense) + (1-lam)/(k+rank_lexical). lam=0.5 reproduces RRF
(equal weight → same ORDER), lam=1.0 reproduces dense-primary (lexical → 0, tails). In between,
dense LEADS the order but a strong lexical match still scores enough to stay near the top.

Measurement only (in-process over retrieve_channels, like mmr_probe — NOT the product path).
Reports R@1 / MRR / Complete@k per slice across the sweep, plus the per-question gold rank for the
questions that moved, so the sweet-spot lam is picked from BOTH the wins and the regressions.

    python -m tools.eval.fusion_probe prod --store-dir <snapshot>/data --questions q.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from mnemo.application.fusion.fuser import Fuser  # noqa: F401  (kept for parity / not used)
from mnemo.application.retrieval import Retrieval
from mnemo.application.search_criteria import SearchCriteria

from tools.eval import core
from tools.eval.gate_calibrate import locomo_queries, prod_queries

RRF_K = 60
# lam = weight on the DENSE leg. 0.5 == plain RRF (equal), 1.0 == dense-primary (lexical→0).
LAMBDAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def wrrf_order(dense_ids: list[str], lexical_ids: list[str], lam: float, k: int = RRF_K) -> list[str]:
    """Weighted reciprocal-rank fusion → ids best-first. Ties keep union order (dense first), so
    lam=1.0 leaves dense in its own order with lexical-only ids tailing."""
    rd = {mid: i for i, mid in enumerate(dense_ids)}
    rl = {mid: i for i, mid in enumerate(lexical_ids)}
    union = list(dict.fromkeys([*dense_ids, *lexical_ids]))

    def score(mid: str) -> float:
        s = 0.0
        if mid in rd:
            s += lam / (k + rd[mid])
        if mid in rl:
            s += (1.0 - lam) / (k + rl[mid])
        return s

    return sorted(union, key=score, reverse=True)


def _first_gold(order_keys: list[set], gold: list[str]) -> int | None:
    ranks = [p for p, ks in enumerate(order_keys) for g in gold if g in ks]
    return min(ranks) if ranks else None


def collect(container, queries, item_keys, limit: int) -> list[dict]:
    rows = []
    for qi, q in enumerate(queries):
        retrieval = Retrieval(
            criteria=SearchCriteria(scope="project", project=q.project),
            limit=limit, text=q.question, vector=container.embedder.encode(q.question),
        )
        channels = container.repository.retrieve_channels(retrieval)
        dense_ids = [h.memory.id for h in channels.dense]
        lexical_ids = [h.memory.id for h in channels.lexical]
        key_by_id = {h.memory.id: item_keys(h.memory)
                     for h in (*channels.dense, *channels.lexical)}
        ranks = {}
        for lam in LAMBDAS:
            order = wrrf_order(dense_ids, lexical_ids, lam)[:limit]
            ranks[lam] = _first_gold([key_by_id[mid] for mid in order], q.gold)
        rows.append({"id": q.id, "slice": q.slice, "ranks": ranks})
        if (qi + 1) % 25 == 0:
            print(f"  scored {qi + 1}/{len(queries)}", flush=True)
    return rows


def report(rows: list[dict]) -> None:
    labels = [f"λ{l:g}" for l in LAMBDAS]
    note = {0.5: " =RRF", 1.0: " =dense"}
    by_slice: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_slice[r["slice"]].append(r)

    def r_at(rs, lam, k):
        return sum(1 for r in rs if (v := r["ranks"][lam]) is not None and v < k) / len(rs)

    def mrr(rs, lam):
        return sum(0.0 if (v := r["ranks"][lam]) is None else 1.0 / (v + 1) for r in rs) / len(rs)

    bar = "=" * (18 + 9 * len(LAMBDAS))
    print(f"\n{bar}\nWeighted-RRF sweep — λ = dense weight (0.5=RRF … 1.0=dense-primary)\n{bar}")
    print("  " + "".join(f"{l:g}{note.get(l,''):>6}" if note.get(l) else f"λ{l:g}".rjust(9)
                          for l in LAMBDAS))
    for sl in sorted(by_slice):
        rs = by_slice[sl]
        print(f"\n{sl}  (n={len(rs)})")
        print(f"  {'R@1':14}" + "".join(f"{r_at(rs, l, 1):>9.3f}" for l in LAMBDAS))
        print(f"  {'R@3':14}" + "".join(f"{r_at(rs, l, 3):>9.3f}" for l in LAMBDAS))
        print(f"  {'MRR':14}" + "".join(f"{mrr(rs, l):>9.3f}" for l in LAMBDAS))

    # Overall R@1/MRR across everything with gold — the single number to balance.
    allr = rows
    print(f"\nOVERALL  (n={len(allr)})")
    print(f"  {'R@1':14}" + "".join(f"{r_at(allr, l, 1):>9.3f}" for l in LAMBDAS))
    print(f"  {'MRR':14}" + "".join(f"{mrr(allr, l):>9.3f}" for l in LAMBDAS))

    # Per-question gold rank ONLY for the questions that move across the sweep — the sweet-spot lam
    # is the one where exact-term regressions recover while semantic/multi-hop wins hold.
    print(f"\nMOVED questions (gold rank; '·'=miss)  [{'  '.join(labels)}]")
    for r in sorted(rows, key=lambda r: (r["slice"], r["id"])):
        vals = [r["ranks"][l] for l in LAMBDAS]
        cells = ["·" if v is None else str(v + 1) for v in vals]
        if len(set(cells)) > 1:  # only rows whose rank changes across lam
            print(f"  [{r['slice']:10}] {r['id']:24} " + "".join(f"{c:>9}" for c in cells))
    print(bar)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["prod", "locomo"])
    p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--questions", type=Path)
    p.add_argument("--data", type=Path, default=core.DEFAULT_DATA)
    p.add_argument("--conversations", type=int)
    p.add_argument("--max-queries", type=int)
    p.add_argument("--embedder", default="pplx")
    p.add_argument("--limit", type=int, default=10, help="page size scored (matches the eval's topk)")
    p.add_argument("--models-dir", default=os.path.expanduser("~/.mnemo/models"))
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    container = core.isolated_container(args.store_dir, args.embedder, args.models_dir)
    if args.mode == "prod":
        if not args.questions:
            raise SystemExit("prod mode needs --questions")
        queries = prod_queries(json.loads(args.questions.read_text())["questions"])
        def item_keys(mem):
            return {mem.topic_key} if mem.topic_key else set()
    else:
        id_to_dia = core.load_manifest(args.store_dir, args.embedder)
        queries = locomo_queries(core.load_dataset(args.data), args.conversations, args.max_queries)
        def item_keys(mem):
            return id_to_dia.get(mem.id, set())
    if args.max_queries:
        queries = queries[:args.max_queries]

    print(f"mode={args.mode}  store={args.store_dir}  embedder={args.embedder}  "
          f"limit={args.limit}  queries={len(queries)}")
    rows = collect(container, queries, item_keys, args.limit)
    report(rows)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
