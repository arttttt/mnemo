#!/usr/bin/env python3
"""MMR coverage probe — does Maximal Marginal Relevance diversification surface BOTH hops of a
multi-hop query into the top-k, where a pure-relevance order crowds them out with near-duplicate
hop-1 memories? The cross-encoder reranker REORDERS by relevance and can't fix this (it has no
diversity term, [[bench/rerank-calibration-result]]); MMR trades a little relevance for coverage.

Prototype/measurement ONLY (like gate_calibrate, NOT the product path): reorder the FUSED pool by
MMR and report Complete@k (multi-hop COVERAGE — all golds in top-k) plus R@1 (single-fact, to catch
the regression a diversity penalty risks), off (the RRF order) vs a lambda sweep. Pool vectors are
recomputed via the embedder and cached by id, so it needs NO store change — if MMR earns its place
here, THEN it gets wired into the search pipeline reading STORED vectors.

MMR relevance = cosine(query, doc) (uniform over dense+FTS pool items); diversity = max cosine to an
already-selected doc. lambda=1.0 is pure cosine relevance (no diversity); lambda<1 adds coverage.

    python -m tools.eval.mmr_probe locomo --store-dir tools/.locomo_store --conversations 2
    python -m tools.eval.mmr_probe prod --store-dir <snapshot>/data --questions q.json
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
from tools.eval.gate_calibrate import locomo_queries, prod_queries

LAMBDAS = [1.0, 0.9, 0.7, 0.5, 0.3]  # 1.0 = pure relevance; lower = more diversity


def mmr_order(rel: list[float], vecs: list, lam: float, k: int) -> list[int]:
    """Greedy MMR selection over the candidate indices: pick the k that maximise
    lam*relevance - (1-lam)*max-similarity-to-already-picked, then append the rest in the
    incoming (RRF) order so positions past k stay defined for scoring."""
    n = len(rel)
    k = min(k, n)
    selected: list[int] = []
    remaining = list(range(n))
    while remaining and len(selected) < k:
        best_i, best = remaining[0], float("-inf")
        for i in remaining:
            div = max((core.cosine(vecs[i], vecs[j]) for j in selected), default=0.0)
            score = lam * rel[i] - (1 - lam) * div
            if score > best:
                best, best_i = score, i
        selected.append(best_i)
        remaining.remove(best_i)
    return selected + remaining


def _rank_of(order_keys: list[set], gk: str) -> int | None:
    return next((p for p, ks in enumerate(order_keys) if gk in ks), None)


def collect(container, fuser, queries, item_keys, pool: int, top: int) -> list[dict]:
    cache: dict[str, list] = {}  # memory.id -> vector; pool items recur across a project's queries

    def vec(mem):
        v = cache.get(mem.id)
        if v is None:
            v = container.embedder.encode(mem.content)
            cache[mem.id] = v
        return v

    rows = []
    for qi, q in enumerate(queries):
        qv = container.embedder.encode(q.question)
        retrieval = Retrieval(
            criteria=SearchCriteria(scope="project", project=q.project),
            limit=pool, text=q.question, vector=qv,
        )
        items = fuser.fuse(container.repository.retrieve_channels(retrieval), pool).pool
        if not items:
            continue
        keys = [item_keys(h.memory) for h in items]
        vecs = [vec(h.memory) for h in items]
        rel = [core.cosine(qv, v) for v in vecs]
        orders = {"off": list(range(len(items)))}
        for lam in LAMBDAS:
            orders[lam] = mmr_order(rel, vecs, lam, top)
        ranks = {name: {gk: _rank_of([keys[i] for i in order], gk) for gk in q.gold}
                 for name, order in orders.items()}
        rows.append({"slice": q.slice, "n_gold": len(q.gold), "ranks": ranks})
        if (qi + 1) % 25 == 0:
            print(f"  scored {qi + 1}/{len(queries)}  (vec cache {len(cache)})", flush=True)
    return rows


def report(rows: list[dict]) -> None:
    names = ["off"] + LAMBDAS
    labels = ["off(RRF)"] + [f"λ{l:g}" for l in LAMBDAS]
    by_slice: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_slice[r["slice"]].append(r)

    def _first(ranks: dict) -> int | None:
        present = [p for p in ranks.values() if p is not None]
        return min(present) if present else None

    def r1(rs, name):
        return sum(1 for r in rs if _first(r["ranks"][name]) == 0) / len(rs)

    def complete(rs, name, k):
        return sum(1 for r in rs
                   if all(p is not None and p < k for p in r["ranks"][name].values())) / len(rs)

    bar = "=" * (20 + 9 * len(names))
    print(f"\n{bar}\nMMR coverage probe — off (RRF) vs lambda sweep  (n={len(rows)})\n{bar}")
    print("  diversity helps COVERAGE: watch C@3/C@5 on multi-hop rise without R@1 on single-hop falling")
    head = f"  {'slice/metric':18}" + "".join(f"{lab:>9}" for lab in labels)
    for sl in sorted(by_slice):
        rs = by_slice[sl]
        multi = sum(r["n_gold"] > 1 for rs_ in [rs] for r in rs_)
        print(f"\n{sl}  (n={len(rs)}, multi-gold={multi})")
        print(head)
        print(f"  {'R@1':18}" + "".join(f"{r1(rs, nm):>9.3f}" for nm in names))
        print(f"  {'Complete@3':18}" + "".join(f"{complete(rs, nm, 3):>9.3f}" for nm in names))
        print(f"  {'Complete@5':18}" + "".join(f"{complete(rs, nm, 5):>9.3f}" for nm in names))
    print(bar)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["locomo", "prod"])
    p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--questions", type=Path)
    p.add_argument("--data", type=Path, default=core.DEFAULT_DATA)
    p.add_argument("--conversations", type=int)
    p.add_argument("--max-queries", type=int)
    p.add_argument("--embedder", default="pplx")
    p.add_argument("--pool", type=int, default=50)
    p.add_argument("--top", type=int, default=10, help="MMR selects this many before falling back to RRF order")
    p.add_argument("--models-dir", default=os.path.expanduser("~/.mnemo/models"))
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    container = core.isolated_container(args.store_dir, args.embedder, args.models_dir)
    fuser = Fuser()

    if args.mode == "locomo":
        id_to_dia = core.load_manifest(args.store_dir, args.embedder)
        queries = locomo_queries(core.load_dataset(args.data), args.conversations, args.max_queries)
        def item_keys(mem):
            return id_to_dia.get(mem.id, set())
    else:
        if not args.questions:
            raise SystemExit("prod mode needs --questions")
        queries = prod_queries(json.loads(args.questions.read_text())["questions"])
        if args.max_queries:
            queries = queries[:args.max_queries]
        def item_keys(mem):
            return {mem.topic_key} if mem.topic_key else set()

    print(f"mode={args.mode}  store={args.store_dir}  embedder={args.embedder}  "
          f"pool={args.pool}  top={args.top}  queries={len(queries)}")
    rows = collect(container, fuser, queries, item_keys, args.pool, args.top)
    report(rows)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
