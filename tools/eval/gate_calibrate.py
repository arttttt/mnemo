#!/usr/bin/env python3
"""Rerank-gate SIGNAL calibration — which retrieval signal separates "bge helps" from
"bge hurts", so the gate can fire selectively instead of ~always.

Companion to rerank_calibrate.py (which it does NOT replace): that tool logs the 4 product
signals on the prod-snapshot/topic_key set; THIS tool adds (1) a LoCoMo dia_id-gold mode —
the prod set has too few help/hurt rows (~4/2) to fit a gate, LoCoMo has many — (2) a much
wider candidate-signal pool computed from the RAW channels (kept OUT of the product
RetrievalSignals until calibration proves which signals matter), and (3) a help-vs-hurt
SEPARABILITY table (per-signal AUC + best threshold) — the actual question the gate needs
answered. Both modes share one collect/report so a signal can be fit on LoCoMo and its
transfer checked on the prod set apples-to-apples.

For each scorable query: build the fused pool + the raw legs, score the gold's rank with NO
rerank vs bge reranking the WHOLE pool (gate OFF — decouples bge's effect from any gate), label
help/hurt, and log the full signal pool. In-process; a research tool, NOT the product path.

    # fit: which signal separates help from hurt (needs the ingested LoCoMo store)
    python -m tools.eval.gate_calibrate locomo --store-dir tools/.locomo_store --conversations 4
    # validate transfer on our own domain
    python -m tools.eval.gate_calibrate prod --store-dir <snapshot>/data --questions q.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from mnemo.application.fusion.fuser import Fuser
from mnemo.application.retrieval import Retrieval
from mnemo.application.search_criteria import SearchCriteria

from tools.eval import core
from tools.eval.models import build_reranker


@dataclass(frozen=True)
class Query:
    id: str
    slice: str
    project: str
    question: str
    gold: list[str]  # topic_keys (prod) or dia_ids (locomo); a pool item is gold if it carries one


# ---- candidate signal pool (computed from the RAW channels, research-only) -----------------

# Ordered so the report reads dense-leg → lexical-leg → cross-leg → counts → query shape.
SIGNAL_KEYS = [
    "dense_top1", "dense_top2", "dense_margin", "dense_mean", "dense_std", "dense_range",
    "dense_gap13", "dense_entropy", "dense_top1_z",
    "bm25_top1", "bm25_margin",
    "agree", "overlap_M", "rank_overlap5", "rank_overlap10", "jaccard10", "spearman",
    "n_dense", "n_lexical", "q_words", "q_chars",
]


def _stats(xs: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(xs) / n
    std = (sum((x - mean) ** 2 for x in xs) / n) ** 0.5
    return mean, std, max(xs) - min(xs)


def _softmax_entropy(xs: list[float]) -> float:
    """Normalised (0..1) entropy of softmax(scores) — high = the dense leg is undecided
    (flat over many candidates), low = one hit dominates."""
    if len(xs) < 2:
        return 0.0
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    ps = [e / s for e in exps]
    ent = -sum(p * math.log(p) for p in ps if p > 0)
    return ent / math.log(len(xs))


def _overlap_at(a_ids: list[str], b_ids: list[str], k: int) -> float:
    sa, sb = set(a_ids[:k]), set(b_ids[:k])
    return len(sa & sb) / k if k else 0.0


def _jaccard_at(a_ids: list[str], b_ids: list[str], k: int) -> float:
    sa, sb = set(a_ids[:k]), set(b_ids[:k])
    u = sa | sb
    return len(sa & sb) / len(u) if u else 0.0


def _spearman(a_ids: list[str], b_ids: list[str]) -> float:
    """Rank correlation of the two legs over the ids they SHARE (0.0 if <2 shared)."""
    ra = {mid: i for i, mid in enumerate(a_ids)}
    rb = {mid: i for i, mid in enumerate(b_ids)}
    shared = [mid for mid in ra if mid in rb]
    n = len(shared)
    if n < 2:
        return 0.0
    d2 = sum((ra[mid] - rb[mid]) ** 2 for mid in shared)
    return 1 - 6 * d2 / (n * (n * n - 1))


def signal_pool(channels, signals, question: str) -> dict[str, float]:
    d = [sm.score for sm in channels.dense]
    b = [sm.score for sm in channels.lexical]
    did = [sm.memory.id for sm in channels.dense]
    bid = [sm.memory.id for sm in channels.lexical]
    dmean, dstd, drange = _stats(d)
    d0, d1, d2 = (d + [0.0, 0.0, 0.0])[:3]
    b0, b1 = (b + [0.0, 0.0])[:2]
    return {
        "dense_top1": d0, "dense_top2": d1, "dense_margin": d0 - d1,
        "dense_mean": dmean, "dense_std": dstd, "dense_range": drange,
        "dense_gap13": d0 - d2, "dense_entropy": _softmax_entropy(d),
        "dense_top1_z": (d0 - dmean) / dstd if dstd else 0.0,
        "bm25_top1": b0, "bm25_margin": b0 - b1,
        "agree": 1.0 if signals.agree else 0.0,
        "overlap_M": signals.overlap,
        "rank_overlap5": _overlap_at(did, bid, 5),
        "rank_overlap10": _overlap_at(did, bid, 10),
        "jaccard10": _jaccard_at(did, bid, 10),
        "spearman": _spearman(did, bid),
        "n_dense": float(signals.n_dense), "n_lexical": float(signals.n_lexical),
        "q_words": float(len(question.split())), "q_chars": float(len(question)),
    }


# ---- scoring -------------------------------------------------------------------------------

def _effect(off: int | None, bge: int | None) -> str:
    if off is None or bge is None:
        return "na"
    if bge < off:
        return "help"
    if bge > off:
        return "hurt"
    return "same"


def collect(container, fuser, reranker, queries: list[Query], item_keys, pool: int) -> list[dict]:
    rows = []
    for qi, q in enumerate(queries):
        retrieval = Retrieval(
            criteria=SearchCriteria(scope="project", project=q.project),
            limit=pool, text=q.question, vector=container.embedder.encode(q.question),
        )
        channels = container.repository.retrieve_channels(retrieval)
        fused = fuser.fuse(channels, pool)
        items = fused.pool
        if not items:
            continue
        off_keys = [item_keys(h.memory) for h in items]
        scores = reranker.rank(q.question, [h.memory.content for h in items])
        order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
        bge_keys = [off_keys[i] for i in order]

        def _rank_of(keysets: list[set], gk: str) -> int | None:
            return next((i for i, ks in enumerate(keysets) if gk in ks), None)

        # Per-gold (per-hop) ranks: None = that gold is OUTSIDE the pool (a retrieval miss
        # reranking can't fix), not a rerank problem.
        golds = [{"key": gk, "off": _rank_of(off_keys, gk), "bge": _rank_of(bge_keys, gk)}
                 for gk in q.gold]
        off_present = [g["off"] for g in golds if g["off"] is not None]
        bge_present = [g["bge"] for g in golds if g["bge"] is not None]
        off_rank = min(off_present) if off_present else None  # rank of the FIRST gold reached
        bge_rank = min(bge_present) if bge_present else None
        rows.append({
            "id": q.id, "slice": q.slice, "off_rank": off_rank, "bge_rank": bge_rank,
            "golds": golds, "effect": _effect(off_rank, bge_rank),
            "sig": signal_pool(channels, fused.signals, q.question),
        })
        if (qi + 1) % 25 == 0:
            print(f"  scored {qi + 1}/{len(queries)}", flush=True)
    return rows


# ---- separability analysis -----------------------------------------------------------------

def _auc(helps: list[float], hurts: list[float]) -> float:
    """P(a random help row scores higher than a random hurt row) — Mann-Whitney. 0.5 = no
    separation; far from 0.5 (either way) = the signal discriminates help from hurt."""
    if not helps or not hurts:
        return 0.5
    wins = sum((1.0 if h > u else 0.5 if h == u else 0.0) for h in helps for u in hurts)
    return wins / (len(helps) * len(hurts))


def _best_threshold(helps: list[float], hurts: list[float]) -> tuple[float, float, str]:
    """Best single cut: returns (threshold, balanced_accuracy, direction). direction '>=' means
    classify as help when value>=T. Balanced acc = 0.5*(help recall + hurt rejection)."""
    vals = sorted(set(helps + hurts))
    cands = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)] or vals
    nh, nu = len(helps) or 1, len(hurts) or 1
    best = (0.0, vals[0] if vals else 0.0, ">=")
    for t in cands:
        for d in (">=", "<"):
            if d == ">=":
                tpr = sum(1 for x in helps if x >= t) / nh
                tnr = sum(1 for x in hurts if x < t) / nu
            else:
                tpr = sum(1 for x in helps if x < t) / nh
                tnr = sum(1 for x in hurts if x >= t) / nu
            bal = 0.5 * (tpr + tnr)
            if bal > best[0]:
                best = (bal, t, d)
    return best[1], best[0], best[2]


def _gate_fires(row: dict, floor: float) -> bool:
    """The current product gate shape: rerank when the top dense hit is weak OR legs disagree."""
    return row["sig"]["dense_top1"] < floor or row["sig"]["agree"] < 0.5


def report(rows: list[dict]) -> None:
    bar = "=" * 96
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["effect"]] += 1
    print(f"\n{bar}\nRerank-gate signal calibration — bge over the whole pool (gate OFF)\n{bar}")
    print(f"n={len(rows)}  help={counts['help']} hurt={counts['hurt']} "
          f"same={counts['same']} na={counts['na']}  (na = a gold never reached the pool)")

    # Per-slice effect on the metric each slice needs: R@1 + Complete@k (multi-hop coverage).
    print("\nPER-SLICE off→bge  (R@1 = first gold@1; C@3/C@5 = ALL golds in top-k)")
    by_slice: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_slice[r["slice"]].append(r)

    def _r1(rs, which):
        return sum(1 for r in rs if r[f"{which}_rank"] == 0) / len(rs)

    def _complete(rs, which, k):
        return sum(1 for r in rs
                   if all(g[which] is not None and g[which] < k for g in r["golds"])) / len(rs)

    for sl in sorted(by_slice):
        rs = by_slice[sl]
        print(f"  {sl:11} n={len(rs):3}  R@1 {_r1(rs, 'off'):.2f}→{_r1(rs, 'bge'):.2f}"
              f"   C@3 {_complete(rs, 'off', 3):.2f}→{_complete(rs, 'bge', 3):.2f}"
              f"   C@5 {_complete(rs, 'off', 5):.2f}→{_complete(rs, 'bge', 5):.2f}")

    # THE question: does any signal separate help from hurt? AUC + best single-threshold cut.
    helps = [r for r in rows if r["effect"] == "help"]
    hurts = [r for r in rows if r["effect"] == "hurt"]
    print(f"\nSEPARABILITY  help={len(helps)} vs hurt={len(hurts)}  "
          f"(AUC=P(signal higher on HELP); |AUC-.5| large ⇒ separating; balAcc=best single cut)")
    if helps and hurts:
        print(f"  {'signal':14}{'help_mean':>11}{'hurt_mean':>11}{'AUC':>7}{'cut':>11}{'balAcc':>8}")
        table = []
        for key in SIGNAL_KEYS:
            hv = [r["sig"][key] for r in helps]
            uv = [r["sig"][key] for r in hurts]
            auc = _auc(hv, uv)
            t, bal, d = _best_threshold(hv, uv)
            table.append((abs(auc - 0.5), key, sum(hv) / len(hv), sum(uv) / len(uv), auc, d, t, bal))
        for _, key, mh, mu, auc, d, t, bal in sorted(table, reverse=True):
            print(f"  {key:14}{mh:>11.3f}{mu:>11.3f}{auc:>7.2f}{f'{d}{t:.3f}':>11}{bal:>8.2f}")
    else:
        print("  (need both help AND hurt rows — widen --conversations / the question set)")

    # The current product gate over a floor sweep — want: fire on help, skip hurt.
    print("\nCURRENT GATE SWEEP  (rerank if dense_top1<floor OR disagree)")
    print(f"  {'floor':>6}{'help_fired':>12}{'hurt_fired':>12}{'net(h-u)':>10}")
    for floor in (0.0, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 1.01):
        hf = sum(_gate_fires(r, floor) for r in helps)
        uf = sum(_gate_fires(r, floor) for r in hurts)
        print(f"  {floor:>6.2f}{hf:>7}/{len(helps):<4}{uf:>7}/{len(hurts):<4}{hf - uf:>10}")
    print(bar)


# ---- query loaders (the only mode-specific code) -------------------------------------------

def locomo_queries(data: list[dict], conversations: int | None, max_queries: int | None) -> list[Query]:
    out: list[Query] = []
    convs = data if conversations is None else data[:conversations]
    for sample in convs:
        slug = sample["sample_id"]
        for i, qa in enumerate(sample["qa"]):
            gold = [str(e) for e in (qa.get("evidence") or [])]
            if not gold:  # adversarial / unscorable — no gold to rank
                continue
            out.append(Query(
                id=f"{slug[:10]}#{i}", slice=core.CATEGORY_LABELS.get(qa.get("category"), "?"),
                project=slug, question=qa["question"], gold=gold,
            ))
    return out[:max_queries] if max_queries else out


def prod_queries(questions: list[dict]) -> list[Query]:
    return [Query(id=q["id"], slice=q["slice"], project=q["project"], question=q["question"],
                  gold=list(q.get("gold_keys") or []))
            for q in questions if q.get("gold_keys")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["locomo", "prod"])
    p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--questions", type=Path, help="prod mode: the questions json")
    p.add_argument("--data", type=Path, default=core.DEFAULT_DATA, help="locomo mode: the dataset")
    p.add_argument("--conversations", type=int, help="locomo mode: cap conversations")
    p.add_argument("--max-queries", type=int, help="cap scored queries (speed)")
    p.add_argument("--embedder", default="pplx")
    p.add_argument("--reranker", default="bge")
    p.add_argument("--pool", type=int, default=50)
    p.add_argument("--models-dir", default=os.path.expanduser("~/.mnemo/models"))
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    container = core.isolated_container(args.store_dir, args.embedder, args.models_dir)
    fuser = Fuser()
    reranker = build_reranker(args.reranker, args.models_dir)

    if args.mode == "locomo":
        id_to_dia = core.load_manifest(args.store_dir, args.embedder)
        queries = locomo_queries(core.load_dataset(args.data), args.conversations, args.max_queries)
        def item_keys(mem):
            return id_to_dia.get(mem.id, set())
    else:
        if not args.questions:
            raise SystemExit("prod mode needs --questions")
        questions = json.loads(args.questions.read_text())["questions"]
        queries = prod_queries(questions)
        if args.max_queries:
            queries = queries[:args.max_queries]
        def item_keys(mem):
            return {mem.topic_key} if mem.topic_key else set()

    print(f"mode={args.mode}  store={args.store_dir}  embedder={args.embedder}  "
          f"reranker={args.reranker}  pool={args.pool}  queries={len(queries)}")
    rows = collect(container, fuser, reranker, queries, item_keys, args.pool)
    report(rows)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
