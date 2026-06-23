"""Pure metrics (NO mnemo import) — so isolated-venv scorers (e.g. MLX) can reuse them.

- Bucket: retrieval Recall@k / AnyEvidence@k / CompleteEvidence@k / MRR.
- Tally: reranker top-1 A/B (baseline vs reranked hit@1, win/loss).
- score_candidates / report_ab: run + print a top-1 A/B over dumped candidates given a
  rank_fn(query, docs)->scores. Every reranker backend (in-process or out-of-process) plugs
  in by providing that one function, so they are all scored identically and fairly.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Bucket:
    """found-one (Any) vs found-all (Complete) splits multi-hop into a coverage problem vs a
    relevance problem; fractional Recall sits between them."""

    n: int = 0
    recall_at: dict = field(default_factory=lambda: defaultdict(float))
    any_at: dict = field(default_factory=lambda: defaultdict(float))
    complete_at: dict = field(default_factory=lambda: defaultdict(float))
    rr_sum: float = 0.0

    def add(self, ranked_dia: list[set[str]], evidence: set[str], k_list: list[int]) -> None:
        self.n += 1
        for k in k_list:
            retrieved: set[str] = set().union(*ranked_dia[:k]) if ranked_dia[:k] else set()
            inter = retrieved & evidence
            self.recall_at[k] += len(inter) / len(evidence)
            self.any_at[k] += 1.0 if inter else 0.0
            self.complete_at[k] += 1.0 if evidence <= retrieved else 0.0
        for rank, dset in enumerate(ranked_dia):
            if dset & evidence:
                self.rr_sum += 1.0 / (rank + 1)
                break

    def summary(self, k_list: list[int]) -> dict:
        if not self.n:
            return {"n": 0}
        return {
            "n": self.n,
            "recall_at_k": {k: round(self.recall_at[k] / self.n, 4) for k in k_list},
            "any_at_k": {k: round(self.any_at[k] / self.n, 4) for k in k_list},
            "complete_at_k": {k: round(self.complete_at[k] / self.n, 4) for k in k_list},
            "mrr": round(self.rr_sum / self.n, 4),
        }


@dataclass
class Tally:
    n: int = 0
    base_hits: int = 0
    rer_hits: int = 0
    changed: int = 0
    wins: int = 0
    losses: int = 0

    def add(self, base_hit: bool, rer_hit: bool, changed: bool) -> None:
        self.n += 1
        self.base_hits += base_hit
        self.rer_hits += rer_hit
        self.changed += changed
        self.wins += rer_hit and not base_hit
        self.losses += base_hit and not rer_hit


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def abstention_curve(positives: list[float], negatives: list[float]) -> dict:
    if not positives or not negatives:
        return {"note": "need both answerable and adversarial queries"}
    grid = [round(0.05 * i, 2) for i in range(21)]
    return {
        "n_answerable": len(positives), "n_adversarial": len(negatives),
        "mean_cosine_answerable": round(sum(positives) / len(positives), 4),
        "mean_cosine_adversarial": round(sum(negatives) / len(negatives), 4),
        "curve": [{
            "T": t,
            "false_refusal_on_answerable": round(sum(p < t for p in positives) / len(positives), 4),
            "true_refusal_on_adversarial": round(sum(n < t for n in negatives) / len(negatives), 4),
        } for t in grid],
    }


def score_candidates(candidates: list[dict], rank_fn) -> tuple:
    """Top-1 A/B over dumped candidates. rank_fn(query, docs) -> scores aligned to docs.
    Baseline top-1 = candidates[0] (RRF order); reranked top-1 = argmax score. Returns
    (overall Tally, {category: Tally}, single_hop Tally, ms_per_query)."""
    overall, single = Tally(), Tally()
    per_cat: dict = {}
    secs = 0.0
    scored = 0
    for i, qd in enumerate(candidates, 1):
        if i % 100 == 0:
            print(f"  scored {i}/{len(candidates)}", flush=True)
        ev = set(qd["evidence"]); cs = qd["candidates"]
        if not cs:
            continue
        t = time.monotonic()
        scores = rank_fn(qd["question"], [c["content"] for c in cs])
        secs += time.monotonic() - t
        scored += 1
        top = max(range(len(scores)), key=lambda j: scores[j])
        b = bool(set(cs[0]["dia"]) & ev); r = bool(set(cs[top]["dia"]) & ev)
        overall.add(b, r, top != 0)
        per_cat.setdefault(qd["category"], Tally()).add(b, r, top != 0)
        if len(ev) == 1:
            single.add(b, r, top != 0)
    return overall, per_cat, single, secs / max(scored, 1) * 1000


def report_ab(title: str, overall: Tally, per_cat: dict, single: Tally, ms: float, labels=None) -> dict:
    bar = "=" * 60
    print(f"\n{bar}\n{title} — top-1 A/B\n{bar}")
    print(f"questions={overall.n}  rerank={ms:.0f}ms/query")
    print("\nslice                 n   base@1   rerank@1    delta")
    print("-" * 52)

    def line(name, t: Tally):
        if t.n:
            print(f"{name:<20}{t.n:>5}  {t.base_hits/t.n:.4f}   {t.rer_hits/t.n:.4f}  "
                  f"{(t.rer_hits-t.base_hits)/t.n:+.4f}")

    line("ANSWERABLE", overall)
    for cat in sorted(per_cat):
        label = (labels or {}).get(cat, str(cat))
        line(f"  {cat} {label}", per_cat[cat])
    line("single-hop (|ev|=1)", single)
    print(f"\nwins={overall.wins} losses={overall.losses} changed_top1={overall.changed/max(overall.n,1):.3f}")
    print(bar)
    return {"n": overall.n, "base@1": round(overall.base_hits/max(overall.n,1), 4),
            "rerank@1": round(overall.rer_hits/max(overall.n,1), 4),
            "delta": round((overall.rer_hits-overall.base_hits)/max(overall.n,1), 4),
            "wins": overall.wins, "losses": overall.losses, "ms_per_query": round(ms, 1),
            "single_hop": {"n": single.n, "base@1": round(single.base_hits/max(single.n,1), 4),
                           "rerank@1": round(single.rer_hits/max(single.n,1), 4)}}
