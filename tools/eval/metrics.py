"""Pure metrics (NO mnemo import) — so isolated-venv scorers (e.g. MLX) can reuse them.

- Bucket: retrieval Recall@k / AnyEvidence@k / CompleteEvidence@k / MRR.
- Tally: a simple top-1 hit/win/loss accumulator.
- score_candidates / report_ab: run + print a reranker A/B over dumped candidates given a
  rank_fn(query, docs)->scores. Reports hit@k (gold in the top-k) for BOTH the baseline RRF
  order and the reranked order, at each k (default 1/5/10), plus @1 win/loss. Every reranker
  backend plugs in via that one function, so they are all scored identically and fairly.
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


def _hit_at_k(cands: list[dict], order: list[int], k: int, evidence: set[str]) -> bool:
    """Does the top-k of `order` (indices into cands) cover any gold dia id? hit@k."""
    retrieved: set[str] = set()
    for j in order[:k]:
        retrieved |= set(cands[j]["dia"])
    return bool(retrieved & evidence)


def scaled_pool(k: int, pool_min: int = 20, pool_cap: int = 50, factor: int = 5) -> int:
    """Over-fetch pool for return-limit k: scale with k so headroom (pool-k) stays adequate,
    floored at pool_min, capped at pool_cap. Default min 20 / cap 50 / factor 5 → @1=20,
    @5=25, @10=50."""
    return min(pool_cap, max(pool_min, factor * k))


def score_candidates(candidates: list[dict], rank_fn, k_list=(1, 5, 10),
                     pool_min: int = 20, pool_cap: int = 50, factor: int = 5) -> dict:
    """Reranker A/B over dumped candidates. rank_fn(query, docs) -> scores aligned to docs.
    For each return-limit k: baseline hit@k = gold in the RRF top-k; reranked hit@k = rerank a
    k-SCALED over-fetch pool (scaled_pool(k)) and take its top-k. One rank_fn pass scores the
    whole available pool; each k just slices+re-sorts it (no re-rerank). Also @1 win/loss."""
    k_list = list(k_list)
    pools = {k: scaled_pool(k, pool_min, pool_cap, factor) for k in k_list}
    n = sh_n = 0
    secs = 0.0
    base = {k: 0 for k in k_list}; rer = {k: 0 for k in k_list}
    shb = {k: 0 for k in k_list}; shr = {k: 0 for k in k_list}
    wins = losses = changed = 0
    for i, qd in enumerate(candidates, 1):
        if i % 100 == 0:
            print(f"  scored {i}/{len(candidates)}", flush=True)
        ev = set(qd["evidence"]); cs = qd["candidates"]
        if not cs:
            continue
        t = time.monotonic()
        scores = rank_fn(qd["question"], [c["content"] for c in cs])
        secs += time.monotonic() - t
        n += 1
        single = len(ev) == 1
        sh_n += single
        for k in k_list:
            pool = list(range(min(pools[k], len(cs))))                   # RRF top-pool(k)
            reranked = sorted(pool, key=lambda j: scores[j], reverse=True)
            bh = _hit_at_k(cs, pool, k, ev); rh = _hit_at_k(cs, reranked, k, ev)
            base[k] += bh; rer[k] += rh
            if single:
                shb[k] += bh; shr[k] += rh
            if k == 1:
                wins += rh and not bh; losses += bh and not rh; changed += reranked[0] != 0

    def rate(d, k, m):
        return round(d[k] / max(m, 1), 4)

    return {
        "n": n, "ms_per_query": round(secs / max(n, 1) * 1000, 1), "k_list": k_list, "pools": pools,
        "overall": {k: {"base": rate(base, k, n), "rerank": rate(rer, k, n)} for k in k_list},
        "single_hop": {"n": sh_n,
                       "at_k": {k: {"base": rate(shb, k, sh_n), "rerank": rate(shr, k, sh_n)} for k in k_list}},
        "win_loss_at1": {"wins": wins, "losses": losses, "changed": round(changed / max(n, 1), 3)},
    }


def report_ab(title: str, result: dict) -> dict:
    """Print the hit@k A/B (baseline RRF top-k vs reranked top-k of a k-scaled pool)."""
    bar = "=" * 60
    print(f"\n{bar}\n{title} — rerank hit@k A/B (pool scales with k)\n{bar}")
    print(f"questions={result['n']}  rerank={result['ms_per_query']:.0f}ms/query")
    print("\nslice          k  pool   base@k   rerank@k    delta")
    print("-" * 54)
    for k in result["k_list"]:
        o = result["overall"][k]; p = result["pools"][k]
        print(f"ANSWERABLE    {k:>2}  {p:>4}   {o['base']:.4f}   {o['rerank']:.4f}  {o['rerank']-o['base']:+.4f}")
    sh = result["single_hop"]["at_k"]
    for k in result["k_list"]:
        s = sh[k]; p = result["pools"][k]
        print(f"single-hop    {k:>2}  {p:>4}   {s['base']:.4f}   {s['rerank']:.4f}  {s['rerank']-s['base']:+.4f}")
    wl = result["win_loss_at1"]
    print(f"\n@1 wins={wl['wins']} losses={wl['losses']} changed_top1={wl['changed']}")
    print(bar)
    return result
