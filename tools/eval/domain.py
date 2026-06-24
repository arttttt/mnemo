#!/usr/bin/env python3
"""Project-fact domain eval (п3) — mnemo's go/no-go readout on its REAL job.

LoCoMo (eval.locomo) only validates the harness against a conversational standard. This eval
measures mnemo on project-fact memory: it ingests the self-contained fixture into an ISOLATED
store, runs the LLM-free search path per question, and scores by slice:
  - answerable / superseded: Recall@k / MRR + Any/Complete against the gold memory keys (the
    shared Bucket); superseded ALSO checks the stale version does not out-rank the current one.
  - irrelevant (REFUSE): the abstention readout — the two candidate input-gate signals, raw dense
    cosine AND lexical corroboration, on answerable vs irrelevant, so the refusal threshold is a
    CURVE, not one number (the LoCoMo run showed cosine alone can't separate on-topic traps).
Tier-1 (no generator). Reuses tools.eval.core (isolated harness) + tools.eval.metrics. Run:
    python -m tools.eval.domain --embedder pplx
    python -m tools.eval.domain --embedder hash         # machinery smoke (no model)
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from mnemo.application.use_cases.create_project import ProjectAlreadyExists

from tools.eval import core
from tools.eval.domain_fixture import Fixture, load_fixture

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "domain_v1.json"


def ingest(container, fixture: Fixture) -> dict[str, str]:
    """Ingest the corpus in array order (a reused topic_key supersedes its predecessor). Returns
    the store-id -> fixture-key bridge used to score search hits against gold keys."""
    for slug in fixture.projects:
        try:
            container.create_project.execute(slug)
        except ProjectAlreadyExists:
            pass
    key_by_id: dict[str, str] = {}
    for memory in fixture.memories:
        result = container.remember.execute(
            content=memory.content, type=memory.type, scope=memory.scope, project=memory.project,
            tags=list(memory.tags), related_files=list(memory.related_files), topic_key=memory.topic_key,
        )
        key_by_id[result.id] = memory.key
    return key_by_id


def _ranked_keys(hits, key_by_id: dict[str, str]) -> list[str]:
    """Hits as their fixture keys, in rank order (every hit comes from our own ingest)."""
    return [key_by_id[h.id] for h in hits if h.id in key_by_id]


def _first_index(keys: list[str], targets: set[str]) -> int | None:
    return next((i for i, key in enumerate(keys) if key in targets), None)


def _corroboration_tradeoff(positives: list[int], negatives: list[int]) -> dict:
    def rate(xs, c):
        return round(sum(x < c for x in xs) / max(len(xs), 1), 3)

    return {
        "mean_answerable": round(sum(positives) / max(len(positives), 1), 2),
        "mean_irrelevant": round(sum(negatives) / max(len(negatives), 1), 2),
        "by_refuse_below": [
            {"refuse_if_below": c, "false_refuse_on_answerable": rate(positives, c),
             "true_refuse_on_irrelevant": rate(negatives, c)}
            for c in (1, 2, 3)
        ],
    }


def evaluate(container, fixture: Fixture, key_by_id, k_list, rerank=None) -> dict:
    """Score the fixture. `rerank(query, hits) -> reordered hits` is the optional prod reranker
    (bge); None = the pure search path. Returns a JSON-serializable report."""
    limit = max(k_list)
    answerable, superseded = core.Bucket(), core.Bucket()
    stale_in_topk = stale_at_or_above = n_sup = 0
    cos: dict = defaultdict(list)
    corr: dict = defaultdict(list)
    per_question = []

    for q in fixture.questions:
        hits = container.search.execute(
            query=q.question, scope="project", project=q.project, limit=limit
        )
        if rerank is not None and hits:
            hits = rerank(q.question, hits)
        keys = _ranked_keys(hits, key_by_id)
        gold = set(q.gold_keys)

        if hits:
            top = hits[0].content
            cos[q.slice].append(core.cosine(container.embedder.encode(q.question),
                                            container.embedder.encode(top)))
            corr[q.slice].append(core.corroboration(q.question, top))

        if q.slice == "answerable":
            answerable.add([{key} for key in keys], gold, k_list)
        elif q.slice == "superseded":
            superseded.add([{key} for key in keys], gold, k_list)
            n_sup += 1
            sp = _first_index(keys[:limit], set(q.stale_keys))
            gp = _first_index(keys[:limit], gold)
            if sp is not None:
                stale_in_topk += 1
                if gp is None or sp <= gp:
                    stale_at_or_above += 1
        per_question.append({"id": q.id, "slice": q.slice, "gold_rank": _first_index(keys, gold)})

    pos_cos = cos["answerable"] + cos["superseded"]
    pos_corr = corr["answerable"] + corr["superseded"]
    return {
        "answerable": answerable.summary(k_list),
        "superseded": {**superseded.summary(k_list), "n_superseded": n_sup,
                       "stale_in_topk": stale_in_topk, "stale_at_or_above_gold": stale_at_or_above},
        "abstention": {
            "cosine": core.abstention_curve(pos_cos, cos["irrelevant"]),
            "corroboration": _corroboration_tradeoff(pos_corr, corr["irrelevant"]),
        },
        "per_question": per_question,
    }


def print_report(report: dict, meta: dict, k_list: list[int]) -> None:
    bar = "=" * 72
    print(f"\n{bar}\nDomain eval (п3) — project-fact memory, search path only\n{bar}")
    print(f"embedder={meta['embedder']}  reranker={meta['reranker']}  "
          f"memories={meta['n_memories']}  questions={meta['n_questions']}")

    def retrieval(name, summary):
        if not summary.get("n"):
            return
        head = "  ".join(f"@{k}".rjust(6) for k in k_list)
        recall = "  ".join(f"{summary['recall_at_k'][k]:.3f}".rjust(6) for k in k_list)
        anyk = "  ".join(f"{summary['any_at_k'][k]:.3f}".rjust(6) for k in k_list)
        print(f"\n{name} (n={summary['n']})   MRR={summary['mrr']:.3f}")
        print("  k     " + head)
        print("  recall" + recall)
        print("  any   " + anyk)

    retrieval("ANSWERABLE", report["answerable"])
    sup = report["superseded"]
    retrieval("SUPERSEDED", sup)
    if sup.get("n_superseded"):
        print(f"  stale-in-top-{max(k_list)}: {sup['stale_in_topk']}/{sup['n_superseded']}  "
              f"stale-at-or-above-gold: {sup['stale_at_or_above_gold']}/{sup['n_superseded']}")

    print("\nABSTENTION  (refuse on irrelevant without false-refusing answerable)")
    cos = report["abstention"]["cosine"]
    if "curve" in cos:
        print(f"  raw cosine    mean: answerable={cos['mean_cosine_answerable']} "
              f"irrelevant={cos['mean_cosine_adversarial']}")
        for pt in cos["curve"]:
            if pt["T"] in (0.3, 0.4, 0.5, 0.6):
                print(f"    T={pt['T']}  false-refuse(ans)={pt['false_refusal_on_answerable']:.2f}  "
                      f"true-refuse(irr)={pt['true_refusal_on_adversarial']:.2f}")
    else:
        print(f"  raw cosine    {cos.get('note', 'n/a')}")
    cor = report["abstention"]["corroboration"]
    print(f"  corroboration mean: answerable={cor['mean_answerable']} irrelevant={cor['mean_irrelevant']}")
    for pt in cor["by_refuse_below"]:
        print(f"    refuse if <{pt['refuse_if_below']} terms  "
              f"false-refuse(ans)={pt['false_refuse_on_answerable']:.2f}  "
              f"true-refuse(irr)={pt['true_refuse_on_irrelevant']:.2f}")
    print(bar)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedder", default="pplx", choices=["pplx", "hash"])
    p.add_argument("--fixture", type=Path, default=_FIXTURE)
    p.add_argument("--store-dir", type=Path, default=None)
    p.add_argument("--models-dir", default=os.path.expanduser("~/.mnemo/models"))
    p.add_argument("--k", default="1,3,5,10")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    k_list = sorted({int(x) for x in args.k.split(",")})
    fixture = load_fixture(args.fixture)

    tmp = None
    if args.store_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="mnemo-domain-")
        store_dir = Path(tmp.name)
    else:
        store_dir = args.store_dir

    print(f"isolated store: {store_dir}  embedder={args.embedder}  fixture={args.fixture.name}")
    container = core.isolated_container(store_dir, args.embedder, args.models_dir)
    key_by_id = ingest(container, fixture)
    print(f"ingested {len(key_by_id)} memories; querying {len(fixture.questions)} questions…")

    report = evaluate(container, fixture, key_by_id, k_list)
    meta = {"embedder": args.embedder, "reranker": "off",
            "n_memories": len(fixture.memories), "n_questions": len(fixture.questions), "k": k_list}
    print_report(report, meta, k_list)

    core.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or core.RESULTS_DIR / f"domain_{args.embedder}.json"
    out.write_text(json.dumps({"meta": meta, "report": report}, indent=2))
    print(f"wrote {out}")
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
