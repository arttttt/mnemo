#!/usr/bin/env python3
"""LongMemEval retrieval benchmark — mnemo SEARCH path only (LLM-free), turn granularity.

Runs the PUBLIC LongMemEval benchmark (Wu et al., ICLR 2025; HF xiaowu0162/longmemeval-cleaned)
through mnemo's real search so the numbers are directly comparable to peers that report it
(doobidoo/mcp-memory-service, Zep, mem0). Like locomo.py this validates the harness MACHINERY
against a public standard — LongMemEval is CONVERSATIONAL chat history, NOT mnemo's project-fact
domain, so read it as cross-system comparability, not a real-domain measure.

Per question: register a project (= question_id), ingest each chat TURN of its haystack as a memory
tagged session:<sid> (TURN granularity — a whole session exceeds the 512-token per-type cap), then
run `search(question)` scoped to that project and score SESSION-LEVEL recall@k / NDCG@k / MRR against
answer_session_ids (a gold session is "found" if ANY of its turns is retrieved in top-k). Abstention
(`_abs`) instances are SKIPPED for retrieval, per the benchmark's own convention (no GT answer
location). Turns over the per-type token cap are skipped and counted (a real mnemo limitation: it
stores short facts, not long chat turns).

    python -m tools.eval.longmemeval --data <longmemeval_s.json> --embedder pplx --subset 48
    python -m tools.eval.longmemeval --data <longmemeval_oracle.json> --embedder hash --subset 12  # smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from mnemo.application.use_cases.create_project import ProjectAlreadyExists

from tools.eval import core

_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(qid: str) -> str:
    return _SLUG.sub("-", qid.lower()).strip("-")[:60] or "q"


def _turn_text(date: str, role: str, content: str) -> str:
    """A chat turn as a stored memory — timestamp + role prefix (timestamps help temporal queries)."""
    return f"[{date}] {role}: {content}"


def stratified(data: list[dict], subset: int | None, types: set[str] | None) -> list[dict]:
    """Round-robin across question_type so a subset stays balanced. Abstention is dropped (the
    benchmark skips it for retrieval — no gold answer location)."""
    items = [d for d in data if not d["question_id"].endswith("_abs")]
    if types:
        items = [d for d in items if d["question_type"] in types]
    if subset is None or subset >= len(items):
        return items
    by_type: dict[str, list] = defaultdict(list)
    for d in items:
        by_type[d["question_type"]].append(d)
    out: list[dict] = []
    row = 0
    while len(out) < subset:
        progressed = False
        for qt in sorted(by_type):
            if row < len(by_type[qt]):
                out.append(by_type[qt][row])
                progressed = True
                if len(out) >= subset:
                    break
        if not progressed:
            break
        row += 1
    return out


def ndcg_at_k(session_rank: list[str], gold: set[str], k: int) -> float:
    """Binary-relevance NDCG@k over the session-level ranked list (unique sessions, first-hit order)."""
    dcg = sum(1.0 / math.log2(i + 2) for i, sid in enumerate(session_rank[:k]) if sid in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def run_question(container, item: dict, k_list: list[int]) -> tuple[set, list, list, int]:
    slug = _slug(item["question_id"])
    try:
        container.create_project.execute(slug)
    except ProjectAlreadyExists:
        pass
    id_to_sid: dict[str, str] = {}
    skipped = 0
    for sid, date, session in zip(
        item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]
    ):
        for turn in session:
            try:
                r = container.remember.execute(
                    content=_turn_text(date, turn.get("role", ""), turn.get("content", "")),
                    type="working-notes", scope="project", project=slug, tags=[f"session:{sid}"],
                )
            except ValueError:
                skipped += 1  # over the per-type token cap — mnemo stores short facts, not long turns
                continue
            id_to_sid[r.id] = sid
    gold = set(item["answer_session_ids"])
    hits = container.search.execute(
        query=item["question"], scope="project", project=slug, limit=max(k_list)
    )
    ranked_sids = [id_to_sid.get(h.id) for h in hits]
    seen: set[str] = set()
    session_rank: list[str] = []
    for s in ranked_sids:
        if s and s not in seen:
            seen.add(s)
            session_rank.append(s)
    return gold, ranked_sids, session_rank, skipped


def evaluate(container, items: list[dict], k_list: list[int]) -> dict:
    by_type: dict[str, core.Bucket] = defaultdict(core.Bucket)
    overall = core.Bucket()
    ndcg_type: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    ndcg_all: dict[int, list] = defaultdict(list)
    skipped_turns = 0
    for n, item in enumerate(items, 1):
        gold, ranked_sids, session_rank, skipped = run_question(container, item, k_list)
        skipped_turns += skipped
        ranked_sets = [{s} if s else set() for s in ranked_sids]
        qt = item["question_type"]
        by_type[qt].add(ranked_sets, gold, k_list)
        overall.add(ranked_sets, gold, k_list)
        for k in k_list:
            v = ndcg_at_k(session_rank, gold, k)
            ndcg_type[qt][k].append(v)
            ndcg_all[k].append(v)
        if n % 10 == 0:
            print(f"  scored {n}/{len(items)}", flush=True)

    def mean_ndcg(d):
        return {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in d.items()}

    return {
        "overall": {**overall.summary(k_list), "ndcg_at_k": mean_ndcg(ndcg_all)},
        "by_type": {qt: {**by_type[qt].summary(k_list), "ndcg_at_k": mean_ndcg(ndcg_type[qt])}
                    for qt in sorted(by_type)},
        "skipped_turns_over_cap": skipped_turns,
    }


def print_report(report: dict, meta: dict, k_list: list[int]) -> None:
    bar = "=" * 84
    print(f"\n{bar}\nLongMemEval retrieval (mnemo search, turn granularity, session-level recall)\n{bar}")
    print(f"embedder={meta['embedder']}  data={meta['data']}  questions={meta['n_questions']}  "
          f"skipped_turns_over_cap={report['skipped_turns_over_cap']}")
    header = "slice".ljust(28) + "n".rjust(4) + "  " + "  ".join(f"R@{k}".rjust(7) for k in k_list) + \
        "  " + "NDCG@10".rjust(8) + "   MRR".rjust(8)
    print("\n" + header + "\n" + "-" * len(header))

    def row(name, s):
        if not s.get("n"):
            return
        cells = "  ".join(f"{s['recall_at_k'][k]:.4f}".rjust(7) for k in k_list)
        ndcg10 = s["ndcg_at_k"].get(10, s["ndcg_at_k"].get(max(k_list), 0.0))
        print(name.ljust(28) + str(s["n"]).rjust(4) + "  " + cells +
              "  " + f"{ndcg10:.4f}".rjust(8) + f"{s['mrr']:.4f}".rjust(8))

    row("OVERALL", report["overall"])
    print()
    for qt, s in report["by_type"].items():
        row(f"  {qt}", s)
    print(bar)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="longmemeval_s.json / _oracle.json")
    p.add_argument("--embedder", default="pplx", choices=["pplx", "hash"])
    p.add_argument("--subset", type=int, default=None, help="stratified subset size (across question types)")
    p.add_argument("--types", default=None, help="comma-separated question_type filter")
    p.add_argument("--k", default="1,3,5,10,20")
    p.add_argument("--store-dir", type=Path, default=None)
    p.add_argument("--models-dir", default=os.path.expanduser("~/.mnemo/models"))
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    k_list = sorted({int(x) for x in args.k.split(",")})
    types = set(args.types.split(",")) if args.types else None
    data = json.loads(args.data.read_text())
    items = stratified(data, args.subset, types)

    tmp = None
    if args.store_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="mnemo-lmeval-")
        store_dir = Path(tmp.name)
    else:
        store_dir = args.store_dir

    print(f"isolated store: {store_dir}  embedder={args.embedder}  questions={len(items)}  data={args.data.name}")
    container = core.isolated_container(store_dir, args.embedder, args.models_dir)

    started = time.time()
    report = evaluate(container, items, k_list)
    meta = {"embedder": args.embedder, "data": args.data.name, "n_questions": len(items),
            "k": k_list, "elapsed_seconds": round(time.time() - started, 1)}
    print_report(report, meta, k_list)
    print(f"elapsed {meta['elapsed_seconds']}s")

    core.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or core.RESULTS_DIR / f"longmemeval_{args.embedder}_{len(items)}q.json"
    out.write_text(json.dumps({"meta": meta, "report": report}, indent=2))
    print(f"wrote {out}")
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
