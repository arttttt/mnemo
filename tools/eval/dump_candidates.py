#!/usr/bin/env python3
"""Dump hybrid top-N retrieval candidates per LoCoMo question to JSON.

Decouples retrieval (here, project venv) from reranking (rerank_ab + the isolated scorers),
so comparing rerankers no longer re-runs search. candidates are in RRF order — candidates[0]
is the no-rerank baseline top-1.

    python -m tools.eval.dump_candidates --store-dir tools/.locomo_store --max-questions 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mnemo.application.results.search_result import SearchResult

from tools.eval import core


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store-dir", type=Path, required=True, help="store dir from an eval.locomo ingest")
    p.add_argument("--data", type=Path, default=core.DEFAULT_DATA)
    p.add_argument("--embedder", default="pplx", choices=["pplx", "hash"])
    p.add_argument("--models-dir", default=str(Path("~/.mnemo/models").expanduser()))
    p.add_argument("--candidates", type=int, default=20, help="hybrid top-N pool per question")
    p.add_argument("--max-questions", type=int, default=400)
    p.add_argument("--out", type=Path, default=core.RESULTS_DIR / "candidates.json")
    args = p.parse_args()

    data = core.load_dataset(args.data)
    id_to_dia = core.load_manifest(args.store_dir, args.embedder)
    container = core.isolated_container(args.store_dir, args.embedder, args.models_dir)

    eligible = core.subsample(core.eligible_questions(data, None, answerable_only=True), args.max_questions)
    print(f"dumping {len(eligible)} questions (top-{args.candidates})…", flush=True)
    out = []
    for i, (slug, qa, evidence) in enumerate(eligible, 1):
        if i % 100 == 0:
            print(f"  dumped {i}/{len(eligible)}", flush=True)
        hits: list[SearchResult] = container.search.execute(
            query=qa["question"], scope="project", project=slug, limit=args.candidates
        )
        out.append({
            "question": qa["question"], "evidence": evidence, "category": qa["category"],
            "candidates": [{"content": h.content, "dia": sorted(id_to_dia.get(h.id, set()))} for h in hits],
        })
    args.out.write_text(json.dumps(out))
    print(f"wrote {len(out)} questions -> {args.out}")


if __name__ == "__main__":
    main()
