#!/usr/bin/env python3
"""LoCoMo Tier-1 retrieval benchmark for mnemo — LLM-free, the SEARCH path only (no recall).

What this is
------------
A small adapter that runs mnemo's own store + `search` over the public LoCoMo dataset
(https://github.com/snap-research/locomo, ``data/locomo10.json``: 10 long multi-session
conversations, ~5.9k turns, ~2k QA pairs whose ``evidence`` field already names the gold
source turns). It validates the harness MACHINERY — ingest -> retrieve -> score against a
standard — and gives the cheap, deterministic, substrate-direction signal (does the gold
source land in top-k) WITHOUT the generator.

What this is NOT
----------------
- Not a measure of mnemo on its real domain (project facts) — LoCoMo is conversational and
  a turn is not an atomic fact, so multi-hop/temporal answers fit a fact store awkwardly.
- Not a leaderboard number. Read the PER-CATEGORY breakdown, never the aggregate: LoCoMo
  headline numbers are contested (Zep's 84 was independently re-scored to ~58). Category 5
  is adversarial — the should-REFUSE slice — and is reported apart from the retrieval mean.

Isolation
---------
Builds mnemo's own Container against a throwaway SQLite store (``--store-dir``, default a
fresh temp dir) with reranker/generator forced OFF, so it NEVER touches the live ~/.mnemo
store and never loads an LLM. One store, one project per conversation; ``search`` is scoped
to the conversation, so conversation A can't answer conversation B.

Per conversation: ``create_project(sample_id)`` -> ingest each dialog turn as a memory
(content = ``Speaker (date): text [+ image caption]``, tagged ``dialog:<dia_id>``) -> per QA
run ``search`` scoped to that project, map the returned memory ids back to dia_ids, and
score Recall@k / MRR@k against ``evidence``. The id->dia_id bridge is captured at ingest
(``remember`` returns the id) because ``SearchResult`` does not carry tags.

Usage
-----
    uv run python tools/locomo_bench.py --embedder pplx                 # the real run (heavy)
    uv run python tools/locomo_bench.py --embedder hash --conversations 1   # machinery smoke
    uv run python tools/locomo_bench.py --embedder pplx --store-dir .bench  # reuse an ingest
    uv run python tools/locomo_bench.py --embedder pplx --abstention        # + refusal curve

``--embedder hash`` is the deterministic test double: it exercises the runner end-to-end in
seconds but is lexical-only, so its retrieval numbers are meaningless — use ``pplx`` (the
production embedder) for real numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Run straight from a checkout without an install: make `import mnemo` resolve.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from mnemo.application.results.search_result import SearchResult  # noqa: E402
from mnemo.application.use_cases.create_project import ProjectAlreadyExists  # noqa: E402
from mnemo.infrastructure.composition import build_container  # noqa: E402
from mnemo.infrastructure.config import Config  # noqa: E402
from mnemo.infrastructure.container import Container  # noqa: E402

DEFAULT_DATA = _REPO / "tools" / "data" / "locomo10.json"
RESULTS_DIR = _REPO / "tools" / "results"
DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

# LoCoMo question categories. Only 5=adversarial is verifiable from the data itself (it
# carries `adversarial_answer` and no `answer`); the rest follow the paper's taxonomy and
# are labels for reading the breakdown, not load-bearing.
CATEGORY_LABELS = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
ADVERSARIAL = 5


# --------------------------------------------------------------------------- dataset


def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"dataset not found: {path}\n"
            f"Download it (third-party, license unstated — local testing only):\n"
            f"    mkdir -p {path.parent} && curl -L -o {path} {DATASET_URL}"
        )
    return json.loads(path.read_text())


_SESSION_RE = re.compile(r"^session_(\d+)$")


def iter_turns(conversation: dict):
    """Yield (session_index, session_datetime, turn) in conversation order."""
    sessions = sorted(
        (int(m.group(1)), key)
        for key, _ in conversation.items()
        if (m := _SESSION_RE.match(key)) and isinstance(conversation[key], list)
    )
    for index, key in sessions:
        date = conversation.get(f"{key}_date_time", "")
        for turn in conversation[key]:
            yield index, date, turn


def turn_content(speaker: str, date: str, turn: dict) -> str:
    """The memory text for one dialog turn. The Speaker/date prefix doubles as temporal
    context AND makes otherwise-identical chitchat unique, so the content-hash dedup does
    not silently fold two turns into one (which would break the 1:1 dia_id map). An image
    turn's caption is appended — some `evidence` points at image turns."""
    text = (turn.get("text") or "").strip()
    caption = (turn.get("blip_caption") or "").strip()
    body = f"{text} [shared image: {caption}]" if caption else text
    prefix = f"{speaker} ({date}): " if date else f"{speaker}: "
    return prefix + body


# --------------------------------------------------------------------------- env


def build_config(store_dir: Path, embedder: str, models_dir: str) -> Config:
    """A Container config pinned to an isolated store with NO model stages — Tier 1 is the
    search path only, so the reranker and generator are forced off and never constructed."""
    store_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        data_dir=str(store_dir),
        embedder=embedder,
        sqlite_path=str(store_dir / "memory.db"),
        models_dir=models_dir,
        reranker="off",
        generator="off",
    )


# --------------------------------------------------------------------------- ingest


def ingest(
    container: Container, data: list[dict], conversations: int | None, max_turns: int | None
) -> tuple[dict[str, set[str]], int]:
    """Register a project per conversation and store each turn as a memory. Returns the
    id -> {dia_id} bridge and the turn count. Idempotent across re-runs: an already-created
    project is reused, and identical content dedups to the same memory id (which then simply
    covers the several dia_ids that share that text)."""
    id_to_dia: dict[str, set[str]] = defaultdict(set)
    n_turns = 0
    convs = data if conversations is None else data[:conversations]
    for sample in convs:
        slug = sample["sample_id"]
        conversation = sample["conversation"]
        try:
            container.create_project.execute(slug)
        except ProjectAlreadyExists:
            pass
        per_conv = 0
        for _, date, turn in iter_turns(conversation):
            if max_turns is not None and per_conv >= max_turns:
                break
            speaker = turn.get("speaker", "")
            dia_id = turn["dia_id"]
            result = container.remember.execute(
                content=turn_content(speaker, date, turn),
                type="discussion",
                scope="project",
                project=slug,
                tags=[f"dialog:{dia_id}"],
            )
            id_to_dia[result.id].add(dia_id)
            per_conv += 1
            n_turns += 1
        print(f"  ingested {slug}: {per_conv} turns", flush=True)
    return id_to_dia, n_turns


# --------------------------------------------------------------------------- scoring


@dataclass
class Bucket:
    """Accumulates retrieval scores for one slice of questions."""

    n: int = 0
    recall_at: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    rr_sum: float = 0.0

    def add(self, ranked_dia: list[set[str]], evidence: set[str], k_list: list[int]) -> None:
        self.n += 1
        for k in k_list:
            retrieved: set[str] = set().union(*ranked_dia[:k]) if ranked_dia[:k] else set()
            self.recall_at[k] += len(retrieved & evidence) / len(evidence)
        for rank, dset in enumerate(ranked_dia):
            if dset & evidence:
                self.rr_sum += 1.0 / (rank + 1)
                break

    def summary(self, k_list: list[int]) -> dict:
        if self.n == 0:
            return {"n": 0}
        return {
            "n": self.n,
            "recall_at_k": {k: round(self.recall_at[k] / self.n, 4) for k in k_list},
            "mrr": round(self.rr_sum / self.n, 4),
        }


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def evaluate(
    container: Container,
    data: list[dict],
    id_to_dia: dict[str, set[str]],
    k_list: list[int],
    conversations: int | None,
    abstention: bool,
) -> dict:
    limit = max(k_list)
    per_category: dict[int, Bucket] = defaultdict(Bucket)
    answerable = Bucket()  # categories 1-4 combined
    adversarial_retrieval = Bucket()  # cat 5, for visibility only
    # Abstention substrate: top-1 RAW cosine (the input-gate signal FEEDBACK calls for —
    # NOT the returned RRF score, which is an uninformative absolute). Positives = answerable,
    # negatives = adversarial. Opt-in: it costs two extra encodes per query.
    abst_pos: list[float] = []
    abst_neg: list[float] = []

    convs = data if conversations is None else data[:conversations]
    # The query phase has no per-item output of its own (~20 min on the full set), so it
    # reads as a hang from outside. Emit a coarse counter against the known total instead.
    total = sum(1 for s in convs for qa in s["qa"] if qa.get("evidence"))
    print(f"querying {total} questions…", flush=True)
    n_questions = 0
    for sample in convs:
        slug = sample["sample_id"]
        for qa in sample["qa"]:
            evidence = set(qa.get("evidence") or [])
            if not evidence:
                continue
            category = qa.get("category")
            question = qa["question"]
            hits: list[SearchResult] = container.search.execute(
                query=question, scope="project", project=slug, limit=limit
            )
            ranked_dia = [id_to_dia.get(hit.id, set()) for hit in hits]
            n_questions += 1

            if category == ADVERSARIAL:
                adversarial_retrieval.add(ranked_dia, evidence, k_list)
            else:
                per_category[category].add(ranked_dia, evidence, k_list)
                answerable.add(ranked_dia, evidence, k_list)

            if abstention and hits:
                top1 = _cosine(
                    container.embedder.encode(question),
                    container.embedder.encode(hits[0].content),
                )
                (abst_neg if category == ADVERSARIAL else abst_pos).append(top1)

            if n_questions % 200 == 0:
                print(f"  queried {n_questions}/{total}", flush=True)
        print(f"  done {slug}: {n_questions}/{total} questions", flush=True)

    report: dict = {
        "n_questions": n_questions,
        "answerable": answerable.summary(k_list),
        "by_category": {
            str(cat): {"label": CATEGORY_LABELS.get(cat, "?"), **per_category[cat].summary(k_list)}
            for cat in sorted(per_category)
        },
        "adversarial_retrieval": {
            "label": "cat 5 — should-refuse; retrieval shown for visibility only",
            **adversarial_retrieval.summary(k_list),
        },
    }
    if abstention:
        report["abstention"] = abstention_curve(abst_pos, abst_neg)
    return report


def abstention_curve(positives: list[float], negatives: list[float]) -> dict:
    """Sweep the input-gate threshold T over top-1 raw cosine: for each T, the false-refusal
    rate on answerable (positives below T) vs the true-refusal rate on adversarial (negatives
    below T). The operating point is a CURVE, not one number. NOTE: LoCoMo adversarial are
    on-topic TRAP questions — a relevant-looking turn IS retrieved — so an input-relevance gate
    is expected to separate them poorly here; that low separation is itself the informative
    result (these need an OUTPUT/answerability gate, not an input one)."""
    if not positives or not negatives:
        return {"note": "need both answerable and adversarial queries; rerun without subsetting"}
    grid = [round(0.05 * i, 2) for i in range(0, 21)]
    points = [
        {
            "T": t,
            "false_refusal_on_answerable": round(sum(p < t for p in positives) / len(positives), 4),
            "true_refusal_on_adversarial": round(sum(n < t for n in negatives) / len(negatives), 4),
        }
        for t in grid
    ]
    return {
        "n_answerable": len(positives),
        "n_adversarial": len(negatives),
        "mean_cosine_answerable": round(sum(positives) / len(positives), 4),
        "mean_cosine_adversarial": round(sum(negatives) / len(negatives), 4),
        "curve": points,
    }


# --------------------------------------------------------------------------- report


def print_report(report: dict, meta: dict, k_list: list[int]) -> None:
    bar = "=" * 72
    print(f"\n{bar}\nLoCoMo Tier-1 retrieval (mnemo search, no recall)\n{bar}")
    print(
        f"embedder={meta['embedder']}  conversations={meta['conversations']}  "
        f"turns={meta['n_turns']}  questions={report['n_questions']}"
    )
    header = "slice".ljust(22) + "n".rjust(6) + "  " + "  ".join(f"R@{k}".rjust(7) for k in k_list) + "   MRR".rjust(9)
    print("\n" + header)
    print("-" * len(header))

    def row(name: str, summary: dict) -> None:
        if not summary.get("n"):
            return
        cells = "  ".join(f"{summary['recall_at_k'][k]:.4f}".rjust(7) for k in k_list)
        print(name.ljust(22) + str(summary["n"]).rjust(6) + "  " + cells + f"{summary['mrr']:.4f}".rjust(9))

    row("ANSWERABLE (1-4)", report["answerable"])
    for cat, summary in report["by_category"].items():
        row(f"  {cat} {summary['label']}", summary)
    row("adversarial (5)*", report["adversarial_retrieval"])
    print("\n* category 5 is the should-REFUSE slice; its retrieval is for visibility, not credit.")

    if "abstention" in report:
        ab = report["abstention"]
        if "curve" in ab:
            print(
                f"\nAbstention (input-gate, top-1 raw cosine) — mean cos: "
                f"answerable={ab['mean_cosine_answerable']} adversarial={ab['mean_cosine_adversarial']}"
            )
            print("   T     false-refusal(ans)   true-refusal(adv)")
            for pt in ab["curve"]:
                if pt["T"] in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
                    print(
                        f"  {pt['T']:.2f}        {pt['false_refusal_on_answerable']:.4f}"
                        f"             {pt['true_refusal_on_adversarial']:.4f}"
                    )
        else:
            print(f"\nAbstention: {ab.get('note')}")
    print(bar + "\n")


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedder", default="pplx", choices=["pplx", "hash"],
                        help="production embedder (pplx) for real numbers; hash = fast machinery smoke")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="path to locomo10.json")
    parser.add_argument("--store-dir", type=Path, default=None,
                        help="reuse/persist the isolated store here (default: a throwaway temp dir)")
    parser.add_argument("--models-dir", default=os.path.expanduser("~/.mnemo/models"),
                        help="model cache dir (reuses the live pplx weights so nothing re-downloads)")
    parser.add_argument("--conversations", type=int, default=None, help="limit to the first N conversations")
    parser.add_argument("--max-turns", type=int, default=None, help="limit turns per conversation (dev)")
    parser.add_argument("--k", default="1,3,5,10,20", help="comma-separated cutoffs for Recall@k / MRR")
    parser.add_argument("--abstention", action="store_true",
                        help="also compute the input-gate refusal curve (extra encodes per query)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="store already populated (with --store-dir): only re-run the query loop")
    args = parser.parse_args()

    k_list = sorted({int(x) for x in args.k.split(",")})
    data = load_dataset(args.data)

    tmp = None
    if args.store_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="mnemo-locomo-")
        store_dir = Path(tmp.name)
    else:
        store_dir = args.store_dir

    config = build_config(store_dir, args.embedder, args.models_dir)
    print(f"isolated store: {store_dir}  embedder={args.embedder}")
    container = build_container(config)

    manifest_path = store_dir / "locomo_manifest.json"
    started = time.time()
    if args.skip_ingest:
        # The memory id is a random uuid (not a content hash), so the id->dia bridge can only
        # come from the ingest run that assigned the ids — reload it from the manifest.
        if not manifest_path.exists():
            raise SystemExit(f"--skip-ingest needs a prior ingest manifest at {manifest_path}")
        saved = json.loads(manifest_path.read_text())
        if saved.get("embedder") != args.embedder:
            raise SystemExit(
                f"store was ingested with embedder={saved.get('embedder')!r}, not {args.embedder!r} "
                f"(vector spaces differ) — re-ingest or match the embedder"
            )
        id_to_dia = {mid: set(dias) for mid, dias in saved["id_to_dia"].items()}
        n_turns = saved["n_turns"]
    else:
        print("ingesting (turns embed inline via the sync scheduler — no queue to drain):")
        id_to_dia, n_turns = ingest(container, data, args.conversations, args.max_turns)
        manifest_path.write_text(json.dumps({
            "embedder": args.embedder,
            "n_turns": n_turns,
            "conversations": args.conversations,
            "id_to_dia": {mid: sorted(dias) for mid, dias in id_to_dia.items()},
        }))
    print(f"ingest/bridge done in {time.time() - started:.1f}s; querying…")

    report = evaluate(container, data, id_to_dia, k_list, args.conversations, args.abstention)
    elapsed = time.time() - started
    meta = {
        "embedder": args.embedder,
        "conversations": args.conversations or len(data),
        "n_turns": n_turns,
        "k": k_list,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": int(time.time()),
    }
    print_report(report, meta, k_list)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"locomo_{args.embedder}_{meta['timestamp']}.json"
    out.write_text(json.dumps({"meta": meta, "report": report}, indent=2))
    print(f"wrote {out}")

    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
