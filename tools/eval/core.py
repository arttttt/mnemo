"""mnemo-side foundation for the eval harness: the isolated Container + manifest I/O + the
LoCoMo dataset loaders. Pure metrics live in tools.eval.metrics and are re-exported here, so
an in-process runner can `from tools.eval import core` and get everything, while an
isolated-venv scorer imports tools.eval.metrics directly (no mnemo).

Imports mnemo via the editable install; run the CLIs with `python -m tools.eval.<name>`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from mnemo.infrastructure.composition import build_container
from mnemo.infrastructure.config import Config
from mnemo.infrastructure.container import Container

from tools.eval.metrics import (  # noqa: F401  (re-exported for the runners)
    Bucket, Tally, abstention_curve, cosine, report_ab, score_candidates,
)

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = _REPO / "tools" / "data" / "locomo10.json"
RESULTS_DIR = _REPO / "tools" / "results"
DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

CATEGORY_LABELS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ADVERSARIAL = 5


# --------------------------------------------------------------------------- dataset (LoCoMo)


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
    """Yield (session_key, session_datetime, turn) in conversation order."""
    sessions = sorted(
        (int(m.group(1)), key)
        for key in conversation
        if (m := _SESSION_RE.match(key)) and isinstance(conversation[key], list)
    )
    for _, key in sessions:
        date = conversation.get(f"{key}_date_time", "")
        for turn in conversation[key]:
            yield key, date, turn


def turn_content(speaker: str, date: str, turn: dict) -> str:
    """One dialog turn as memory text. The Speaker/date prefix adds temporal context AND keeps
    otherwise-identical chitchat unique (so content-hash dedup doesn't fold two turns into one
    and break the dia_id map). An image turn's caption is appended."""
    text = (turn.get("text") or "").strip()
    caption = (turn.get("blip_caption") or "").strip()
    body = f"{text} [shared image: {caption}]" if caption else text
    prefix = f"{speaker} ({date}): " if date else f"{speaker}: "
    return prefix + body


def eligible_questions(data: list[dict], conversations: int | None, *, answerable_only: bool):
    """(slug, qa, sorted-evidence) triples worth scoring; answerable_only drops adversarial."""
    convs = data if conversations is None else data[:conversations]
    out = []
    for sample in convs:
        for qa in sample["qa"]:
            if not qa.get("evidence"):
                continue
            if answerable_only and qa.get("category") == ADVERSARIAL:
                continue
            out.append((sample["sample_id"], qa, sorted(set(qa["evidence"]))))
    return out


def subsample(items: list, n: int | None) -> list:
    """Evenly subsample to n items across the list (deterministic)."""
    if not n or len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


# --------------------------------------------------------------------------- harness


def build_config(store_dir: Path, embedder: str, models_dir: str) -> Config:
    """Config pinned to an ISOLATED store with NO model stages — the eval is the search path
    only (reranker/generator off, never touches the live ~/.mnemo store)."""
    store_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        data_dir=str(store_dir), embedder=embedder,
        sqlite_path=str(store_dir / "memory.db"), models_dir=models_dir,
        reranker="off", generator="off",
    )


def isolated_container(store_dir: Path, embedder: str, models_dir: str) -> Container:
    return build_container(build_config(store_dir, embedder, models_dir))


def load_manifest(store_dir: Path, embedder: str) -> dict[str, set[str]]:
    """The memory-id -> {dia_id} bridge captured at ingest (SearchResult carries no tags, and
    the id is a uuid not a content hash, so only the ingest run can map them)."""
    path = store_dir / "locomo_manifest.json"
    if not path.exists():
        raise SystemExit(f"no manifest at {path} — ingest the store first (eval.locomo)")
    saved = json.loads(path.read_text())
    if saved.get("embedder") != embedder:
        raise SystemExit(f"store ingested with embedder={saved.get('embedder')!r}, not {embedder!r}")
    return {mid: set(dias) for mid, dias in saved["id_to_dia"].items()}


def save_manifest(store_dir: Path, embedder: str, id_to_dia: dict[str, set[str]], n_turns: int) -> None:
    (store_dir / "locomo_manifest.json").write_text(json.dumps({
        "embedder": embedder, "n_turns": n_turns,
        "id_to_dia": {mid: sorted(dias) for mid, dias in id_to_dia.items()},
    }))
