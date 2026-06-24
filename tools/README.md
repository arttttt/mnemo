# tools/eval — in-repo evaluation harness

Developer benchmarks. Not shipped in the wheel; not imported by `mnemo`. Run from a checkout
with `uv run python -m tools.eval.<name>` (the package resolves `mnemo` via the editable
install; data/outputs are gitignored).

## Layout

```
tools/eval/
  core.py          # isolated mnemo Container + manifest I/O + LoCoMo dataset loaders (mnemo)
  metrics.py       # PURE (no mnemo): Recall@k / MRR / AnyEvidence@k / CompleteEvidence@k,
                   #   the top-1 A/B Tally, score_candidates / report_ab, abstention curve,
                   #   lexical corroboration
  rerankers.py     # in-process reranker backends (ONNX CPU, GGUF Metal) + sanity_check
  models.py        # config-driven reranker registry (RerankerSpec + build_reranker, runtime-abstracted)
  locomo.py        # CLI: LoCoMo Tier-1 retrieval benchmark (search path, no recall)
  dump_candidates.py  # CLI: dump hybrid top-N candidates -> candidates.json (once, reusable)
  rerank_ab.py     # CLI: reranker top-1 A/B over candidates.json (ONNX / GGUF backends)
  domain.py        # CLI: project-fact domain eval (п3) — the real go/no-go (retrieval + abstention)
  domain_fixture.py   # pure loader + integrity validator for the domain fixture
  fixtures/
    domain_v1.json # checked-in domain corpus + question slices (answerable/irrelevant/superseded)
  scorers/         # offline scorers that read candidates.json with a non-stock runtime:
    jina_v3_mlx.py #   jina-reranker-v3 (MLX/Metal) — runs in an ISOLATED mlx venv
    qwen3.py       #   Qwen3-Reranker-0.6B (GGUF Q8/Metal)
```

`metrics.py` imports no mnemo, so the isolated-venv scorers reuse it; `core.py` re-exports it
for the in-process runners.

## LoCoMo Tier-1 retrieval

LLM-free, the **search path only** — "if the gold source isn't retrieved, no gate or generator
can recover it". Validates the machinery against a public standard; **not** mnemo on its real
domain. Read the per-category breakdown, never the aggregate (LoCoMo headline numbers are
contested). `AnyEvidence@k` vs `CompleteEvidence@k` splits multi-hop into found-one (coverage)
vs found-none (relevance).

```bash
uv run python -m tools.eval.locomo --embedder pplx --store-dir tools/.locomo_store   # real
uv run python -m tools.eval.locomo --embedder hash --conversations 1                 # smoke
uv run python -m tools.eval.locomo --embedder pplx --store-dir tools/.locomo_store --skip-ingest --abstention
```

The dataset is third-party (license unstated — local testing only) and not committed:
```bash
mkdir -p tools/data
curl -L -o tools/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

## Reranker top-1 A/B

Hypothesis: when a caller asks for the single best memory (`limit=1`), pay for a reranker over
the hybrid top-N. Dump the candidates once, then score any reranker against the same set.

```bash
uv run python -m tools.eval.dump_candidates --store-dir tools/.locomo_store --max-questions 400
# in-process (stock stack):
uv run python -m tools.eval.rerank_ab --backend gguf \
  --gguf-repo gpustack/bge-reranker-v2-m3-GGUF --gguf-file bge-reranker-v2-m3-Q8_0.gguf
uv run python -m tools.eval.rerank_ab --backend onnx \
  --reranker jinaai/jina-reranker-v2-base-multilingual
# out-of-process (isolated mlx venv):
<mlxvenv>/bin/python -m tools.eval.scorers.jina_v3_mlx --model-dir <jina-v3-mlx> \
  --candidates tools/results/candidates.json
uv run python -m tools.eval.scorers.qwen3 --gguf <qwen3-q8.gguf> \
  --candidates tools/results/candidates.json
```

**Finding (LoCoMo, 400-question subset, multilingual rerankers):** reranking the hybrid top-20
lifts top-1 meaningfully (base@1 0.39 → best 0.555). Ranking: **bge-reranker-v2-m3** (+16.5pp,
Q8 GGUF Metal, in-process) > jina-reranker-v3 (+14.5pp, MLX) > Qwen3-Reranker-0.6B (+13.0pp) >
jina-v2 (+10.8pp, ONNX CPU). bge also leads the neutral MIRACL multilingual benchmark — it is
the pick. Caveat: LoCoMo is conversational; the final decision belongs to the domain eval (п3).
```

## п3 — domain eval (project-fact, the real go/no-go)

mnemo on its ACTUAL job: project-fact recall + safe abstention. A self-contained, checked-in
fixture (`fixtures/domain_v1.json` — a fictional `orchard` service + `console` UI + global rules,
27 memories) with a 26-question set across three slices: **answerable**, **irrelevant** (REFUSE;
on-topic traps), and **superseded** (gold = the CURRENT version). Gold is referenced by a stable
memory key, so the fixture survives re-ingestion. The runner ingests into an isolated store, runs
the LLM-free search path, and scores per slice; the chosen prod reranker (bge) is opt-in.

```bash
uv run python -m tools.eval.domain --embedder pplx                 # Tier-1 retrieval + abstention
uv run python -m tools.eval.domain --embedder pplx --reranker bge  # the reranker go/no-go
uv run python -m tools.eval.domain --embedder hash                 # machinery smoke (no model)
```

Metrics: Recall@k / MRR + Any/Complete on `answerable`/`superseded` (the shared `Bucket`); a
stale-leakage check on `superseded` (does an outdated version out-rank the current one); and the
**abstention readout** — both candidate input-gate signals (raw dense cosine AND lexical
corroboration) on answerable vs irrelevant, so the refusal threshold is a curve.

**Findings (domain_v1, pplx):** retrieval is strong (answerable R@1 0.94, MRR 1.00). The chosen
reranker **bge fixes the staleness case** (superseded R@1 0.67 → 1.00; an outdated note no longer
out-ranks the current decision), roughly neutral on easy answerable@1 (small n). Both abstention
signals separate (cosine ans 0.52 / irr 0.32; corroboration ans 2.8 / irr 1.4) — "accept on
strong-cosine OR corroboration ≥ 2" is the promising input gate. Adding a reranker model is a
one-line spec in `models.py` (runtimes are abstracted behind `.rank`).

## Tests

The harness is a separate developer add-on, not shipped product code, so its tests live under
`tools/eval/tests/` and are **kept out of the project suite** (`testpaths = ["tests"]`). Run them
explicitly:

```bash
uv run pytest tools/eval/tests        # eval tooling tests (fixture integrity, runner, registry)
uv run pytest tests                   # the project suite (does NOT include the tooling tests)
```
