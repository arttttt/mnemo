# tools

Developer benchmarks and one-off scripts. Not shipped in the wheel; not imported by
`mnemo`. Run them from a checkout with `uv run python tools/<script>.py`.

## `locomo_bench.py` — LoCoMo Tier-1 retrieval benchmark (LLM-free)

Runs mnemo's own store + `search` over the public
[LoCoMo](https://github.com/snap-research/locomo) dataset and scores **Recall@k / MRR@k**
of the gold source turns. This is the cheap, deterministic, **search-path-only** signal —
**no recall, no generator, no LLM** — i.e. Tier 1 of the benchmark plan: "if the gold
source is not retrieved, no gate or generator can recover it."

### What it does

1. **Isolated environment.** Builds the `Container` against a throwaway SQLite store with
   the reranker and generator forced **off**. It never touches the live `~/.mnemo` store
   and never loads a model beyond the embedder.
2. **One project per conversation.** `create_project(sample_id)`; `search` is scoped to
   that project, so conversation A cannot answer conversation B (mnemo's
   `project = ? OR scope='global'`).
3. **Ingest each turn as a memory.** `content = "Speaker (date): text [+ image caption]"`,
   tagged `dialog:<dia_id>`. The Speaker/date prefix adds temporal context **and** keeps
   otherwise-identical chitchat unique so the content-hash dedup doesn't fold two turns
   into one. Turns embed inline via the sync scheduler — there is no queue to drain.
4. **Score.** Per QA, `search` scoped to the conversation → map returned memory ids back to
   dia_ids → Recall@k / MRR@k against the QA's `evidence`. The id→dia_id bridge is captured
   at ingest (`remember` returns the id) because `SearchResult` does not carry tags.

### Running

```bash
# Real numbers — the production embedder (heavy: ~6k turns + ~2k queries to embed on CPU).
uv run python tools/locomo_bench.py --embedder pplx

# Fast machinery smoke — deterministic hash double (lexical-only; numbers are MEANINGLESS).
uv run python tools/locomo_bench.py --embedder hash --conversations 1

# Persist the ingested store and re-query without re-embedding.
uv run python tools/locomo_bench.py --embedder pplx --store-dir tools/.locomo_store
uv run python tools/locomo_bench.py --embedder pplx --store-dir tools/.locomo_store --skip-ingest

# Add the input-gate refusal curve (extra encodes per query).
uv run python tools/locomo_bench.py --embedder pplx --abstention
```

Key flags: `--conversations N` / `--max-turns N` (subset for dev), `--k 1,3,5,10,20`
(cutoffs), `--store-dir` (reuse an ingest), `--skip-ingest` (reload the bridge from the
store's manifest). Results are written to `tools/results/locomo_<embedder>_<ts>.json`.

### The dataset

Not committed (third-party, **license unstated** in the LoCoMo README — fine for local
private testing; verify before publishing any numbers). Download once:

```bash
mkdir -p tools/data
curl -L -o tools/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

### Reading the output — caveats (do not over-read)

- **Per-category, never the aggregate.** LoCoMo headline numbers are contested (Zep's 84
  was independently re-scored to ~58). The harness prints a per-category breakdown; read it.
- **This validates the MACHINERY against a standard, not mnemo on its real job.** LoCoMo is
  conversational; a turn is not an atomic fact, so multi-hop/temporal answers (which need
  aggregation across turns) fit a fact store awkwardly. A modest absolute here is expected
  and is not the project-fact-memory number.
- **Category 5 is adversarial — the should-REFUSE slice.** Its retrieval is shown for
  visibility only, never credited as recall. With `--abstention`, the input-gate curve
  (true-refusal on adversarial vs false-refusal on answerable, swept over a top-1 **raw
  cosine** threshold — not the uninformative RRF score) is computed. Expect *poor*
  separation: LoCoMo adversarial are on-topic trap questions, so a relevant-looking turn is
  still retrieved — that low separation is the informative result (these need an
  output/answerability gate, not an input-relevance one).
- `--embedder hash` is for exercising the runner only; it is lexical bag-of-tokens and its
  retrieval quality is not representative.

### Not yet here (Tier 2, needs the generator / a gate)

Answer correctness and faithfulness (answer entailed by the gold sources), and the
search-vs-recall head-to-head. Those require the recall pipeline and are out of scope for
this LLM-free pass.
