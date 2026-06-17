# 08 — Background consolidation

The goal is to move all the "smart" processing off the hot path into the background, so that writes
stay cheap and the small models work in rare batches (which is why small models suffice, and why
there is no resident RAM "hog").

## What consolidation does

1. **Near-dup merge.** The write path no longer suppresses near-duplicates, so the worker merges genuine
   near-duplicates / paraphrases here — with full context, not a blind write-time threshold.
2. **Cluster summarization.** A group of small same-topic records → one compressed memory (originals kept as `superseded`).
3. **Insight extraction.** Recurring patterns/gotchas → a `learning` record (cross-project ones → `__global__`).
4. **Contradiction flags.** *Flags* likely contradictions for human/agent review — it does **not** auto-mark
   anything stale (currency changes only on an explicit signal; see [04-data-model.md](04-data-model.md)).
5. **(post-MVP)** Precompute a `recall` digest off the hot path, once the deferred `recall` lands.
6. **(post-MVP) Importance scoring.**

Everything is soft (recoverable), with audit fields. No physical deletion.

## When it runs (triggers)

- **By volume:** `MNEMO_CONSOLIDATE_EVERY` new records accumulated (default 50).
- **By idle:** agents inactive for X minutes, but the service is still alive before grace — a convenient window.
- **Manually:** `consolidate(project?, force=true)` / CLI.
- **Never on the hot path** and not during peak writes (we check the write-queue load).

## How it runs — a staged pipeline (transient)

Consolidation is **not** one mega-call to a single LLM. It is a pipeline of focused stages, each handled
by the **model class best suited to it**. This was measured (see
[research/generator-benchmark.md](research/generator-benchmark.md)): small specialist cross-encoders beat a
general 3–4B LLM at the routing/classification stages, and the LLM is reserved for the one task that
genuinely needs text synthesis.

```
trigger → select candidates (new records + their embedding neighbours, scope = project)
        → STAGE 1  routing / dedup — a cross-encoder RERANKER scores seed-vs-candidate and thresholds
            them into same-topic (consolidate) vs unrelated. Tiny, fast, multilingual, and it can only
            ever return the given candidate ids, so it cannot hallucinate ids.
        → STAGE 2  contradiction — an NLI cross-encoder flags likely contradictions (run BOTH
            directions; NLI is asymmetric), flag-only — it never auto-invalidates.
        → STAGE 3  generation — load the small LLM generator ONLY for the confirmed same-topic clusters
            → one faithful, concise merged / summarized record (originals → superseded). Unload after; RAM freed.
        → apply to the store (merge / supersede / insert) through the write queue
        → write a consolidation log
```

**Why staged.** Routing ("are these the same specific topic, or just embedding-near?") and contradiction
detection are *classification* problems, not generation. On real candidate groups from the live store a
**107M multilingual reranker reached ~0.91 accuracy** with a balanced safe/recall profile and **zero id
hallucination**, while 3–4B LLMs collapsed on the same task (recall of true duplicates near zero, frequent
id hallucination, broken JSON). So each stage uses the right tool — lighter on RAM, more reliable — and the
LLM only writes the merged record. This is the project's "minimum models, each earns its place" principle
made concrete. The cheap cross-encoder stages absorb the candidate volume; the LLM runs only on the few
clusters that survive routing.

**Safety is asymmetric.** A wrong merge permanently corrupts memory; a missed merge is harmless. So every
stage prefers `keep_separate` when unsure, and **topic granularity is respected** — two records about the
same subject under different `topic_key`s are kept separate (that separation is the user's intent, not a
mistake to "fix").

## Pipeline abstraction — composable stages

The staged pipeline above is **not** a hard-coded monolith; it is **composed from interchangeable
stages**, so a new consolidation task is sketched at development time by listing blocks. Four pieces,
all honoring the layering in [09-tech-stack.md](09-tech-stack.md):

- **`PipelineStage`** (port, in `application`). One stage = one responsibility (SRP): it reads named
  slots from the context, uses **one model behind its own port**, and writes named slots. It declares a
  light contract — `requires` / `provides` (the slot names it needs / fills).
- **`Pipeline`** (in `application`). A trivial **sequential** runner. At construction it validates the
  chain (every stage's `requires` is produced by an earlier stage or the `Job`); after each stage it
  asserts the stage actually filled its `provides`. No event bus, no DAG — the boundary between stages
  is simply the function return.
- **`Job` → `PipelineContext` → `Result`.** `Job` is the immutable input (the triggering `seeds`,
  scope, config). It seeds an immutable `PipelineContext` that each stage extends copy-on-write. The
  output is a `Result` — a **plan** of operations (`merge` / `supersede` / `insert` / `flag`), never a
  direct store write.
- **`Executor`** (adapter). Applies the plan **idempotently** through the write queue. Operations
  address records by id, so re-running a batch cannot corrupt data.

**Assembly is static, at development time** — one small builder function per task, reviewed as plain code:

```python
def build_dedup_pipeline(deps):            # application/pipelines/dedup.py
    return Pipeline([
        SelectCandidatesStage(deps.embedder, deps.repo),
        RoutingStage(deps.reranker),
        ContradictionStage(deps.nli),
        GenerationStage(deps.generator),
        FaithfulnessGate(),
        PlanStage(),
    ])
```

A new task = a new builder file listing different blocks. Swapping a model = a new adapter behind the
same port (OCP, no stage change); a new model *class* = a new stage + a new port (OCP, the core is
untouched). Stages are swappable (LSP) and depend only on ports (DIP).

**Lifecycle and RAM.** A heavy stage loads its model lazily at the start of its `run` and **unloads at
the end** (the generator frees its RAM before the next stage), keeping the on-demand budget in
[07-lifecycle-and-ram.md](07-lifecycle-and-ram.md). The stage boundary is also the natural logging
point (`stage=routing in=12 out=3 dur=…`).

**What is validated — and what is not.** Validation is layered and targeted, never a generic per-stage
content check:
- **Structural, every stage (cheap):** the `requires` / `provides` contract — catches a mis-assembled
  pipeline at construction, not as a runtime surprise.
- **Reranker / NLI:** the output is scores — structurally always valid; the decision is a *calibrated
  threshold*, not a post-hoc check.
- **Generator:** valid JSON is **guaranteed by grammar / guided decoding** — enforced, not validated.
  The one content risk (a fabricated or degenerate merge) is checked by a dedicated **`FaithfulnessGate`
  stage** — grounding (every id / number in the output ⊆ the cluster inputs) + coherence (not
  degenerate). A failed gate does not error; it **falls back to `keep_separate`** (asymmetric safety: a
  missed merge is harmless, a wrong merge corrupts memory).
- **`PlanStage`:** deterministic, with invariants asserted (never merges across `topic_key`, never
  supersedes a record outside the candidates).

**Failure.** Any stage that throws or times out aborts the **whole job** — nothing is applied, the
source memories are untouched ("processing must not drop data"). Because apply is a single plan at the
end (not per-stage writes), there is never a partial application.

## The models (one per stage)

Everything is multilingual (no language is guaranteed to be English), to match the embedder.

| Stage | Model class | Why this class | Model |
|---|---|---|---|
| candidate select | **embedder** (bi-encoder) | cheap recall over the whole store; one vector per record, cached | pplx-embed-v1-0.6b int8 (chosen) |
| routing / dedup | **reranker** (cross-encoder) | sees the pair together (cross-attention) → precise same-vs-different on the few candidates; can't hallucinate ids | small multilingual — **not yet chosen** (benchmarking) |
| contradiction | **NLI** (cross-encoder) | purpose-built entailment / contradiction; run bidirectionally | small multilingual — **not yet chosen** |
| generation | small instruct **LLM** | the only stage that needs text synthesis | ≤4B multilingual — **not yet chosen** |

No specific model is committed except the embedder; candidates and measurements live in the research doc.

Embedder vs reranker: the embedder encodes each text independently into a reusable vector (fast, scales to
the whole store, but approximate); the reranker scores a *pair together* (precise, but one forward pass per
pair, so only on the embedder's shortlist). They are a classic two-stage retrieve-then-rerank.

**Generation-call reliability (the LLM stage only):** a **flat** JSON schema (no nested `$defs` — nesting
sharply raises invalid output at small sizes), **grammar / guided decoding** (GBNF or `guided_json` — valid
JSON guaranteed by the decoder), `temperature=0`, and a focused, concise-by-instruction prompt with one
example. Describe the schema **in the prompt** — the grammar is not injected into it.

## Idempotency and safety
- Consolidation is repeatable: operations apply by id; re-running the same batch does not corrupt data.
- On a model failure/timeout — the batch is skipped, the source memories are untouched ("processing must not drop data").
- A cap on batch size and `MAX_TOKENS` so the window and RAM don't balloon.

## Degradation — economy mode
The stages degrade independently:
- **No generator** (`MNEMO_GENERATOR=off`): routing + contradiction-flag still run (the cross-encoders are
  cheap); near-dups are marked `superseded` by reranker score + rules, but no new summaries/insights are written.
- **No reranker:** fall back to embedder cosine + a threshold for dedup (coarser; more false neighbours).
- **Nothing but the embedder:** memory still works as a plain cheap vector store, just "less smart".
