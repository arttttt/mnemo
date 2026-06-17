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
