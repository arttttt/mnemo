# 08 — Background consolidation

The goal is to move all the "smart" processing off the hot path into the background, so that writes
stay cheap and the small model works in rare batches (which is why a small model suffices, and why
there is no RAM "hog").

## What consolidation does

1. **Near‑dup merge.** The write path no longer suppresses near‑duplicates, so the worker merges genuine
   near‑duplicates / paraphrases here — with full context, not a blind write‑time threshold.
2. **Cluster summarization.** A group of small same‑topic records → one compressed memory (originals kept as `superseded`).
3. **Insight extraction.** Recurring patterns/gotchas → a `learning` record (cross‑project ones → `__global__`).
4. **Contradiction flags.** *Flags* likely contradictions for human/agent review — it does **not** auto‑mark
   anything stale (currency changes only on an explicit signal; see [04-data-model.md](04-data-model.md)).
5. **(post‑MVP) Precompute a `recall` digest** off the hot path, once the deferred `recall` lands.
6. **(post‑MVP) Importance scoring.** Once `importance` returns (post‑MVP), the worker can score it consistently
   (heuristic and/or the small model).

Everything is soft (recoverable), with audit fields. No physical deletion.

## When it runs (triggers)

- **By volume:** `MNEMO_CONSOLIDATE_EVERY` new records accumulated (default 50).
- **By idle:** agents inactive for X minutes, but the service is still alive before grace — a convenient window.
- **Manually:** `consolidate(project?, force=true)` / CLI.
- **Never on the hot path** and not during peak writes (we check the write‑queue load).

## How it runs (transiently)

```
trigger → select a batch of candidates (new records + their embedding neighbors, scope = project)
        → load the generator (llama.cpp + Qwen3‑4B Q4)        # ← model load
        → for each group: a single guided‑JSON call
            (input: a set of memories; output: {action, merged_content, supersede_ids, tags})
        → apply to the store (merge/supersede/insert) through the write queue
        → unload the generator                                # ← RAM freed
        → write a consolidation log
```

The generator lives only for the job's duration. Consolidation is **designed concurrent from the start** —
a worker pool over batches, not one serial pass — since 10+ agents accumulate memory quickly. The local
inference server is chosen for concurrency: **vLLM/SGLang** (continuous batching) when the model must serve
parallel requests; llama.cpp on‑demand for the lightest single‑stream case. The store backend (SQLite)
absorbs concurrent writes.

## Model call contract (reliability at 4B)

- **Flat JSON schema**, no nested `$defs` (nesting sharply raises the share of invalid output at 4B).
- **Grammar/guided decoding**: GBNF (llama.cpp) or `guided_json` — valid JSON guaranteed by the decoder.
- `temperature=0`, a few‑shot example in the prompt.
- Example response schema for one group:
```json
{
  "action": "merge | keep_separate | supersede | summarize",
  "result_content": "final markdown",
  "supersede_ids": ["id1", "id2"],
  "tags": ["..."],
  "reason": "short why"
}
```

## Idempotency and safety
- Consolidation is repeatable: operations apply by id; re‑running the same batch does not corrupt data.
- On a model failure/timeout — the batch is skipped, the source memories are untouched (like dedup: "processing must not drop data").
- A cap on batch size and `MAX_TOKENS` so the window and RAM don't balloon.

## Degradation without the generator
If `MNEMO_GENERATOR=off` (economy mode / no resources):
- near‑dup merge — by cosine + rules (keep the newer/more important one, the old → superseded);
- no summaries/insights;
- memory still works (a plain cheap vector store), just "less smart".
