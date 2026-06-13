# 08 — Background consolidation

The goal is to move all the "smart" processing off the hot path into the background, so that writes
stay cheap and the small model works in rare batches (which is why a small model suffices, and why
there is no RAM "hog").

## What consolidation does

1. **Near‑dup merge.** Merges near‑duplicates and paraphrases that the cosine threshold on write only flagged as candidates.
2. **Cluster summarization.** A group of small same‑topic records → one compressed memory (keeping the originals as inactive).
3. **Insight extraction.** Recurring patterns/gotchas → a `learning` record (cross‑project ones → `__global__`).
4. **Staleness.** Marks records `superseded`/`inactive` that contradict newer ones (with a reason).
5. **(MAY) Update the session summary** for `session_recap`.
6. **(planned) Importance scoring.** Re‑score `importance` consistently (heuristic and/or the small model) —
   it is not set automatically on write today (the caller supplies it, default 0.5).

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
            (input: a set of memories; output: {action, merged_content, supersede_ids, tags, importance})
        → apply to the store (merge/supersede/inactivate/insert) through the write queue
        → unload the generator                                # ← RAM freed
        → write a consolidation log
```

The generator lives only for the job's duration. Since this is **a single batch process**, there is
no concurrent inference → llama.cpp on‑demand is sufficient (vLLM/continuous batching unnecessary).

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
  "importance": 0.7,
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
