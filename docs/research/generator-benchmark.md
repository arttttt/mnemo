# Research — consolidation models benchmark

Picking the models for the background **consolidation pipeline** (see
[../08-consolidation.md](../08-consolidation.md)). The headline finding reshaped the architecture: the
consolidation worker is **not one LLM call** but a staged pipeline, and the routing/classification stages
belong to small **cross-encoders**, not the LLM. Status: **in progress** — the routing result is decisive;
the generator pick is being finalized.

> Numbers below are from a local bench (a separate workspace) on a corpus of ~109 real memories exported
> from the live store, with candidate pairs built from the store's own pplx embedding neighbours. The
> sample is small (~19 groups / ~24 cases per stage), so treat single-point gaps of ±0.1 as noise — but the
> large gaps (reranker vs LLM) and the eyeballed failures are robust.

## What the benchmark shows — **no model is selected yet**

The only thing settled is the **architecture**: a staged pipeline where routing/classification belong to a
cross-encoder *class*, not an LLM. The specific models are **open** — the tables below are the evidence to
choose from, not a decision.

- **Routing / dedup: the reranker class beats the LLM class.** The best cross-encoder tested hit accuracy
  **0.91** with balanced safe/recall and **zero id-hallucination**; every 3–4B LLM collapsed on the same
  task (recall of true duplicates ~0.0, id-hallucination, broken JSON). So routing is a reranker, not an LLM.
- **Contradiction:** an NLI cross-encoder (entailment/contradiction), run bidirectionally.
- **Generation (merge/summarize):** the only stage that needs an LLM — a small multilingual one; candidates
  are under test, a clean re-run pending.
- **Multilingual is mandatory** everywhere (no language is guaranteed English; the embedder is multilingual).

## Methodology (hard-won)

- **Faithfulness gate = grounding + coherence.** Grounding = every number/identifier in the output appears
  in the inputs (no fabrication). Coherence was added after a 350M model passed grounding with *garbage*
  output (`">>>>"`, `"store store store"`) — junk has no facts to ground, so grounding alone false-passes
  it. Eyeballing samples is essential.
- **Asymmetric errors.** A wrong merge permanently corrupts memory; a missed merge is harmless. So we score
  **SAFE** (of truly-unrelated candidates, the share correctly kept apart — anti over-merge) **separately**
  from **ACT** (of true duplicates, the share caught). A symmetric "accuracy" hides the dangerous direction.
- **Prompt dominates the decision.** A single one-shot prompt biased models toward over-merge; balancing it
  swung SAFE by 0.3–0.9. Faithfulness is prompt-robust; routing/classification is not — which is part of why
  a scored reranker beats a prompted LLM here.
- **Realistic cases.** Routing pairs/groups are the store's actual nearest embedding neighbours, so the
  hard negatives are real look-alikes (e.g. two "competitive landscape" notes at cosine 0.925 under
  different `topic_key`s). `topic_key` is the user's intentional granularity — same-subject-different-topic
  records must stay separate, and that is the hardest, most important case.

## Stage 1 — routing / dedup (classify candidates into same-topic vs unrelated)

Same task, same data; reranker = score seed-vs-candidate then threshold, LLM = emit id lists.

| Model | Type | RAM | acc | SAFE | ACT | id-valid |
|---|---|---:|---:|---:|---:|---:|
| **mmarco-mMiniLMv2-L12** (107M, multilingual) | reranker | 0.67 GB | **0.91** | 0.89 | **0.89** | **1.00** |
| bge-reranker-v2-m3 (568M, multilingual) | reranker | 1.9 GB | 0.83 | 0.91 | 0.61 | 1.00 |
| gte-multilingual-reranker-base (306M) | reranker | 0.8 GB | 0.76 | 0.93 | 0.29 | 1.00 |
| lfm2.5-1.2b | LLM | 0.9 GB | 0.73 | 0.91 | 0.21 | **0.26** |
| qwen3.5-2b | LLM | 1.6 GB | 0.71 | 0.95 | **0.00** | 0.79 |
| granite4.1-3b | LLM | 2.4 GB | 0.68 | 0.90 | **0.08** | 0.95 |
| nemotron / gemma-4 / qwen3-4b | LLM | 2.3–3.7 GB | 0.43–0.57 | — | ≤0.39 | 0.47–0.68 |

The LLMs are uselessly conservative on group classification (ACT ≈ 0 — they put almost nothing in
`same_topic`), hallucinate ids, and break JSON. The reranker, scoring each pair with cross-attention and
thresholding, separates cleanly — and **structurally cannot hallucinate an id** (it only scores given
candidates). English-only rerankers (ms-marco-MiniLM, ettin, gte-modernbert) were excluded regardless of
accuracy. Threshold here is the best operating point (oracle); deployment sets it from a validation split.

## Stage 3 — generation (faithful merge/summarize of a confirmed same-topic cluster)

Preliminary (a clean re-run is pending — large clusters bloated some prompts and added noise). The robust
signal: faithfulness leaders were `granite-4.1-3b`, `qwen3.5-2b`, and the Gemma-4 pair; sub-1B models and
the old Ministral GGUF failed (degenerate / hallucinated output). Strict action-label accuracy was a poor
metric (merge ≈ supersede); grounding + coherence + the eyeball are what matter.

## Open items

- Clean Stage-3 re-run (cap cluster size; add non-English cases) on the multilingual LLM finalists → final
  generator pick.
- Wire the contradiction stage (`mDeBERTa-v3-xnli`, bidirectional) and measure precision/recall.
- Calibrate the reranker threshold on held-out data (the bench uses an oracle threshold).
