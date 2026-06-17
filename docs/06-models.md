# 06 — Models

Principle: **minimum models, each earns its place — and each does only what it is best at.** The embedder
is mandatory (search + candidate selection). Background **consolidation is a staged pipeline** (see
[08-consolidation.md](08-consolidation.md)) that adds three more small models, one per stage: a **reranker**
(routing/dedup), an **NLI** model (contradiction), and the **generator** LLM (the merged text). The
cross-encoders are tiny and cheap; the generator is transient. Further specialists (NER, extractor) stay
optional upgrades.

> **Everything is multilingual** — no language is guaranteed to be English, so every stage must match the
> multilingual embedder. Licenses: we prefer Apache‑2.0 where convenient. All options are local, no cloud.

## 1. Embedder (mandatory)

Needed while the service is alive. It is not an "LLM" but part of semantic search — small and fast.
It serves two jobs: **retrieval** (semantic search — "recall" here means *retrieval quality*, not the `recall(project)`
digest tool, which is the generator's job) and **near-duplicate** detection (for supersede). One embedder is
mandatory and shared by the whole service; different embedders give incompatible vector spaces, so exactly one is
chosen and switching it later forces a full reindex. A trivial hash embedder is used for offline tests; it is not a
real option.

> **Not strictly "hot path".** Retrieval (`search`) needs the embedder synchronously, but the **write** embed is
> **deferred off the hot path** (the chosen pplx is ~0.4 s/memory, not ms) — see
> [03-architecture.md](03-architecture.md#deferred-embedding-async-vector-computation). So a slow embedder is fine
> for writes; it only bounds `search` latency (~200 ms, acceptable for on-demand recall).

**Default: `pplx-embed-v1-0.6b`, int8 ONNX, on CPU.** Chosen 2026-06 after a full benchmark — see
[research/embedder-benchmark.md](research/embedder-benchmark.md) for the evidence. (q4 is the
**fast profile** — switch to it only if recall latency/throughput becomes a measured problem.)

**Requirements (what the chosen embedder had to satisfy):**
- **Local / offline.** Runs entirely on the machine, no cloud call ever.
- **CPU‑feasible.** Must run acceptably on CPU — the 16 GB target machine often has no discrete/CUDA GPU
  (e.g. Apple Silicon). A GPU/Metal may be used if present (faster) but is **not** required.
- **Generous context window.** Comfortably **thousands of words**, not ~512 tokens — a memory is one
  vector, so the window bounds how large one memory may be (see [04-data-model.md](04-data-model.md)).
- **Good semantic quality.** Strong retrieval on paraphrased, conceptual queries.
- **Hybrid‑friendly (ideally).** Plays well with the dense + lexical/sparse hybrid step.
- **Fixed dimension.** The dimension is fixed at store init — changing the embedder later means a reindex.

**Why pplx-0.6b** (full results in [research/embedder-benchmark.md](research/embedder-benchmark.md)):
on our 44-memory / 50-query corpus it is the quality leader (r@1 .72 / MRR .96 / r@5 .95) and the
**only** candidate strong at **paraphrase** (MRR .96) — the dominant real recall pattern, since an agent
asks in different words than a memory was written; Gemma/Granite collapse on paraphrase (.71–.85).
CPU-only is a feature here: it meets the mandatory CPU-feasible requirement with no Metal dependency
(pplx OOMs on torch-MPS and has no MLX build, but doesn't need either).

**Why int8, not q4:** int8 is ≈ lossless vs fp32 (r@1 .72 / MRR .96) and the lightest pplx footprint
(1.1 GB); its cost is latency — ~195 ms/query, fine for an on-demand recall (not a hot path). q4 trades
−3 pp quality for 37 ms (≈ 5× faster) and is the **fast profile** to switch to *if* recall
latency/throughput under heavy concurrency turns out to be a real bottleneck — a one-line precision
change (`onnx/model_q4.onnx`). Start with int8; optimize latency only if measured.

| Variant | Runtime | p50 | RAM (footprint) | r@1 / MRR | Note |
|---|---|---:|---:|---:|---|
| **pplx-embed-v1-0.6b int8** ✅ | CPU | ~195 ms | 1.1 GB | .72 / .96 | default — MIT, dim 1024, 32K ctx, paraphrase .96, ≈ lossless |
| pplx-embed-v1-0.6b q4 | CPU | 37 ms | 1.3 GB | .69 / .94 | fast profile — ~5× faster, −3 pp quality; use if latency-bound |
| granite-embedding-97m-multilingual-r2 q4 | CPU | 3.6 ms | 0.8 GB | .58 / .85 | fallback: hard RAM/latency floor, Apache, multilingual; paraphrase .75 |
| embeddinggemma-300m 8bit | MLX | 7 ms | 3.4 GB | .64 / .91 | fallback: Metal-only target; no CPU path, gated |
| pplx-embed-v1-late-0.6b (ColBERT) | CPU | 124 ms | ~1.5 GB | .68 / .92 | rejected: ~26× storage (multi-vector), no win over dense |

**Deploy contract:** `perplexity-ai/pplx-embed-v1-0.6b` → `onnx/model_quantized.onnx` (int8; `model_q4.onnx`
for the fast profile), onnxruntime **CPU** EP, **no** query/doc prefix, normalize + cosine/RRF. Custom arch
`PPLXQwen3Model` ⇒ `trust_remote_code=True`
with a **pinned revision**. Fixed **dimension 1024** at store init — switching the embedder later means a
full reindex. pplx is Perplexity's bidirectional embedding adaptation of Qwen3-0.6B (the raw Qwen3
embedder scored only r@1 .51, so the quality is the training).

### Content window (over-size handling)

Because one memory is one vector, the embedder's context window is the **hard upper bound on a memory's size**
— this is an embedder concern, not a memory‑layer one. The chosen embedder's adapter **owns and enforces** it:
on input that exceeds the window, `encode()` raises an explicit, actionable error stating the limit and the
actual size. It never silently truncates and never auto‑splits (auto‑split needs an LLM and is a post‑MVP
ingestion task). The write use case simply **surfaces** that error to the caller — already an LLM — which
compresses or splits the content and re‑submits, keeping the write path LLM‑free. With the embedder now
chosen, the concrete limit is pplx's **32K tokens** (`max_position_embeddings = 32768`) — comfortably above
a single focused memory, so the bound rarely bites. The offline hash embedder has no meaningful window and
imposes no limit. The policy ("never truncate") is in [04-data-model.md](04-data-model.md).

## 2. Generator — the generation stage (background; on demand; transient)

A single small instruct LLM, loaded only for the consolidation **generation** stage (Stage 3), then
unloaded. It no longer does routing or classification — cross-encoders do that far better and lighter
(§3). The LLM is reserved for the one task that needs synthesis: writing the merged / summarized record
from a confirmed same-topic cluster. So the bar is **faithful, concise, multilingual generation**, not
tool-use.

Requirements: ≤ ~4B (transient on a 16 GB machine), **multilingual**, local on Apple Silicon (GGUF via
llama.cpp/Metal), permissive license preferred, and — the decisive measured axis — **faithfulness** (no
fabricated facts when merging). Selection is on a local benchmark (see
[research/generator-benchmark.md](research/generator-benchmark.md)) — **no model is committed yet.** The
candidates are multilingual ≤4B instruct models (Apache preferred); smaller models that produced degenerate
or hallucinated output were eliminated. On a RAM-tight machine: the smallest passing candidate, or
`MNEMO_GENERATOR=off` (the pipeline still dedups via the reranker — see
[08-consolidation.md](08-consolidation.md#degradation--economy-mode)).

### Structured-output reliability on a small model
JSON validity is a *solved* problem at any size with **grammar/guided decoding** (llama.cpp GBNF or
`guided_json`) + a **flat** schema (no nested `$defs`) + `temperature=0`. So size buys not "valid JSON" but
the *right values*: faithful merges, no fabricated facts. Keep the prompt focused on the one task and
describe the schema in it (the grammar is not injected into the prompt).

## 3. Consolidation specialists — core, not optional

These are **not** "optional upgrades": the pipeline's classification stages are better *and* lighter as
small cross-encoders than as an LLM (a 107M reranker beat every 3–4B LLM at routing — recall of true
duplicates ~0.91 vs ~0.0, and it cannot hallucinate ids). Both are multilingual and run on CPU via
`sentence-transformers.CrossEncoder`. Evidence: [research/generator-benchmark.md](research/generator-benchmark.md).

| Stage | Model class | Why | Status |
|---|---|---|---|
| routing / dedup | reranker (cross-encoder) | scores seed-vs-candidate → threshold; can't hallucinate ids | small multilingual — **not yet chosen** (benchmarking) |
| contradiction | NLI (cross-encoder) | entailment / contradiction; run BOTH directions (NLI is asymmetric) | small multilingual — **not yet chosen** |

### Still optional (must earn their place)
| Role | Model | When to add |
|---|---|---|
| Entity extraction / dedup | GLiNER2 (<500M, ONNX) | if you need precise entity-merge |
| Structured extraction | NuExtract3 (4B, Apache-2.0) | if you actively parse code/docs into JSON |
| Memory specialist | driaforall/mem-agent (Qwen3-4B-Thinking) | a reasoning layer over markdown memory; expensive (agentic loop) |

## Inference engines

- **Generator (LLM):** **llama.cpp** (`llama-cpp-python`, Metal, GGUF, GBNF grammar) — precise RAM control,
  load for the generation stage → unload. Ollama (`keep_alive=0`) is an alternative; vLLM is not needed
  (single batch background job). For a truly clean RAM reset, run the batch in a killable subprocess.
- **Reranker + NLI (cross-encoders):** `sentence-transformers.CrossEncoder` on CPU — small, fast, and CPU
  gives an accurate RAM number (Metal undercounts mmap'd weights).

## Layout under 16 GB
- Resident: the embedder (always, ~1.1 GB) + the thin service/connector.
- Consolidation window (transient): the **reranker** (~0.1–0.7 GB) and **NLI** (~0.3 GB) run cheaply over
  the candidates; the **generator** (~1.5–3 GB at ≤4B Q4) loads only for the few confirmed clusters, then
  frees immediately.
- If a local coding-agent model runs alongside it dominates RAM — schedule consolidation for machine-idle.
