# 06 — Models

Principle: **minimum models**. Only the embedder is mandatory. The generator is a single, small,
background model. Specialist models (NER, reranker) are optional upgrades, not for v1.

> Licenses don't matter to us; where convenient we prefer Apache‑2.0. All options are local, no cloud.

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

## 2. Generator (background; on demand; transient)

A single small instruct LLM. Loaded only for the consolidation window, then unloaded.

| Model | Size | Context | Notes |
|---|---|---|---|
| **Qwen3‑4B‑Instruct‑2507** ⭐ | 4B | 262K | most battle‑tested for structured/JSON + tool‑use at 4B; Apache‑2.0 |
| **Qwen3.5‑4B‑Instruct** | 4B | 262K | newer family (~Mar 2026); alternative |
| **Gemma 4 E4B** | ~4.5B eff. | 128K | works; slightly weaker at pure tool‑calling than Qwen at this size; Gemma license (ToS) |
| **Qwen3‑1.7B** | 1.7B | — | if 16 GB is under pressure: cheaper, for simple consolidation tasks |

**v1 default:** `Qwen3-4B-Instruct-2507` (GGUF Q4) via llama.cpp on‑demand.
On a RAM‑tight machine — `Qwen3-1.7B` or disable the generator entirely (cosine dedup).

**Why not Gemma 4 by default, although we discussed it:** Gemma 4 works, but for the role "reliable
JSON/tool‑use at 4B", Qwen3‑4B‑Instruct‑2507 is better proven and has no Gemma ToS. We keep
Gemma 4 E4B as a supported alternative.

### Structured‑output reliability on a small model
At 4B, function‑calling/JSON success depends heavily on the "contract": the same model ranges ~7%…100%.
Therefore **mandatory**: a fixed format + grammar/guided decoding (llama.cpp GBNF grammar or
vLLM `guided_json`) + `temperature=0`. A flat JSON schema is more reliable than a nested one
(`$defs` sharply raise the error rate).

## 3. Optional specialist models (upgrades, not v1)

**Opt‑in by default.** Using several small specialist models (NER, reranker, extractor) is fine — but each
must **earn its place** with a measured quality/perf gain, or be enabled explicitly by config. Don't add
models speculatively: the default build is just the embedder + (optional) one generator.

| Role | Model | Where | When to add |
|---|---|---|---|
| Entity extraction / dedup | **GLiNER2** (<500M, ONNX) | CPU | if you need precise entity‑merge without a GPU |
| Candidate reranking | **Qwen3‑Reranker‑0.6B** / bge‑reranker‑v2‑m3 | CPU | if search quality hits a ceiling |
| Structured extraction | **NuExtract3** (4B, Apache‑2.0) | on demand | if you actively parse code/docs into JSON |
| Memory specialist | **driaforall/mem-agent** (Qwen3‑4B‑Thinking) | on demand | as a reasoning layer over markdown memory; expensive (agentic loop) |

## Inference engine for the generator

- **Default: llama.cpp** (via `llama-cpp-python`) — precise RAM control, on‑demand load/unload, GGUF quants, GBNF grammar. Ideal for "load for consolidation → unload".
- **Ollama with `keep_alive=0`** — easier to install; unloads the model after idle. OK as an alternative.
- **vLLM** — NOT needed here: its advantage (continuous batching for many concurrent requests) matters when the LLM is on the hot path; ours is a single batch background job.

## Layout under 16 GB
- On CPU: the embedder (always) + optional GLiNER/reranker.
- The generator: loaded into RAM (or VRAM, if available) only for the consolidation window; Q4‑4B ≈ 3–4 GB, freed immediately.
- If a local model for the coding agent itself runs alongside — it dominates; then move consolidation to machine‑idle and/or use the 1.7B.
