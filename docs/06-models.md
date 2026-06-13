# 06 — Models

Principle: **minimum models**. Only the embedder is mandatory. The generator is a single, small,
background model. Specialist models (NER, reranker) are optional upgrades, not for v1.

> Licenses don't matter to us; where convenient we prefer Apache‑2.0. All options are local, no cloud.

## 1. Embedder (mandatory; hot path; CPU)

Needed while the service is alive. It is not an "LLM" but part of semantic search — tiny, fast, on CPU.

| Model | Size | Dim | Context | Notes |
|---|---|---|---|---|
| **Qwen3‑Embedding‑0.6B** ⭐ | 0.6B | up to 1024 (MRL) | **32K** | long context for large memories; Apache‑2.0; GGUF/ONNX |
| **bge‑small‑en‑v1.5** | 33M | 384 | 512 | ultra‑light, minimal RAM; for short records |
| **embeddinggemma‑300m** | 300M | 768 (MRL 128/256/512) | 2048 | great quality/weight; Matryoshka — trim dim for speed |
| **bge‑m3** | 568M | 1024 | 8192 | dense+sparse+ColBERT in one — handy for hybrid |

**v1 default:** `bge-small-en-v1.5` (minimal RAM, enough for short records) **or**
`Qwen3-Embedding-0.6B` (if memories are long). The dimension is fixed at store init
(changing the embedder = reindex).

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
