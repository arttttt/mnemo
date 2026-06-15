# Embedder selection benchmark (2026-06)

The study that chose the default embedder. **Outcome: `pplx-embed-v1-0.6b`, int8 ONNX, on CPU**
(q4 is the fast profile if latency-bound). Summary and the deploy contract live in
[../06-models.md](../06-models.md); this is the full evidence.

## Why this study

One embedder is mandatory and shared by the whole service; it serves two jobs — semantic **recall**
and **near-duplicate** detection (for supersede). Different embedders produce incompatible vector
spaces, so exactly one model must be picked, and switching it later forces a full reindex. The pick
therefore had to be made on measured evidence, against the requirements in
[../06-models.md](../06-models.md) (local/offline, CPU-feasible, generous window, good semantic
quality, fixed dimension).

## Method

- **Corpus:** 44 active memories from the real store + 85 memories (active + superseded) for near-dup.
- **Queries:** 50 hand-written, tagged by type — **paraphrase** (semantic restatement),
  **question**, **keyword**, **topic**. This split is the point: it shows *how* a model retrieves,
  not just an average.
- **Metrics:** recall@1/5/10 and MRR@10 per query type; near-dup MRR@10; per-query p50 latency;
  peak RAM. Reported headline = OVERALL r@1 / MRR, plus per-type MRR.
- **Runtimes, deployment-realistic:** ONNX via `onnxruntime` **CPU** EP (the CPU-floor target);
  PyTorch **MPS** and **MLX** for Apple Metal. Each variant ran in its own subprocess so RAM
  reflects only that model. We tested **multiple quantizations** per model (fp32 / int8 / quint8 /
  q4) where builds exist.
- **RAM = vmmap "Physical footprint"** (== Activity Monitor's Memory column). This matters — see
  [the measurement note](#ram-measurement-the-only-trustworthy-number-is-physical-footprint).

## Candidates

Surveyed across HuggingFace and non-HF sources. Finalists carried into the full matrix:
**pplx-embed-v1-0.6b** (dense), **embeddinggemma-300m**, **granite-embedding-97m-multilingual-r2**,
**granite-embedding-311m-multilingual-r2**. Cut earlier: **Harrier-OSS-270m** (mediocre; worst
recall@5 in the field), **Qwen3-Embedding-0.6B** (raw, r@1 .51 — much weaker than the pplx
derivative built on it).

## Results — latency / RAM / overall quality

RAM = physical footprint. ★ = the deployable pick per model (best quality/cost trade).

| Variant | Runtime | p50 | RAM | r@1 | r@5 | MRR | near-dup |
|---|---|---:|---:|---:|---:|---:|---:|
| **pplx int8** ★ | CPU | 197 ms | **1.1 GB** | 0.72 | 0.95 | 0.96 | 0.98 |
| pplx q4 | CPU | 37 ms | 1.3 GB | 0.69 | 0.95 | 0.94 | 0.99 |
| pplx fp32 | CPU | 193 ms | ~1.5 GB | 0.72 | 0.95 | 0.96 | 0.98 |
| pplx | Metal (torch) | — | — | — | — | — | **OOM** |
| pplx | MLX | — | — | — | — | — | **arch unsupported** |
| Gemma 8bit ★ | MLX | 7.2 ms | 3.4 GB | 0.64 | 0.92 | 0.91 | 0.98 |
| Gemma 4bit | MLX | 7.4 ms | 3.3 GB | 0.65 | 0.91 | 0.90 | 0.97 |
| Gemma fp32 | Metal (torch) | 20 ms | 3.4 GB | 0.60 | 0.93 | 0.86 | 0.99 |
| Granite-97m q4 ★ | CPU | 3.6 ms | **0.8 GB** | 0.58 | 0.88 | 0.85 | 0.98 |
| Granite-97m fp32 | CPU | 4.1 ms | 0.9 GB | 0.54 | 0.91 | 0.84 | 0.98 |
| Granite-97m quint8 | CPU | 2.6 ms | 0.6 GB | 0.54 | 0.85 | 0.77 | 0.98 |
| Granite-97m int8 | CPU | 2.3 ms | 0.6 GB | 0.46 | 0.84 | 0.75 | 0.95 |
| Granite-97m fp32 | Metal (torch) | 9.0 ms | 1.6 GB | 0.54 | 0.91 | 0.84 | 0.98 |
| Granite-311m q4 ★ | CPU | 12 ms | 1.4 GB | 0.59 | 0.93 | 0.85 | 0.97 |
| Granite-311m fp32 | CPU | 11 ms | 1.6 GB | 0.60 | 0.89 | 0.85 | 0.97 |
| Granite-311m quint8 | CPU | 6.6 ms | 1.0 GB | 0.52 | 0.88 | 0.83 | 0.99 |
| Granite-311m int8 | CPU | 5.9 ms | 0.9 GB | 0.55 | 0.90 | 0.82 | 0.97 |
| Granite-311m fp32 | Metal (torch) | 13 ms | 3.0 GB | 0.60 | 0.89 | 0.85 | 0.97 |
| Qwen3-0.6B | Metal (torch) | 39 ms | 3.2 GB | 0.51 | 0.88 | 0.81 | 0.96 |

## Results — quality per query type (MRR@10)

| Variant | paraphrase | question | keyword | topic | near-dup |
|---|---:|---:|---:|---:|---:|
| pplx fp32 / int8 | **0.96** | 0.95 | 1.00 | 0.92 | 0.98 |
| pplx q4 | **0.93** | 0.95 | 0.95 | 0.92 | 0.99 |
| Gemma 8bit | 0.85 | 0.91 | 1.00 | 1.00 | 0.98 |
| Gemma 4bit | 0.83 | 0.91 | 1.00 | 1.00 | 0.97 |
| Granite-97m q4 | 0.75 | 0.87 | 1.00 | 1.00 | 0.98 |
| Granite-97m fp32 | 0.74 | 0.87 | 1.00 | 0.92 | 0.98 |
| Granite-311m q4 | 0.75 | 0.95 | 0.95 | 0.92 | 0.97 |
| Granite-311m fp32 | 0.77 | 0.90 | 0.95 | 0.92 | 0.97 |

## Findings

1. **Paraphrase is the differentiator.** It is the hardest, purely-semantic type — and the dominant
   real recall pattern: an agent asks in different words than the memory was written. **Only pplx
   handles it** (MRR .93–.96); Gemma .83–.85, Granite .71–.77. Gemma and Granite are
   topic/keyword specialists (topic up to 1.00) but collapse on paraphrase. For a *memory* system,
   paraphrase strength is decisive, and it is why pplx leads OVERALL.
2. **pplx is CPU-only — and that's fine.** torch-MPS OOMs (bidirectional 0.6B hits the watermark cap
   on 16 GB) and there is no working MLX build (custom `bidirectional_pplx_qwen3` arch is
   unsupported by `mlx_embeddings`). CPU is its home: int8 (default) is ≈ lossless at ~195 ms / 1.1 GB;
   q4 is the fast profile at 37 ms / 1.3 GB for a −3 pp quality cost. CPU-only satisfies the mandatory CPU-feasible
   requirement with no Metal dependency.
3. **Quantization is model-specific.** int8 ≈ lossless for pplx; for Granite, the *community* q4
   builds are best (and beat official fp32: 97m q4 r@1 .58 vs fp32 .54), while official int8/quint8
   regress (97m int8 r@1 .46). Use Granite q4, not int8/quint8.
4. **Gemma is Metal-only in practice** (no usable CPU-ONNX path — its ONNX export lacks the ST
   pooling/Dense modules), and its MLX quants beat its own fp32-torch (.64–.65 vs .60). Strong, but
   3.3–3.4 GB and no CPU fallback.
5. **Granite on Metal is not faster than on CPU** (97m: 9 ms Metal vs 2–4 ms CPU) — for small models
   the Metal overhead doesn't pay off.
6. **near-dup is non-differentiating** (0.95–0.99 for everyone), so recall quality decides the pick.
7. **pplx lineage:** it is Perplexity's embedding adaptation of **Qwen3-0.6B** (bidirectional
   conversion + diffusion continued-pretraining + embedding fine-tune, MIT). Proof the value is the
   training, not the base: raw Qwen3-Embedding-0.6B scored r@1 .51 vs pplx .72.

## Decision

**`pplx-embed-v1-0.6b`, int8 ONNX, CPU** — best overall quality (≈ lossless vs fp32, r@1 .72 /
MRR .96), the only model that handles paraphrase, lightest pplx footprint (1.1 GB), MIT, dim 1024,
32K context. Used for both recall and near-dup. Its cost is latency (~195 ms/query) — acceptable for
on-demand recall; **q4** is the one-line fast profile (37 ms, −3 pp quality) to switch to *only if*
recall latency/throughput under concurrency proves to be a measured bottleneck. Start with int8,
optimize latency only if it bites. Deploy contract in [../06-models.md](../06-models.md).

**Fallbacks (only if):** a hard sub-GB / sub-5 ms floor, or a pure-Apache / no-`trust_remote_code`
constraint, or a multilingual need → **granite-embedding-97m-multilingual-r2 q4** (0.8 GB, 3.6 ms,
but paraphrase .75). A target that is *guaranteed* Apple-Silicon-with-Metal and wants topic/keyword
strength → **embeddinggemma-300m 8bit MLX** (7 ms, 3.4 GB) — not the default: no CPU fallback, gated
license.

## Appendix — late-interaction (ColBERT), tested and rejected

`pplx-embed-v1-late-0.6b` is a token-level late-interaction (MaxSim/ColBERT) sibling of the dense
model. Tested via PyLate for completeness:

| Variant | p50 | RAM | store / doc | r@1 | MRR | paraphrase |
|---|---:|---:|---:|---:|---:|---:|
| pplx-late fp32 | 124 ms | ~1.5 GB | **213 × 128** | 0.68 | 0.92 | 0.88 |
| pplx-late int8 | 96 ms | 1.2 GB | 213 × 128 | 0.54 | 0.84 | 0.75 |
| pplx **dense** q4 | 37 ms | 1.3 GB | 1 × 1024 | 0.69 | 0.94 | 0.93 |

It does **not** beat the dense model on our data (r@1 .68 vs .69; paraphrase *worse*, .88 vs .93),
yet it stores **~213 token-vectors per document** (≈ 26× the dense single vector) and needs MaxSim
scoring + a ColBERT index (PLAID/PyLate) — incompatible with the single-vector SQLite + sqlite-vec
store and against the RAM budget for 10+ agents. Rejected. (No public quant builds exist; the int8
here was made via torch dynamic quantization and it crushed quality, .68 → .54.)

## RAM measurement — the only trustworthy number is "Physical footprint"

Three ways to read a process's memory disagree, verified on pplx-q4: `ru_maxrss` = 2355 MB,
`ps rss` = 2246 MB, **vmmap "Physical footprint" = 1.4 GB** (== Activity Monitor). `ru_maxrss` and
`ps rss` over-count by ~30–45% because RSS includes mmap'd model-weight file pages (file-backed,
reclaimable) and shared framework pages. **Always report vmmap Physical footprint.** Caveat: for
PyTorch loading fp32 safetensors the weights are mmap'd, so footprint *under*-counts them (add the
resident weight size back); `onnxruntime` loads weights into memory, so ONNX footprints are complete.
All RAM figures above are physical footprint.
