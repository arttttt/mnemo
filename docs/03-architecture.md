# 03 — Architecture

## Overview

A single core — **one service process, started on demand** — with embedded storage and an
embedded embedder. Agents connect to it through thin shims. The heavy generative model lives
separately and only runs in the background.

```
┌─────────────┐  ┌─────────────┐         ┌─────────────┐
│  Agent A    │  │  Agent B    │   ...   │  Agent N    │   (10+)
│ (Claude Code│  │ (Cursor)    │         │             │
│  / Cursor)  │  │             │         │             │
└──────┬──────┘  └──────┬──────┘         └──────┬──────┘
       │ MCP            │ MCP                   │ MCP
   ┌───▼────┐       ┌───▼────┐              ┌───▼────┐
   │mnemo-  │       │mnemo-  │     ...      │mnemo-  │   ← thin stdio proxies (~40 MB each,
   │ mcp    │       │ mcp    │              │ mcp    │      no embedder/store)
   └───┬────┘       └───┬────┘              └───┬────┘
       └────────────────┼───────────────────────┘
                        │ HTTP/streamable‑http (localhost)
              ┌─────────▼──────────────────────────────┐
              │        mnemo service (ONE process)      │  ← starts on demand,
              │  ┌───────────────┐  ┌────────────────┐  │     exits on grace timer
              │  │  MCP router    │  │ write queue /  │  │
              │  │  (FastMCP)     │  │ async handlers │  │
              │  └──────┬─────────┘  └───────┬────────┘  │
              │  ┌──────▼─────────┐  ┌───────▼────────┐  │
              │  │  Embedder      │  │  Store         │  │
              │  │  (ONNX, CPU)   │  │ SQLite + vec   │  │
              │  │  loaded while  │  │ + FTS5         │  │
              │  │  service alive │  │ (on disk)      │  │
              │  └────────────────┘  └───────┬────────┘  │
              └──────────────────────────────┼───────────┘
                                             │ read/write
                                    ┌────────▼─────────┐
                                    │  ~/.mnemo/data/  │  files on disk
                                    └──────────────────┘

              ┌──────────────────────────────────────────┐
              │   Consolidation worker (background)       │
              │   loads llama.cpp + Qwen3‑4B (GGUF)       │  ← transient: load → run → unload
              │   dedup‑merge / summary / insights        │
              └──────────────────────────────────────────┘
```

## Components

### 1. Connector (`mnemo-mcp`, thin MCP proxy)
- What goes into each agent's config (`command: mnemo-mcp`) — the agent‑facing command is unchanged; only its
  internals became a proxy.
- A **stdio** process that proxies MCP calls to the shared service over streamable‑http. No embedder/store —
  ~40 MB (Python + the MCP SDK), one per agent, living in the agent's own process tree.
- **Starts the service** on launch if it is not up (single‑spawn lock + readiness poll — see
  [07-lifecycle-and-ram.md](07-lifecycle-and-ram.md)); the service counts live connectors (per‑run `flock` markers
  the kernel frees on death) and idle‑exits on grace.
- **Owns the run's session id** (one per connector run) and sends it to the service as request metadata, so the
  service stamps provenance without inventing it.

### 2. mnemo service (core, one process)
- **MCP router** (FastMCP): exposes the tools from [05-mcp-api.md](05-mcp-api.md).
- **Write path** (hot, cheap): `remember` → insert + lexical index → **enqueue embed**. **No LLM, no embed on
  the hot path** — the vector is computed off it (see [Deferred embedding](#deferred-embedding-async-vector-computation)).
  Writes are serialized through an internal queue/lock to avoid races.
- **Read path** (retrieval): `search` → query embedding → ANN/hybrid + payload filter → (optional) rerank.
- **Embedder**: the ONNX model (pplx int8) loaded into the process while the service is alive. CPU. Powers both
  `search` and the background embed worker.
- **Store**: an embedded SQLite database (`sqlite-vec` for vectors + FTS5 for lexical). One file in `~/.mnemo/data/`. No separate daemon/Docker. See [adr/0001-storage-engine.md](adr/0001-storage-engine.md).

### 3. Consolidation worker (background)
- Triggered by N new records / idle / schedule.
- Loads the generative model (Qwen3‑4B / Gemma 4 E4B) via llama.cpp **only while running**, then unloads it.
- Does: dedup‑merge, cluster summarization, insight/rule extraction, marking stale records.
- Writes results back to the store. Never blocks the hot path. See [08-consolidation.md](08-consolidation.md).

## Data flows

### Write (hot path — no LLM, no embed)
```
remember(content, type, project, scope, related_files, ...)
   → normalize + hash
   → exact-dup: hash match? bump counter & skip  |  topic_key match? supersede (upsert)
              | else: insert record + FTS5 index
   → enqueue embed                       # the vector is computed off the hot path
   → reply to agent (id)                 # milliseconds — no model on this path
```
Only the cheap, deterministic part is synchronous (hash‑dedup, `topic_key` supersede, insert + lexical index).
The embedding is **deferred** — see [Deferred embedding](#deferred-embedding-async-vector-computation). The record
is lexically searchable (FTS5) immediately; it joins dense/hybrid search once its vector lands.

### Deferred embedding (async vector computation)
The embedder is no longer "ms": pplx int8 is ~0.4 s for a typical memory (seconds for a large one), so embedding on
the write path would break the cheap‑write axiom. It is **deferred off the hot path**, and the whole design follows
one insight: **the database is the durable queue** — a memory with no vector *is* a pending job, so there is **no
separate queue store** and recovery is free.

**Write does only the cheap part.** `remember` (the use case) inserts the record + FTS5 index with a **pending (null)
vector**, then asks the scheduler to embed it, and returns in ms. The record is **lexically searchable immediately**;
it joins dense/hybrid search once its vector lands — never "lost", degrades to lexical‑only then upgrades.

**Embedding scheduler — a port, called by the use case.** Signalling lives in the **application layer, not the
repository** (the repo stays pure persistence). The use case owns "store, then schedule".
- `SyncEmbeddingScheduler` — embeds inline (`content_for(id) → encode → set_vector`). Used by the **CLI** (one‑shot
  process) and offline tests.
- `AsyncEmbeddingScheduler` — used by the **service**: `schedule(id)` just **notifies** (the row is already pending
  in the DB); a pool of worker threads drains the pending rows.

**Worker loop — event‑driven, no polling.**
```
loop:
  while (ids := repo.next_unembedded(limit)):       # drain to empty
    for id in ids:
      if repo.has_vector(id): continue              # idempotent skip
      repo.set_vector(id, embedder.encode(repo.content_for(id)))   # upsert
  with cond:
    while repo.pending_count() == 0 and not stopping:  # predicate under the lock
      cond.wait()
```
- **No poll‑fallback.** Every source of work is a known trigger that notifies (`remember`, a migration). The worker
  drains to empty before sleeping (work arriving mid‑batch is picked up next iteration) and checks the predicate
  **under the lock** before waiting, so a lost wakeup is impossible. Ordering is strict: **commit the insert, then
  notify.**
- **Recovery is free.** The worker's first pass drains any pre‑existing pending rows (left by a crash or a
  migration) — no special re‑enqueue.

**Properties & config.**
- **Cannot clog — backpressure.** Bounded by pending count: when `pending_count() >= MNEMO_EMBED_QUEUE_MAX`,
  `schedule` **embeds synchronously** instead of deferring. Worst case is "as slow as one sync embed", never an
  unbounded backlog; writes always succeed. In normal use the worker far outpaces the *deliberate* write rate, so
  the backlog stays near‑empty.
- **RAM bound = concurrency.** `MNEMO_EMBED_WORKERS` (default **1**) = how many encodes run at once. At 1, RAM = one
  encode (onnxruntime already uses all cores for it). Higher drains faster but multiplies activation RAM (a set of
  long inputs in flight is the balloon trap), so raise it deliberately. No batching.
- **Failure → retry, capped (give‑up tracking is in‑memory by design).** A failed encode is retried up to
  `MNEMO_EMBED_MAX_RETRIES` (default 3); after that the memory is marked embed‑failed (stays lexical‑only) and
  logged. A memory deleted before processing is skipped. The give‑up set is held **in memory, not persisted to the
  DB** — deliberately: a *permanent, per‑row* encode failure is essentially impossible (the embedder is a local
  deterministic model, over‑window content is already rejected at write time and truncated by the encoder anyway,
  and the tokenizer is total over strings). Real encode failures are **transient and global** (model load, OOM,
  disk) — they hit every row and self‑heal, via the retry/backoff or on the next service start, where the in‑memory
  state resets and the still‑pending rows are re‑attempted. So the bookkeeping can only grow within a single uptime
  under a rare transient outage and clears on the regular idle‑exit; persisting it would instead make a transient
  failure *stick* across restarts. Intended behaviour, not a leak to fix.
- **Superseded jobs still run.** A memory superseded after scheduling keeps its embed job — it is kept as history and
  can be retrieved, so its vector is still computed (no coalescing).
- **Idle‑exit drain.** The on‑demand service does not idle‑exit while work remains: it waits for
  `pending_count() == 0 and in_flight == 0` (a job is "in flight" once a worker has taken it). Observable via
  `mnemo doctor`.
- **MVP.** Embedding may start synchronous (a typical ~0.4 s memory is acceptable); the async path above is the
  upgrade, safe by construction (backpressure + DB‑as‑queue).

Store support this needs: `add(memory, vector=None)` (pending), `set_vector(id, v)` (upsert), `has_vector(id)`,
`content_for(id)`, `next_unembedded(limit)`, `pending_count()`.

### Read
```
search(query, scope?, project?, type?, filters?)   # scope="all" → cross-project
   → embedder.encode(query)
   → store.query(vector, filter=payload)  # dense (+ sparse, RRF); project = hard filter (project + global), scope="all" widens
   → (optional) reranker.rerank(top_k)
   → return top‑N with payload
```

### On‑demand retrieval (no session‑start bundle)
> **Terminology.** "Retrieval" / `search` = finding relevant memories by meaning (the **embedder's** job). This is
> distinct from the **`recall(project)` digest** — an aggregated session/project *summary* — which needs text
> synthesis (the **generator's** job) and is post‑MVP. See [05-mcp-api.md](05-mcp-api.md). When the embedder docs
> say "recall", they mean retrieval quality, not the digest tool.

There is no aggregated session‑start call in the MVP — the `recall` **digest** is post‑MVP (a useful one needs LLM
synthesis, kept off the read path). The agent retrieves on demand with `search`:
```
search(project, type="rule")       → active rules (project + __global__)
search(project, type="progress")   → where it left off
search(project, "<question>")      → relevant decisions / notes
```

## Concurrency (10+ agents)

This is an **architecture‑level** concern, not a store one — it is decided and validated *here*, against the
target topology (one shared process + thin shims), not in the memory layer. (Today's transitional setup is the
opposite: one `mnemo-mcp` process per agent; the model below describes the target, and the concurrency work
waits until that target is built.)

- All agents hit **one process** → no "many processes, one file" problem (the failure mode of process‑per‑agent
  setups on a shared file) — independent of the store engine.
- Inside the process: reads run in parallel; writes go through a **single queue/lock** (short, ms‑scale operations).
- Writes are cheap (**insert only — embed is deferred**), so even a burst from 10+ agents drains the write queue
  quickly; the embed queue then drains in the background (bounded + backpressured, so it cannot grow without bound).
- The generator is **not** in the hot path → no need for "vLLM continuous batching for 10+ concurrent" here:
  background consolidation is a single batch job that llama.cpp on‑demand handles fine.
- **Done when:** a stress test with ≥10 concurrent agents shows zero lost writes and no lock errors, with reads
  running alongside. (Validates the architecture; this was formerly tracked as a memory‑layer step.)

## Key decisions (and why)

| Decision | Why |
|---|---|
| Embedded store (SQLite + `sqlite-vec` + FTS5), not a Qdrant daemon | on‑demand + no always‑on Docker; one process ⇒ the single‑writer lock is not a problem; SQLite fits the mutable typed core (see [adr/0001-storage-engine.md](adr/0001-storage-engine.md)) |
| One shared service + shims, not N stdio servers | otherwise N copies of the embedder in RAM and N writers on one file |
| LLM in the background only, transiently | cheap concurrent writes; no RAM "hog"; a small model is enough |
| llama.cpp on‑demand, not a resident vLLM | RAM ~0 when idle; the generator is off the hot path ⇒ vLLM batching is unnecessary |
| Embedder on CPU | does not compete for the GPU; works on 16 GB without a discrete GPU |
