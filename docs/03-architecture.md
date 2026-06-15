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
The embedder is no longer "ms": the chosen pplx int8 is ~0.4 s for a typical memory and up to seconds for a large
one (see [06-models.md](06-models.md)). Embedding synchronously on the write path would break the cheap‑write
axiom, so it runs **off the hot path**:
- A single **embed worker** in the service drains a queue: `encode(content) → store.set_vector(id)`. The embedder is
  already resident (shared, on‑demand); reads and writes are not blocked while it runs.
- **Immediately lexical, eventually semantic.** FTS5 indexes the text on insert, so a just‑written memory is findable
  by token at once; it enters dense/hybrid search a few seconds later when its vector is stored. A memory is never
  "lost" in the gap — it degrades to lexical‑only, then upgrades.
- **It cannot clog — bounded queue + backpressure.** The queue is bounded; if it fills (bulk seed / a runaway
  writer), new writes **degrade to a synchronous embed**. The worst case is "writes as slow as one sync embed",
  never an unbounded backlog — writes always succeed. In normal use the worker (~2–3 short embeds/s) far outpaces
  the *deliberate* write rate (a few memories/min across agents), so the queue stays near‑empty.
- **Worker hygiene.** Batch short pending memories (amortize); embed long ones singly (a big batch of long inputs
  balloons activation RAM). **Coalesce**: skip a record already superseded by a later `topic_key` write. **Drain
  before idle‑exit**: the on‑demand service does not exit while the queue is non‑empty. Expose **queue depth**
  (`mnemo doctor`) so backlog is observable.
- **MVP note.** Embedding may start **synchronous** (a typical ~0.4 s memory is acceptable, and the rare slow one is
  over the size guideline anyway). Deferred‑with‑backpressure is the designed upgrade, added when write latency is
  measured to matter — backpressure makes that upgrade safe by construction. The store must therefore allow a record
  to exist before its vector (a "pending vector" state), which the FTS5 row already provides for retrieval.

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
