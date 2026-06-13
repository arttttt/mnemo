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
   │ shim   │       │ shim   │     ...      │ shim   │   ← tiny stdio proxies (~0 RAM)
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
              │  │  (ONNX, CPU)   │  │ LanceDB        │  │
              │  │  loaded while  │  │ (embedded,     │  │
              │  │  service alive │  │ on disk)       │  │
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

### 1. Shim (thin MCP proxy)
- What goes into each agent's config (`command: mnemo-shim`).
- A **stdio** process that simply proxies MCP calls to the shared service over HTTP.
- Loads nothing heavy (a few MB). Having 10+ of them is fine.
- Handles **ref‑counting**: on start it brings up the service (if not running); on exit it reports "one less client".
- Alternative to the shim — direct MCP HTTP transport + socket activation (see below and [07-lifecycle-and-ram.md](07-lifecycle-and-ram.md)).

### 2. mnemo service (core, one process)
- **MCP router** (FastMCP): exposes the tools from [05-mcp-api.md](05-mcp-api.md).
- **Write path** (hot, cheap): `remember` → embedding → upsert into the store. **No LLM.**
  Writes are serialized through an internal queue/lock to avoid races.
- **Read path**: `search` → query embedding → ANN/hybrid + payload filter → (optional) rerank.
- **Embedder**: a small ONNX model loaded into the process while the service is alive. CPU, no GPU.
- **Store**: an embedded vector DB (LanceDB). Files in `~/.mnemo/data/`. No separate daemon/Docker.

### 3. Consolidation worker (background)
- Triggered by N new records / idle / schedule.
- Loads the generative model (Qwen3‑4B / Gemma 4 E4B) via llama.cpp **only while running**, then unloads it.
- Does: dedup‑merge, cluster summarization, insight/rule extraction, marking stale records.
- Writes results back to the store. Never blocks the hot path. See [08-consolidation.md](08-consolidation.md).

## Data flows

### Write (hot path — no LLM)
```
remember(content, type, project, scope, related_files, ...)
   → normalize + hash
   → dedup: hash match? cosine > threshold vs neighbors? → merge/skip or new record
   → embedder.encode(content)            # CPU, ms
   → store.upsert(vector, payload)       # through the write queue
   → reply to agent (id)
```

### Read
```
search(query, scope?, project?, type?, filters?)   # scope="all" → cross-project
   → embedder.encode(query)
   → store.query(vector, soft_filter=payload)  # dense (+ sparse, RRF); project = soft boost, not a wall
   → (optional) reranker.rerank(top_k)
   → return top‑N with payload
```

### Session start
```
recall(project)
   → rules (project + __global__, type=rule)
   → recent activity (project, recent_only)
   → pending tasks
   → session_recap (last session)
```

## Concurrency (10+ agents)

- All agents hit **one process** → no "many processes, one file" problem (the main failure mode of SQLite‑based approaches like engram).
- Inside the process: reads run in parallel; writes go through a **single queue/lock** (short, ms‑scale operations).
- Writes are cheap (embed + insert), so even a burst from 10+ agents drains the queue quickly.
- The generator is **not** in the hot path → there is no need for "vLLM continuous batching for 10+ concurrent" here:
  background consolidation is a single batch job that llama.cpp on‑demand handles fine.

## Key decisions (and why)

| Decision | Why |
|---|---|
| Embedded store (LanceDB), not a Qdrant daemon | on‑demand + no always‑on Docker; one process ⇒ the single‑writer lock is not a problem |
| One shared service + shims, not N stdio servers | otherwise N copies of the embedder in RAM and N writers on one file |
| LLM in the background only, transiently | cheap concurrent writes; no RAM "hog"; a small model is enough |
| llama.cpp on‑demand, not a resident vLLM | RAM ~0 when idle; the generator is off the hot path ⇒ vLLM batching is unnecessary |
| Embedder on CPU | does not compete for the GPU; works on 16 GB without a discrete GPU |
