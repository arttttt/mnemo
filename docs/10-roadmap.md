# 10 — Roadmap

The MVP is **all of Phases 0–4**. Everything under "Post‑MVP" lands **after Phase 4**.

Each phase is broken into small, independently verifiable steps. Every step has a **Done when** —
a concrete, checkable condition (a passing test, an observable behavior). A step should be small enough
to ship and review on its own.

---

## Phase 0 — Walking skeleton ✅ (done, merged)
An agent can store and search memory locally via MCP.
- [x] Clean‑Architecture scaffold (domain / application / adapters / infrastructure), `uv`, src‑layout, CLI.
- [x] In‑memory + JSON store behind `MemoryRepositoryPort`; hash + fastembed embedders.
- [x] FastMCP `remember` / `search`; CLI `store` / `search` / `stats`.
- [x] Unit + integration tests; offline by default.

---

## Phase 1 — Memory layer (the core)
**Goal:** typed, scoped, persistent memory with hybrid search, `recall`, sessions, deletion, and the
deterministic part of evolution — correct under 10+ concurrent agents.

- **1.1 LanceDB repository adapter** (`LanceMemoryRepository` implements `MemoryRepositoryPort`).
  *Done when:* it passes the same integration suite as the in‑memory store (round‑trip, ranking) under
  `MNEMO_STORE=lancedb`; the in‑memory store stays as the offline test double.
- **1.2 Hybrid search** (dense + lexical/FTS, merged via RRF) + payload filters (`type`, `tags`, `related_files`, recency).
  *Done when:* a conceptual query and an exact term (e.g. a function name) both retrieve the right record; filters work; tests.
- **1.3 Finalize the write fields** (no `importance`) across domain, MCP, CLI.
  *Done when:* `remember` stores `content/type/scope/project/related_files/tags/topic_key`; MCP and CLI agree; tests.
- **1.4 Exact‑dup + `topic_key` upsert on write** (no near‑dup suppression).
  *Done when:* identical content bumps `duplicate_count` (no new row); reusing a `topic_key` supersedes the prior
  record (old → `superseded`, linked); near‑similar content coexists; tests.
- **1.5 Deletion tools** `delete(ids)` / `clear(project)` / `purge()` (agent MCP + CLI, hard).
  *Done when:* each removes exactly the targeted records; CLI + MCP both work; tests.
- **1.6 Sessions + session‑tracking.**
  *Done when:* a session row is recorded (`id/project/started_at/ended_at/summary`); new memories carry `session_id`.
- **1.7 `recall(project)` MCP tool.**
  *Done when:* it returns rules (project + global) + recent activity + `session_recap`; tested end‑to‑end.
- **1.8 Deterministic links + provenance** (`supersedes` via `topic_key`; `derived_from` via optional `source_ids`).
  *Done when:* a topic_key upsert writes a `supersedes` edge; `source_ids` writes `derived_from`; edges carry
  `provenance`; retrievable; schema reserves a generic `links` shape and keeps temporal fields forward‑compatible; tests.
- **1.9 Concurrency: one shared process serves 10+ agents** (internal write queue/lock).
  *Done when:* a test with ≥10 parallel writers shows no lost writes and no "database is locked".

**Phase 1 done when:** FR‑1..FR‑13 hold; the 10‑agent concurrency test passes; LanceDB is the backend.

---

## Phase 2 — On‑demand lifecycle
**Goal:** nothing resident; one shared service spins up under load and exits on idle.

- **2.1 Shared service + thin stdio shim** (agents → one HTTP/streamable service; one embedder loaded).
  *Done when:* several agents talk to a single service process via the shim.
- **2.2 Ref‑count + grace shutdown.**
  *Done when:* the service exits N minutes after the last client; a new client within the grace window keeps it alive; test.
- **2.3 Socket activation** (launchd plist + systemd socket/service).
  *Done when:* the first connection starts the service, idle exits it, the OS re‑listens — verified on macOS and Linux.
- **2.4 `mnemo init <client>`** writes the MCP config + installs activation.
  *Done when:* one command configures Claude Code / Cursor.
- **2.5 RAM budget verification.**
  *Done when:* measured ~0 idle and ~1 GB active (NFR‑8).

---

## Phase 3 — Background consolidation (concurrent from the start)
**Goal:** a background worker improves memory off the hot path, designed for concurrency from day one.

- **3.1 Generator adapter** (llama.cpp on‑demand load/unload; vLLM option for concurrent serving) with GBNF/guided‑JSON.
  *Done when:* the model loads on a trigger and unloads after; output is schema‑valid; RAM is transient.
- **3.2 Concurrent consolidation engine** (a worker pool over batches — **not** one serial pass).
  *Done when:* batches process in parallel with backpressure; the hot path is never blocked.
- **3.3 Triggers** (by volume / idle / manual `consolidate`).
  *Done when:* it fires per `MNEMO_CONSOLIDATE_EVERY`, on idle, and on demand.
- **3.4 Operations:** near‑dup merge, cluster summarize, insight extraction, **contradiction flagging (flag‑only)**.
  *Done when:* each is implemented with guided‑JSON, idempotent and failure‑isolated; nothing is auto‑invalidated.
- **3.5 Semantic links (background):** cosine top‑k → typed link *proposals* (`related_to` / `contradicts`).
  *Done when:* proposals are stored with `provenance=llm` and are **not** auto‑applied.
- **3.6 Degradation:** `MNEMO_GENERATOR=off` path (cosine/rules only).
  *Done when:* memory still works with the generator disabled.

---

## Phase 4 — Polish & optional upgrades
- **4.1** Optional reranker + GLiNER entity‑extraction (opt‑in, off by default).
- **4.2** Export / import (portability).
- **4.3** Packaging (`uvx` / `pipx`) + per‑IDE install guides.
- **4.4** Air‑gapped mode (pre‑seeded model cache).
- **4.5** `tasks` API (optional).
*Done when:* each shipped item has its own test/guide; defaults stay lean.

---

## Post‑MVP (after Phase 4)
- **Full bi‑temporal validity model** — transaction‑time (`created_at`/`expired_at`) + valid‑time
  (`valid_from`/`valid_to`), point‑in‑time queries, retro‑corrections. Done **in full** (no half‑measures);
  the MVP schema is forward‑compatible so the four timestamps add as nullable fields.
- **`importance`** reintroduced — ranking blend (relevance + importance + recency) + decay + optional auto‑scoring.
- **Revision tooling** for flagged contradictions (review UX); the human/agent decides — the system never auto‑invalidates.
- **`session` scope** (transient session‑keyed memory).
- Deferred indefinitely: knowledge graph / multi‑hop; web dashboard; document/PDF ingestion; multi‑user / RBAC / cloud sync.

---

## Definition of done for the MVP
Phases 0–4 complete; runs on a 16 GB machine with 10+ agents, strictly offline, on‑demand (no resident
daemon, no Docker), on LanceDB; an agent works through `recall` / `remember` / `search` / `delete`.
