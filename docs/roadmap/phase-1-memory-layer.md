# Phase 1 — Memory layer (the core)

**Goal:** typed, scoped, persistent memory with hybrid search, session tracking, deletion, and the
deterministic part of evolution — correct under 10+ concurrent agents.

Each step is **Why** (the requirement and the reasoning behind it) · **What** (exactly what to build, including
what *not* to do) · **Done when** (verifiable) · **Depends on**.

> **Status.** Done: 1.1 (LanceDB), 1.2 (hybrid + filters), 1.3 (no `importance`), 1.4 (dedup / `topic_key`
> upsert), 1.6 (deletion) — 1.3 / 1.4 / 1.6 landed with the Phase‑0 skeleton. Remaining: 1.7 (session
> tracking), 1.9 (links + provenance), 1.10 (concurrency). **1.5** (over‑window) moved to the embedder
> boundary — see [06-models.md](../06-models.md). **1.8** (`recall`) deferred to [post‑MVP](post-mvp.md).
> Step numbers are kept stable to avoid churn.

---

### 1.1 LanceDB store backend

**Why.** Phase 0 ships an in‑memory + JSON store on purpose — a placeholder so the core could be built and tested
with zero heavy dependencies. The real product needs two things the JSON store can't give: genuine semantic search
that stays fast as memory grows (approximate nearest‑neighbour, not a brute‑force scan), and hybrid dense+lexical
retrieval. We committed to **LanceDB as the single embedded store** — embedded (a directory of files, not a server,
no Docker daemon), real ANN, with built‑in full‑text search for the hybrid step. No second database, no graph DB.
The store sits behind `MemoryRepositoryPort`, so this is an additive adapter; domain and use cases don't change.

**What.** Implement the LanceDB‑backed store against the existing port, persisting each memory's vector plus its
full payload (type, scope, project, tags, related_files, hash, status, topic_key, links, timestamps). It is selected
at startup by config; the in‑memory store stays available as the offline/test backend so unit and most integration
tests keep running with no downloads. All data lives under the single `~/.mnemo/data/` directory.

**Migration of existing data.** Anyone dogfooding the tool already has memories in the JSON store, and switching
backends must **lose nothing**. Provide a **one‑time, idempotent migration** that reads the existing JSON store and
writes every record (re‑embedding as needed) into LanceDB. Expose it as a **menu action in the dev helper script**
so a developer can run it explicitly; it must be safe to run twice (no duplicates) and must not delete the JSON
source (the original stays as a fallback until the developer removes it).

**Done when.**
- The LanceDB store passes the **same store‑contract tests** as in‑memory: write → find by hash, similarity
  ranking, list, and persistence across a reopen.
- Choosing the backend is one config switch; nothing else in the code changes.
- The in‑memory backend still works for offline tests.
- Running the migration moves all existing JSON memories into LanceDB with nothing lost; running it again is a
  no‑op (idempotent); the dev script offers it as a menu item.

**Depends on:** Phase 0.

---

### 1.2 Hybrid search + filters

**Why.** Developer queries are two kinds at once: conceptual ("how do we handle auth errors") and exact
("handleAuthCallback", an error code). Pure vector search is good at the first and **misses** the second; pure
keyword is the reverse. We require both, so retrieval must **blend** them. On top, retrieval must honor the agreed
**soft scoping**: default = current project + global (so cross‑cutting rules and lessons always surface), with
`scope="all"` for first‑class cross‑project search — projects organize memory but never wall it off. Filters
(type, tags, files, recency) are what make "materials for article X" or "decisions in this project" answerable.

**What.** Merge dense (vector) and lexical (full‑text) results into one ranked list (reciprocal‑rank fusion or
equivalent). Support filters by `type`, `tags`, `related_files`, and recency. Implement the scope rule: `project`
(default = this project + global), `global`, `all`. Cross‑project hits may also appear in the default scope, ranked
lower and labeled with their project (soft isolation, not a hard wall).

**Done when.**
- A paraphrased query **and** an exact token (e.g. a function name) both return the right memory.
- Each filter (`type` / `tags` / `file` / recency) narrows results correctly.
- `scope="all"` returns matches from every project; the default scope returns this project + global.
- Tests cover each case.

**Depends on:** 1.1.

---

### 1.3 Finalize the write fields (remove `importance`)

**Why.** We decided `importance` has no concrete use in the initial scope, and a stored field with no behavior
behind it is exactly the "stub field" anti‑pattern we saw in other systems and chose to avoid. Keeping the model
honest means the field set matches what the system actually uses today. `importance` returns later, together with
the ranking/decay that gives it meaning.

**What.** Remove `importance` from the memory record and from the write tool (both the agent tool and the CLI).
Settle the field set a memory carries: `content, type, scope, project, related_files, tags, topic_key, hash,
status, links, session_id, created_at` (+ housekeeping counters). `tags` stays a plain, optional, **searchable
property** — it is not a type.

**Done when.**
- Writing a memory accepts and stores exactly that field set; `importance` is gone from the model, the agent tool,
  and the CLI.
- Existing tests are updated and green.

**Depends on:** Phase 0.

---

### 1.4 Exact‑dup + `topic_key` upsert (and explicitly no near‑dup suppression)

**Why.** The write path must stay cheap (no LLM, not even an embedding‑neighbour scan) and — more important — the
system must **not silently drop** a memory just because it looks similar to an existing one. A near‑duplicate can
differ in a small but important detail, and discarding it is "deciding for the user", which we rejected. So on write
we collapse only *truly identical* content, and we let *intentional evolution* be an explicit signal. Real
near‑duplication, if it accumulates, is cleaned up later by the background worker, which has full context.

**What.**
- **Exact duplicate:** if the hash of normalized content (whitespace‑collapsed, lowercased) matches an existing
  record, don't create a new row — bump its "seen again" counter.
- **`topic_key` upsert (explicit evolution):** if the writer passes a `topic_key` that already exists, the new
  memory **supersedes** the old one — the old record is marked `superseded` (kept, linked), the new one becomes
  current. This is the writer's deliberate "same evolving thing" signal, not deduplication.
- **No near‑dup suppression:** do **not** compare embeddings to drop similar memories on write. Two near‑but‑distinct
  memories both persist; search returns both.

**Done when.**
- Storing identical content bumps the duplicate counter and creates **no** new record.
- Reusing an existing `topic_key` marks the prior record `superseded`, links them, and makes the new one current.
- Two near‑but‑not‑identical memories both persist and both appear in search.
- Tests cover all three.

**Depends on:** 1.1, 1.3.

---

### 1.5 Reject over‑window content — moved to the embedder boundary

The over‑window guard is an **embedder concern**, not a memory‑layer step: the limit *is* the chosen embedder's
context window, so the embedder owns it (its `encode()` raises an explicit "too large" error with the limit and
the actual size; the write use case surfaces it; never truncate, never auto‑split). Concrete enforcement lands
**with the embedder choice** (still TBD). The contract is in [06-models.md](../06-models.md); the "never truncate"
policy is in [04-data-model.md](../04-data-model.md).

---

### 1.6 Deletion — `delete` / `clear` / `purge`

**Why.** Managing memory — removing wrong or obsolete entries, wiping a project, starting over — is a real, frequent
task, so it belongs in the API (minimal ≠ "only read/write"). We decided deletion is **hard** (an agent's mental
model is simply "delete"), with **no soft‑delete/inactivation** to keep it simple, and that all three operations are
available to **both the agent (MCP) and the human (CLI)**, with no confirmation guardrail. Superseding is a separate
mechanism that keeps history; deletion physically removes.

**What.** Three hard operations, as MCP tools and CLI commands:
- `delete(ids)` — remove specific memories.
- `clear(project)` — remove all memories of one project (the project is required, so "all" can't happen by accident).
- `purge()` — remove everything.

**Done when.**
- Each removes exactly its target set and nothing else.
- All three work as MCP tools and as CLI commands.
- Tests cover each.

**Depends on:** 1.1.

---

### 1.7 Session tracking

**Why.** Knowing which run produced a memory is useful on its own: provenance ("which session wrote this"),
grouping ("what this session did"), and a coherent working set for the background consolidation worker. We do
**not** build a "resume last session" experience — there is no agent‑start hook, and with 10+ parallel agents in
one project there is no single "last session". ("Where did I leave off" is meanwhile an on‑demand `search` for
`type=progress`; an aggregated `recall` is post‑MVP.)

**What.** A small **session provider** behind a port yields the run's id: lazily generated once and returned for
every write of the run (a read‑only run generates nothing). `remember` stamps each stored memory with it; the agent
never sets it. **No session entity/table** — which projects a run touched and when are derivable from the memories
(each carries `project` + `session_id` + `created_at`), so a separate session record would be a redundant
denormalization with no current consumer (deferred until one needs it, e.g. consolidation). One process = one run
today; a shared‑process deployment swaps the provider for a per‑connection one behind the same port. Done now
because provenance is **write‑time‑only** — it can't be reconstructed for memories written before stamping.

**Done when.**
- Every memory written in a run carries that run's `session_id` (one id per run).
- Distinct runs get distinct session ids.
- The agent never sets `session_id`; the provider supplies it.
- Tests.

**Depends on:** 1.1.

---

### 1.8 `recall(project)` — deferred to post‑MVP

A single aggregated context call was the original "magic word", but a *useful* recall (concise, not a context
dump) needs **LLM synthesis**, which we keep off the read path. So `recall` is **post‑MVP** (see
[post-mvp.md](post-mvp.md)). In the MVP the agent retrieves on demand with `search` — `type=rule` for rules,
`type=progress` for where it left off.

---

### 1.9 Deterministic links + provenance

**Why.** Relations between memories (this decision *superseded* that one; this learning *derived from* that debug
session) make context navigable and give evolution real structure — but the **coding agent must not be the one
building the graph**. Its job is to code, and we saw it can't even reliably pick a type; making it author typed edges
would be unreliable and burdensome. So links are created **for** the agent, deterministically, as a by‑product of
normal actions. And, taking the one genuinely useful idea from code‑graph tools, every link carries **provenance**
(how it was created) so a caller can tell a certain, deterministic link from a later, model‑inferred guess.

**What.**
- **`supersedes`** — written automatically when a `topic_key` upsert happens (from 1.4).
- **`derived_from`** — written when the writer passes an optional `source_ids` (memories the new one builds on);
  never required.
- Every link stores its `provenance` (e.g. `topic_key`, `source_ids`). Reserve a generic, typed `links` shape now
  (so associative/contradiction links from the background worker fit later), and keep the record forward‑compatible
  for the post‑MVP bi‑temporal fields (nullable, non‑breaking).
- **No semantic/inferred links here** — that's the background worker's job later.

**Done when.**
- A `topic_key` upsert writes a `supersedes` edge; `source_ids` writes a `derived_from` edge; both carry `provenance`.
- Links for a memory are retrievable.
- A schema check confirms adding the bi‑temporal timestamps later is non‑breaking.
- Tests.

**Depends on:** 1.4.

---

### 1.10 Concurrency — one shared process, 10+ agents

**Why.** The defining constraint of the project: a single shared process must serve **10+ agents** without losing
writes or throwing "database is locked" — the failure mode of the multi‑process‑on‑one‑file tools we surveyed.
Because writes are cheap (embed + insert, no LLM), serializing them inside one process is enough; we don't need a
heavyweight server.

**What.** Make the single service safe under concurrency: serialize writes internally (a queue or lock) so
concurrent writers never corrupt or lose data, while reads run in parallel.

**Done when.**
- A stress test with **≥10 parallel writers** shows zero lost writes and no lock errors.
- Reads run concurrently with writes.

**Depends on:** 1.1–1.6.

---

**Phase done when:** the MVP FRs hold (FR‑11 `recall` is post‑MVP); the 10‑agent concurrency test passes; LanceDB is the backend.
