# 04 — Data Model

## Memory types

The type defines *what* artifact is stored and *how* it should later be searched/used.
Type is a **parameter** of a single write tool — not a separate tool per type (see [05-mcp-api.md](05-mcp-api.md)).

| Type | What it captures |
|---|---|
| `decision` | Architecture/approach choice, alternatives considered, rationale |
| `debug` | Root cause, applied fix, affected files |
| `progress` | Where we left off, what's done, what's next |
| `feature` | Design notes, implementation approach |
| `research` | Findings, evaluations, references |
| `code-snippet` | Working patterns, reusable solutions |
| `rule` | Behavioral rules for the agent (per‑project or global) |
| `learning` | A lesson learned, gotchas, the non‑obvious |
| `discussion` | Agreements, notes from discussions |
| `design` | Architecture diagrams, system design |
| `working-notes` | Scratch pad, temporary context (default if `type` omitted) |

## Scopes (local / global / session) — project isolation + global, cross‑project on request

A memory has a **scope** that controls where it lives and how it surfaces in search.
Reads **isolate by project** (plus always‑visible global), and **cross‑project search is a first‑class
opt‑in** via `scope="all"` — projects are an organizing dimension you can cross deliberately, not an automatic blend.

| Scope | Meaning | Default in reads |
|---|---|---|
| `project` (local) | Belongs to one project (`payload.project`). The common case. | yes — the current project |
| `global` (`__global__`) | Cross‑project knowledge and rules; applies everywhere. | yes — always included |
| `all` (search‑only) | Not a storage scope — a **search mode** spanning every project. | on request (`scope="all"`) |
| ~~`session`~~ (deferred) | Ephemeral working context keyed by `session_id`. **Not in v1.** | — |

**Scope rules (as implemented):**
- Default `search` (`scope="project"`) covers **the named project + global** only — other projects are a hard
  WHERE filter `(m.project = ? OR m.scope = 'global')`, excluded from the candidate set, so cross‑cutting
  rules/lessons (global) always surface but unrelated projects never leak in. The project must be named
  explicitly — there is no inferred "current project" — so `scope="project"` without a `project` is **rejected**.
  A search for a project with no memories returns an empty result (no error — "nothing remembered yet").
- `scope="all"` drops the project filter and searches **across all projects** (the cross‑project capability).
- `scope="global"` returns only global memories. Passing `project` together with `scope in {all, global}` is
  **rejected** (those scopes ignore `project` — `scope` is authoritative — so accepting it would silently drop
  the filter and return a wrong‑scoped result; the API refuses the contradiction instead).
- Implementation: **one collection + a `project` column used as a hard filter** (not a soft boost, not a
  per‑project partition) — so cross‑project search is just dropping that filter, never expensive.

> **`session` scope is deferred (not in v1).** For transient context, use `working-notes` for now. Adding a
> `session` scope later is cheap (one scope value + a filter), so we leave it out until there's a clear need.
> Note: `session_id` is still recorded on every memory (session tracking — provenance/grouping) — that is independent of a session *scope*.

## Record

Logical schema of one memory. Stored as a point in the vector store: `vector` + `payload`.

```jsonc
{
  "id": "5f3e9c…",                     // 160-bit random hex (40 chars) — longer than a uuid4
  "vector": [/* float32[dim] — embedding of content */],
  "payload": {
    "content": "Markdown text of the memory (problem / solution / reasoning)",
    "type": "decision",                 // one of the types above; default "working-notes"
    "scope": "project",                // project | global   (session scope deferred, not v1)
    "project": "checkout-api",          // kebab-case; ignored when scope=global
    "session_id": "5f3e9c…",             // which run created this (session tracking)
    "related_files": ["src/auth/jwt.ts"],
    "tags": ["authentication", "jwt"],   // optional keyword list; a property (not a type), searchable
    "hash": "sha256(normalized_content)",// for exact dedup
    "status": "active",                // active | superseded
    "supersedes": null,                // id of the previous version (record evolution)
    "topic_key": "auth/jwt-model",      // stable key for upserting an "evolving" record
    "created_at": "2026-06-13T18:42:00Z",
    "updated_at": "2026-06-13T18:42:00Z"
  }
}
```

### Recommended `content` shape
```markdown
## <short title>
**Problem:** what was wrong / why
**Solution:** what changed and why this way
**Files:** key files + what changed in each
**Testing:** how it was verified
```

## Memory size & atomicity

One memory becomes **one embedding vector**, so its size is bounded by the chosen embedder's context
window (see [06-models.md](06-models.md); the requirement is "comfortably thousands of words").

**Guidance (soft).** Keep a memory to **one logical unit** — one decision, one lesson, one debugging
outcome — and keep it focused. Don't write a sprawling wall of text, and don't over‑fragment either: a
coherent thought stays a single memory even if it spans a few paragraphs. Two failure modes to avoid:
- **Too big** → a blurry "average" vector that recalls poorly and dilutes precision.
- **Too fine** → fragmented memories that lose cohesion and add extra linking overhead.

**Hard rule — never silently truncate.** If content exceeds the embedder's window, the write is
**rejected with an explicit, actionable error** (stating the limit and the actual size) — enforced at the
embedder boundary (see [06-models.md](06-models.md)), so the calling
agent — already an LLM — compresses or splits it and re‑submits. The write path stays LLM‑free: it does
**not** auto‑summarize or auto‑split. Splitting a large document into linked atoms is a separate
background/ingestion capability — **post‑MVP** (see [10-roadmap.md](10-roadmap.md)).

## Dedup & evolution on write (no LLM)

1. **Exact duplicate** → the `hash` of normalized content matches an existing record: don't spawn a new record,
   return its id with `status: "duplicate"`. Identical text carries no new information.
2. **`topic_key` upsert (explicit evolution)** → if `topic_key` matches an existing record, the new one
   supersedes it (old → `status: superseded`, kept). This is the writer's explicit signal — **not** dedup.
3. **Near‑similar is NOT acted on here.** We do **not** suppress near‑duplicates on write — a small but
   important difference could be lost, and the system must not decide that for the user. Such memories coexist;
   search returns them. Genuine duplication is merged/flagged later by the background worker, with context
   ([08-consolidation.md](08-consolidation.md)).

Only a `hash` lookup and (for upsert) a `topic_key` lookup are on the write path — both cheap; no
embedding‑neighbour scan, no LLM.

## Evolution & staleness

**Currency changes only on an explicit signal from the writer — the system never decides on its own.**

- A changed fact/decision → a new record; the writer signals it by **reusing the same `topic_key`** (or an
  explicit `supersede`). In response the system *mechanically* marks the old record `status: superseded` and
  links them (`supersedes` = old id). This is bookkeeping, not a judgement. Recoverable; nothing is deleted.
- **No automatic staleness.** The background consolidation worker may only *flag* likely contradictions for
  review — it never marks a memory stale by itself. The human or coding‑agent decides, via `delete` (smarter
  review tooling for flagged contradictions is **post‑MVP**). Exact revision UX is still **TBD**.
- Age is surfaced (`created_at`) so callers judge freshness themselves. Re‑verifying a fact against current
  code is the **coding‑agent's** job, not the memory's — mnemo does **not** watch files or index code.

**Temporal model:** the MVP stays simple — `created_at` + supersede chain + `status`. A full **bi‑temporal**
validity model (transaction‑time + valid‑time, point‑in‑time queries, retro‑corrections) is a committed
**post‑MVP** item, to be done *in full* (not half‑measures). The schema stays forward‑compatible: the four
timestamps are added later as nullable fields, no breaking migration. See [10-roadmap.md](10-roadmap.md).

## Deletion

Hard delete only — no soft‑delete/inactivation. All three are available to **both the agent and the CLI**:
- `delete(ids)` — remove specific memories.
- `clear(project)` — remove all memories of one project.
- `purge()` — remove everything.

Superseding is separate: it keeps history via `status: superseded`; deletion physically removes records.

## Additional entities (v1+)

- **sessions:** `id, project, started_at, ended_at` — tracking only (provenance/grouping). A stored summary and `recall` are **post‑MVP**.
- **rules:** plain memories with `type: rule` (scope `project` or `global`).
- **tasks (MAY):** `id, project, description, status, linked_memory_ids[]`.

## Storage

- Embedded: **SQLite** (`sqlite-vec` for dense vectors + FTS5 for lexical, fused by RRF; relational core for metadata/edges/transactions). One backend only — no mixing stores.
- **One set of tables for all projects** (cross‑project search is cheap); `project`/`scope` are columns.
- Data directory: `~/.mnemo/data/` — one SQLite file = the whole state (backup/move = copy the file).
- Engine choice and rationale — in [09-tech-stack.md](09-tech-stack.md) and [adr/0001-storage-engine.md](adr/0001-storage-engine.md).
