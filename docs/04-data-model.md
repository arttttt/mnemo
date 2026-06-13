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

## Scopes (local / global / session) — soft, not hard isolation

A memory has a **scope** that controls where it lives and how it surfaces in search.
**Cross‑project search is a first‑class capability** — projects are an organizing dimension, **not a wall**.

| Scope | Meaning | Default in reads |
|---|---|---|
| `project` (local) | Belongs to one project (`payload.project`). The common case. | yes — the current project |
| `global` (`__global__`) | Cross‑project knowledge and rules; applies everywhere. | yes — always included |
| `all` (search‑only) | Not a storage scope — a **search mode** spanning every project. | on request (`scope="all"`) |
| ~~`session`~~ (deferred) | Ephemeral working context keyed by `session_id`. **Not in v1.** | — |

**Soft isolation rules:**
- Default `search` covers **current project + global**, so cross‑cutting rules/lessons always surface.
- `scope="all"` searches **across all projects** (the must‑have cross‑project capability).
- Even in the default scope, strongly‑relevant hits from other projects MAY be included, ranked lower and
  labeled with their `project` (a soft boost, not a hard partition). This is "no hard isolation by search".
- Implementation consequence: **one collection + a `project` payload field** (a soft filter/boost), **not**
  a collection‑per‑project — because a per‑project partition would make cross‑project search expensive.

> **`session` scope is deferred (not in v1).** For transient context, use `working-notes` for now. Adding a
> `session` scope later is cheap (one scope value + a filter), so we leave it out until there's a clear need.
> Note: `session_id` is still recorded on every memory (for `session_recap`) — that is independent of a session *scope*.

## Record

Logical schema of one memory. Stored as a point in the vector store: `vector` + `payload`.

```jsonc
{
  "id": "uuid",
  "vector": [/* float32[dim] — embedding of content */],
  "payload": {
    "content": "Markdown text of the memory (problem / solution / reasoning)",
    "type": "decision",                 // one of the types above; default "working-notes"
    "scope": "project",                // project | global   (session scope deferred, not v1)
    "project": "checkout-api",          // kebab-case; ignored when scope=global
    "session_id": "2026-06-13T...",     // which session created this (used by session_recap)
    "related_files": ["src/auth/jwt.ts"],
    "tags": ["authentication", "jwt"],
    "importance": 0.8,                  // 0.0..1.0
    "hash": "sha256(normalized_content)",// for exact dedup
    "status": "active",                // active | superseded | inactive
    "supersedes": null,                // id of the previous version (record evolution)
    "topic_key": "auth/jwt-model",      // stable key for upserting an "evolving" record
    "created_at": "2026-06-13T18:42:00Z",
    "updated_at": "2026-06-13T18:42:00Z",
    "last_seen_at": "2026-06-13T18:42:00Z"
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

## Dedup (on the hot path, no LLM)

1. **Exact:** the `hash` of normalized content matches → update `last_seen_at`, `duplicate_count`, do not spawn a record.
2. **Near‑dup:** cosine to nearest neighbors > threshold (e.g. 0.95) within the same `scope`(+`project` for local) →
   mark as a merge candidate (the merge itself happens in the background) or upsert by `topic_key`.
3. **topic_key:** if set — upsert the "live" record (versioned) instead of a new row.

Cosine and hash are cheap, so dedup stays on the hot path. "Smart" semantic merging (paraphrases)
is done by background consolidation ([08-consolidation.md](08-consolidation.md)).

## Evolution / staleness

- A changed decision → a new record with `supersedes` = id of the old one; the old one → `status: superseded` (recoverable).
- Background consolidation can mark stale memories `inactive` (with a reason) without physically deleting them.

## Additional entities (v1+)

- **sessions:** `id, project, started_at, ended_at, summary` — for `session_recap`.
- **rules:** plain memories with `type: rule` (scope `project` or `global`).
- **tasks (MAY):** `id, project, description, status, linked_memory_ids[]`.

## Storage

- Embedded: **LanceDB** (file tables, dense + full‑text) or **sqlite‑vec** (a single `.sqlite`).
- **One collection/table for all projects** (cross‑project search is cheap); `project`/`scope` are payload fields.
- Data directory: `~/.mnemo/data/`. One directory = the whole state (backup/move = copy the folder).
- Engine choice and rationale — in [09-tech-stack.md](09-tech-stack.md).
