# 05 — MCP API (tool surface)

**Design rule: a small, obvious agent‑facing API driven by real tasks** — behavior via parameters
(not a tool per memory type, not a tool per operation). Minimal ≠ exactly two: we add a tool when a
real task needs it. An agent should learn the surface in seconds.

## Agent‑facing MCP tools

> **Terminology — two senses of "recall".** (1) **Retrieval** = finding relevant memories by meaning; this is what
> `search` does and what the **embedder** powers (measured as recall@k). (2) **`recall(query, project)` answer** =
> a synthesized answer to a question; it *retrieves* the relevant memories (the embedder) and then *synthesizes* an
> answer over them (the **generator**). So recall builds on retrieval and adds generation: `search` returns the
> hits, `recall` returns a written answer. Below, `recall` always means sense (2).

### `recall(query, project, force?) -> {project, summary, sources}` — opt‑in LLM synthesis
The one read tool that runs an LLM. It retrieves the memories most relevant to `query` (the same hybrid
retrieval as `search`), then a local generator writes a concise answer using **only** those memories — replying
exactly "No relevant memories found." when none are relevant (never outside knowledge). Returns the synthesized
`summary` plus lightweight `sources` — `{id, type}` for each memory the answer drew on, **not** their content,
so the answer stays light on the caller's context. Unlike `search` (ranked hits) it returns a written answer. The
write path stays LLM‑free; `recall` is the single opt‑in LLM read tool, and the generator is **transient** (loaded
on demand, unloaded after). Disable synthesis entirely with `MNEMO_GENERATOR=off`.
- **Experimental — `force` gate.** Because the answer is LLM‑generated it can be wrong, so `recall` is gated: `force` defaults to `false` and the call is **rejected** with a warning until you pass `force=true` to acknowledge it. Prefer `search`/`browse` for verbatim memories. The gate lives in the MCP tool and fires before any model loads.

### `remember(content, type?, project?, scope?, related_files?, tags?, topic_key?) -> {id, status}`
The single write tool. No LLM on this path. (`importance` is **post‑MVP** — not a parameter yet.)
```python
remember("Using JWT with refresh rotation; httpOnly cookies; ...",
         type="decision", project="checkout-api", related_files=["src/auth/jwt.ts"])

remember("Always confirm destructive DB ops", type="rule", scope="global")   # a rule
remember("quick note about the retry loop", project="checkout-api")          # type defaults to working-notes
remember("Auth model v2: ...", type="decision", project="checkout-api",
         topic_key="auth/model")                                             # evolves the same record
```
- `type` default `working-notes`; `scope` default `project`. A project‑scoped write **requires** a `project` (or use `scope="global"`), and passing a `project` with `scope="global"` is rejected — the same scope↔project contract the read tools enforce.
- Behavior: reject blank content → normalize → **exact‑dup** check (hash, only against **active** records in the **same project/scope**) → **`topic_key` upsert** if matched (supersede) → **insert +
  lexical index → enqueue embed** (the vector is computed off the hot path — see
  [03-architecture.md](03-architecture.md#deferred-embedding-async-vector-computation)). Near‑similar memories are
  **not** suppressed here (see [04-data-model.md](04-data-model.md)).
- Returns `{id, status}`: `status` is `"created"` (new record), `"duplicate"` (identical content already **active in the same project/scope** — nothing written, the existing id is returned), or `"superseded"` (a `topic_key` upsert replaced a prior record; the prior is marked superseded, recorded in the `supersedes` column).
- **Rules** are just `remember(type="rule")` and surface via `search` (`type=rule`) — no separate rule tools. The
  agent stores a rule only on an explicit user request.

### `search(query, scope?, project?, type?, tags?, related_files?, created_after?, limit?) -> list[MemoryHit]`
The single retrieval tool. `scope` decides project vs. global vs. **cross‑project**; the optional filters narrow it.
```python
search("how do we handle auth errors", project="checkout-api")  # project + global (default scope)
search("connection pool limits", scope="all")                   # cross‑project
search("redis", scope="all", type="decision", limit=5)          # cross‑project, typed
search("article-x", scope="all", tags=["article-x"])            # ALL given tags must be present
search("jwt", scope="all", related_files=["src/auth/jwt.ts"])   # references ANY of these files
search("changes", scope="all", created_after="2026-06-01")      # created at/after an ISO‑8601 instant
```
- `scope`: `project` (default = the named project + global; **requires** `project`) | `global` | `all` (every project). Passing `project` with `global`/`all` is rejected. (`session` is post‑MVP.)
- Filters (all optional): `type` (one type), `tags` (memory must carry **all** of them), `related_files`
  (references **any** of them), `created_after` (created at/after an ISO‑8601 date or datetime).
- Project scope is a **hard filter** (the named project + `global`); other projects are excluded from the default scope. Cross‑project search is the explicit `scope="all"` (see [04-data-model.md](04-data-model.md)).
- Hits come in **rank order** (the list order is the ranking); there is **no relevance score** — the underlying
  RRF value is opaque and easily misread as a confidence, so it stays internal. Read the hit content to judge it.
- Hits carry `topic_key` (dereference a hit with `get`, or evolve it with `remember(topic_key=…)`) but **no
  `status`** — `search` returns active heads only, so a status field would be a constant.
- `MemoryHit = {id, type, scope, project, content, related_files, created_at, topic_key}`.

### `browse(scope?, project?, type?, tags?, related_files?, created_after?, status?, limit?) -> list[BrowseHit]`
The query‑less companion to `search`: retrieve a **category**, newest first, with no relevance ranking. Use it
when a semantic query would only bias the order ("all `type=decision` in this project", "everything tagged
`feedback`"). Same filters and scoping rules as `search` (a `project` is required for `scope="project"`).
```python
browse(project="checkout-api")                       # newest ACTIVE memories in the project + global
browse(project="checkout-api", type="decision")      # all active decisions, newest first
browse(scope="all", tags=["feedback"])               # a category across every project
browse(project="checkout-api", status="superseded")  # audit: only replaced versions
```
- No `query`, no ranking: hits are ordered by recency, so there is **no `score`**.
- `status`: `active` (default — current versions only) | `superseded` (only replaced) | `all`. The lever for
  **auditing** replaced versions; `search` has no such filter (it is always active).
- `BrowseHit = {id, type, scope, project, content, related_files, created_at, topic_key, status}` — `MemoryHit` plus
  `topic_key`/`status` for audit; neither carries a relevance score.

### `get(id? | topic_key?, project?, scope?, chain_limit?, chain_after?) -> GetMemory`
Dereference **one** memory exactly — by global `id` or by `topic_key` — and get its **supersede chain** (version
history). Deterministic (an indexed point lookup, no LLM, no ranking): the counterpart to semantic `search` and
filtered `browse`. A `[[wikilink]]` **is** a `topic_key`, so this is how you follow one.
```python
get(id="9af3…")                                          # the exact record (any status), by global id
get(topic_key="auth/jwt-model", project="checkout-api")  # the chain's ACTIVE head + its history
get(topic_key="rule/x", scope="global")                  # a global topic_key
get(topic_key="auth/jwt-model", project="checkout-api", chain_limit=5, chain_after="…")  # page older versions
```
- **`id` and `topic_key` are mutually exclusive** — pass exactly one (both, or neither, is a loud error). `id` is the
  stronger key: a globally‑unique, immutable handle resolving the exact record of **any** status (so it reaches a
  **superseded** version `search`/`browse` won't surface); it ignores `scope`/`project`. `topic_key` resolves the
  chain's **active head** within a `project` (or `scope="global"`) — the same scope↔project contract as the read tools.
- A **miss is a loud error** (with near‑match suggestions for a `topic_key`), not a silent empty — you are
  dereferencing a handle you believe exists (e.g. a stale `[[wikilink]]`).
- **Chain.** `chain` is the version lineage walked along the `supersedes` pointers, **newest → oldest**, as **light**
  entries `{id, status, created_at}` (no content — fetch a specific old version with another `get(id=…)`). Capped at
  `chain_limit` (default 10) and paged with `chain_after` (the id of the last entry you saw → the next‑older window);
  `chain_total` is the full depth, so you know whether to page.
- `GetMemory = {id, type, scope, project, content, related_files, created_at, topic_key, status, supersedes, chain, chain_total}`.

### Projects — `create_project` / `update_project` / `list_projects` / `delete_project`
A project is a **registered entity**, not just a slug on a memory: you must create it before writing to it, so a
typo can't silently spawn an invisible phantom project. `remember`/`search`/`browse` on an unknown `project` raise
an error carrying **near‑match suggestions**, so you can fix the slug or create it.
```python
create_project("checkout-api", description="payments + checkout")  # the only way to add a project
update_project("checkout-api", "new description")                  # set/change the description
list_projects()                                                    # the registered projects (global is not one)
delete_project("checkout-api")                                     # delete it AND all its memories (cascade)
```
- `create_project(name, description?) -> {slug, description, created_at}`. Re‑creating an existing slug **errors** (use `update_project` to change it).
- `update_project(name, description) -> {...}`. The only way to set/change a description; errors (with near‑match) on an unknown slug.
- `list_projects() -> [{slug, description, created_at}]`, newest first; the reserved global scope is excluded.
- `delete_project(name) -> {slug, description, created_at}` (the project removed). Deletes the project and, via the store's `ON DELETE CASCADE`, **all its memories — atomically**. Errors (with near‑match) on an unknown slug.

### Deletion — `delete` / `delete_project`
Hard delete only (no soft‑delete). Available to **both the agent and the CLI**. Wiping everything (`purge`) is **CLI‑only** (too destructive for the agent surface, and gated by a confirmation prompt) — see the operational section.
```python
delete(ids=["..."])                # remove specific memories
delete(ids=["..."], cascade=True)  # also remove every OLDER version they supersede (the whole lineage)
delete_project("x")         # remove a project and ALL its memories (one atomic cascade)
```
A whole project is the unit of bulk deletion (`delete_project`); there is no per‑project `clear`. Superseding (evolution) is separate and keeps history; deletion physically removes records. `delete(ids, cascade=True)` expands each id to itself plus every older member it transitively supersedes (down to the chain root) and removes them in one transaction — the way to drop a whole supersede lineage in one call — superseded versions are hidden from default `search`/`browse` (surface them with `browse(status=…)` or `get`), so cascade is the convenient way to remove the lineage by its head.

## Operational — CLI (not agent‑facing MCP tools)
Kept off the agent surface:
```
mnemo stats [--project P]            # total + pending (awaiting a vector) + counts by type
mnemo reindex [--dry-run]            # re-embed all memories; restarts the shared service
mnemo consolidate [--project P]      # run background consolidation now (Phase 3)
mnemo doctor                         # health: store/embedder/disk/warnings
mnemo export / import                # portability (Phase 4)
mnemo purge [--yes]                  # delete EVERYTHING: memories + project registry (prompts to confirm; CLI-only)
```
(`delete` / `delete-project`, and the project tools `create-project` / `update-project` / `list-projects`, are also available as CLI commands.)

## Design notes
- One write verb, **four** read verbs (`search` by meaning, `browse` by filter, `get` by exact id/topic_key, `recall` for an LLM‑synthesized answer), plus deletion — type/scope/filters are parameters. The read split is by **mode** (semantic / filter‑survey / exact‑dereference / synthesis), not one tool per entity.
- Projects are **registered first‑class entities** (`create_project`/`update_project`/`list_projects`/`delete_project`); writing to or reading an unknown project is a hard error with near‑match suggestions, so a typo can't create a phantom project. `delete_project` deletes the project and its memories in one DB cascade.
- `search` defaults to **hybrid** (dense + lexical) so exact matches (function names, error codes) aren't missed.
- `remember` is fast and idempotent by `hash` / `topic_key`.
- Cross‑project search is a `scope="all"` flag, not a separate tool.
- Background consolidation is automatic; `mnemo consolidate` is only for manual runs/debugging.
