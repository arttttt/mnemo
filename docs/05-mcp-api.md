# 05 — MCP API (tool surface)

**Design rule: a small, obvious agent‑facing API driven by real tasks** — behavior via parameters
(not a tool per memory type, not a tool per operation). Minimal ≠ exactly two: we add a tool when a
real task needs it. An agent should learn the surface in seconds.

## Agent‑facing MCP tools

> **Terminology — two senses of "recall".** (1) **Retrieval** = finding relevant memories by meaning; this is what
> `search` does and what the **embedder** powers (measured as recall@k). (2) **`recall(project)` digest** = an
> aggregated session/project *summary*; this is text synthesis, the **generator's** job. They are different tools
> and different models. Below, `recall` always means sense (2).

### `recall(project)` — **post‑MVP** (the digest tool, not retrieval)
A single aggregated context bundle (a session/project **summary**) was the original "magic word", but a *useful* one
(concise, not a context dump) needs **LLM synthesis** (the generator) — which we keep off the read path — so the
`recall` digest is **deferred to post‑MVP** (see [roadmap/post-mvp.md](roadmap/post-mvp.md)). It would *select*
memories (by `session_id`/date, or by meaning via the embedder) and *synthesize* a summary (the generator). In the
MVP the agent instead **retrieves on demand** with `search`: `search("...", type="rule")` for rules,
`search("...", type="progress")` for where it left off.

### `remember(content, type?, project?, scope?, related_files?, tags?, topic_key?) -> {id, dedup}`
The single write tool. No LLM on this path. (`importance` is **post‑MVP** — not a parameter yet.)
```python
remember("Using JWT with refresh rotation; httpOnly cookies; ...",
         type="decision", project="checkout-api", related_files=["src/auth/jwt.ts"])

remember("Always confirm destructive DB ops", type="rule", scope="global")   # a rule
remember("quick note about the retry loop", project="checkout-api")          # type defaults to working-notes
remember("Auth model v2: ...", type="decision", project="checkout-api",
         topic_key="auth/model")                                             # evolves the same record
```
- `type` default `working-notes`; `scope` default `project` (or `global` if no project).
- Behavior: normalize → **exact‑dup** check (hash) → **`topic_key` upsert** if matched (supersede) → **insert +
  lexical index → enqueue embed** (the vector is computed off the hot path — see
  [03-architecture.md](03-architecture.md#deferred-embedding-async-vector-computation)). Near‑similar memories are
  **not** suppressed here (see [04-data-model.md](04-data-model.md)).
- Returns `{id, dedup}`: `dedup` is `null` (new), or `"exact"` (identical existing record).
- **Rules** are just `remember(type="rule")` and surface via `search` (`type=rule`) — no separate rule tools. The
  agent stores a rule only on an explicit user request.

### `search(query, scope?, project?, type?, tags?, related_files?, recency_days?, limit?) -> list[MemoryHit]`
The single retrieval tool. `scope` decides project vs. global vs. **cross‑project**; the optional filters narrow it.
```python
search("how do we handle auth errors")                  # current project + global (default)
search("connection pool limits", scope="all")           # cross‑project
search("redis", scope="all", type="decision", limit=5)  # cross‑project, typed
search("article-x", tags=["article-x"])                 # ALL given tags must be present
search("jwt", related_files=["src/auth/jwt.ts"])        # references ANY of these files
search("recent changes", recency_days=7)                # only memories from the last 7 days
```
- `scope`: `project` (default = current project + global) | `global` | `all` (every project). (`session` is post‑MVP.)
- Filters (all optional): `type` (one type), `tags` (memory must carry **all** of them), `related_files`
  (references **any** of them), `recency_days` (created within the last N days).
- Project scope is a **hard filter** (current project + `global`); other projects are excluded from the default scope. Cross‑project search is the explicit `scope="all"` (see [04-data-model.md](04-data-model.md)).
- `MemoryHit = {id, score, type, scope, project, content, related_files, created_at}`.

### Deletion — `delete` / `clear` / `purge`
Hard delete only (no soft‑delete). Available to **both the agent and the CLI**.
```python
delete(ids=["..."])     # remove specific memories
clear(project="x")      # remove all memories of one project
purge()                 # remove everything
```
Superseding (evolution) is separate and keeps history; deletion physically removes records.

## Operational — CLI (not agent‑facing MCP tools)
Kept off the agent surface:
```
mnemo stats [--project P]            # counts by type, projects, size
mnemo reindex [--dry-run]            # re-embed all memories; restarts the shared service
mnemo consolidate [--project P]      # run background consolidation now (Phase 3)
mnemo doctor                         # health: store/embedder/disk/warnings
mnemo export / import                # portability (Phase 4)
```
(`delete` / `clear` / `purge` are also available as CLI commands.)

## Design notes
- One write verb, one read verb, plus deletion — type/scope/filters are parameters. (`recall` is post‑MVP.)
- `search` defaults to **hybrid** (dense + lexical) so exact matches (function names, error codes) aren't missed.
- `remember` is fast and idempotent by `hash` / `topic_key`.
- Cross‑project search is a `scope="all"` flag, not a separate tool.
- Background consolidation is automatic; `mnemo consolidate` is only for manual runs/debugging.
