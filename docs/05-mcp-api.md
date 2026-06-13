# 05 — MCP API (tool surface)

**Design rule: a small, obvious agent‑facing API driven by real tasks** — behavior via parameters
(not a tool per memory type, not a tool per operation). Minimal ≠ exactly two: we add a tool when a
real task needs it. An agent should learn the surface in seconds.

## Agent‑facing MCP tools

### `recall(project) -> Context`  *(Phase 1)*
The "magic word" at session start. Loads aggregated context:
```jsonc
{
  "project": "checkout-api",
  "rules":        [/* type=rule, scope project + global */],
  "recent":       [/* recent activity for the project */],
  "pending_tasks":[/* open tasks, if the tasks feature is enabled */],
  "session_recap":"where we left off last time"
}
```

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
- Behavior: normalize → **exact‑dup** check (hash) → **`topic_key` upsert** if matched (supersede) → embed → insert.
  Near‑similar memories are **not** suppressed here (see [04-data-model.md](04-data-model.md)).
- Returns `{id, dedup}`: `dedup` is `null` (new), or `"exact"` (identical existing record).
- **Rules** are just `remember(type="rule")`; `recall` returns existing rules (no separate rule tools). The agent
  stores a rule only on an explicit user request.

### `search(query, scope?, project?, type?, tags?, limit?, full?) -> list[MemoryHit]`
The single retrieval tool. `scope` decides project vs. global vs. **cross‑project**.
```python
search("how do we handle auth errors")                 # current project + global (default)
search("connection pool limits", scope="all")          # cross‑project
search("redis", scope="all", type="decision", limit=5) # cross‑project, typed
search("article-x", tags=["article-x"])                # filter by tag
```
- `scope`: `project` (default = current project + global) | `global` | `all` (every project). (`session` is post‑MVP.)
- Cross‑project hits may also surface in the default scope, ranked lower and labeled with their `project` (soft isolation).
- `full=false` → previews (`MemoryHit`); `full=true` → complete `content` (replaces a separate `expand` tool).
- `MemoryHit = {id, score, type, project, scope, content_preview, related_files, created_at}`.

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
mnemo consolidate [--project P]      # run background consolidation now (Phase 3)
mnemo doctor                         # health: store/embedder/disk/warnings
mnemo export / import                # portability (Phase 4)
```
(`delete` / `clear` / `purge` are also available as CLI commands.)

## Design notes
- One write verb, one read verb, one context verb, plus deletion — type/scope/filters are parameters.
- `search` defaults to **hybrid** (dense + lexical) so exact matches (function names, error codes) aren't missed.
- `remember` is fast and idempotent by `hash` / `topic_key`.
- Cross‑project search is a `scope="all"` flag, not a separate tool.
- Background consolidation is automatic; `mnemo consolidate` is only for manual runs/debugging.
