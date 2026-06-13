# 05 — MCP API (tool surface)

**Design rule: a small, obvious agent‑facing API.** Few tools, behavior controlled by parameters
(not a tool per memory type, not a tool per operation). An agent should learn it in seconds.

## Agent‑facing tools — 3 core verbs (+1 optional)

The whole surface an agent needs:

### 1. `recall(project) -> Context`
The "magic word" at session start. Loads aggregated context:
```jsonc
{
  "project": "checkout-api",
  "rules":        [/* type=rule, scope project + global */],
  "recent":       [/* recent activity for the project */],
  "pending_tasks":[/* open tasks (if any) */],
  "session_recap":"where we left off last time"
}
```

### 2. `remember(content, type?, project?, scope?, related_files?, tags?, importance?, topic_key?) -> {id}`
The single write tool for everything. `type` and `scope` are parameters.
```python
remember("Using JWT with refresh rotation; httpOnly cookies; ...",
         type="decision", project="checkout-api",
         related_files=["src/auth/jwt.ts"], importance=0.7)

remember("Always confirm destructive DB ops", type="rule", scope="global")   # a rule
remember("quick note about the retry loop", project="checkout-api")          # type defaults to working-notes
```
- `type` default: `working-notes`. `scope` default: `project` (or `global` if no project given).
- Behavior: normalize → dedup (hash/cosine/`topic_key`) → embed → upsert. Returns the `id`
  (or the id of an existing record on dedup). **No LLM on this path.**
- **Rules are just `remember(type="rule")`.** Convention: the agent stores a rule only on an explicit
  user request, and `recall` already returns existing rules (no separate rule tools).

### 3. `search(query, scope?, project?, type?, filters?, limit?, full?) -> list[MemoryHit]`
The single retrieval tool. `scope` decides project vs. global vs. **cross‑project**.
```python
search("how do we handle auth errors")                      # current project + global (default)
search("connection pool limits", scope="all")               # CROSS‑PROJECT search
search("handleAuthCallback", mode="keyword")                # exact term
search("redis", scope="all", type="decision", limit=5)      # cross‑project, typed
```
- `scope`: `project` (default = current project + global) | `global` | `all` (every project). (`session` deferred, not v1.)
- Cross‑project hits in the default scope may still appear, ranked lower and labeled with their `project`
  (soft isolation — see [04-data-model.md](04-data-model.md)).
- `full=false` returns previews (`MemoryHit`); `full=true` returns complete `content` (replaces a separate `expand` tool).
- `MemoryHit = {id, score, type, project, scope, content_preview, related_files, created_at}`.

### 4. (optional) `forget(ids, reason) -> {count}`
Soft delete / inactivation (recoverable). Optional in v1; can live in the CLI instead.

That's it for the agent: **recall, remember, search** (+ optional **forget**). No `store_decision`/
`store_debug`/`expand`/`update`/`supersede`/`get_rules` as separate tools — folded into the three above.

## NOT agent‑facing — operational (CLI only)

These keep the agent surface tiny. Exposed via the CLI / internal scheduler, **not** as MCP tools:
```
mnemo stats [--project P]            # counts by type, projects, size
mnemo consolidate [--project P]      # run background consolidation now
mnemo doctor                         # health: store/embedder/disk/warnings
mnemo export / import                # portability (MAY)
```

## API design notes
- One write verb, one read verb, one context verb — type/scope/filters are parameters.
- `search` defaults to `hybrid` (dense + lexical) so exact matches (function names, error codes) aren't missed.
- `remember` must be fast and idempotent by `hash`/`topic_key`.
- Cross‑project search is a `scope="all"` flag, not a separate tool — keeps the surface minimal while making
  cross‑project a first‑class action.
- Background consolidation is automatic; `mnemo consolidate` is only for manual runs/debugging.
