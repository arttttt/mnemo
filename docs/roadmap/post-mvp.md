# Post‑MVP (after the initial scope)

These are committed for later. They are deliberately **not** in the initial scope, but the MVP is designed so
they add cleanly (e.g. the schema is forward‑compatible).

Format: **Why** (the requirement) · **What** (exactly what to do).

---

### Full bi‑temporal validity model
**Why:** answer "what was true / what did we know as of date X" and support retro‑corrections, without deleting
history — the only rigorous way to handle changing facts over time.
**What:** add transaction‑time (`created_at` / `expired_at`) and valid‑time (`valid_from` / `valid_to`) on records,
with point‑in‑time queries. Done **in full** (no half‑measures); the schema is already forward‑compatible.

### `importance` reintroduced
**Why:** prioritise and age memory in retrieval.
**What:** blend importance + recency into ranking, add decay, and optionally auto‑score importance in consolidation.

### Revision tooling for flagged contradictions
**Why:** the human/agent decides what is stale — the system never auto‑invalidates; they need a way to act on flags.
**What:** a review/resolve flow over the contradictions the background worker flagged.

### `session` scope
**Why:** some context is only relevant within a session.
**What:** a `session` scope value + the matching search filter.

### Deferred indefinitely
**Why:** out of the local, single‑user, lightweight scope mnemo targets.
**What (not doing unless the scope changes):** knowledge graph / multi‑hop traversal; web dashboard; document/PDF
ingestion; multi‑user / RBAC / cloud sync.
