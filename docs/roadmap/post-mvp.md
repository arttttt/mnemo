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

### `recall(project)` — aggregated session context
**Why:** one call that returns a concise "here's where you are" (rules + what matters now) instead of forcing
several searches.
**What:** an aggregated context bundle. Deferred because a *useful* recall — concise, not a context dump —
needs **LLM synthesis**, and the read path stays LLM‑free in the MVP. A post‑MVP design can precompute a
digest in the background worker (off the hot path) and have `recall` just read it. In the MVP, the agent
retrieves on demand with `search` (`type=rule` for rules, `type=progress` for where it left off).

---

## Retrieval & recall surface — problems found by dogfooding

Using mnemo on its own memory surfaced one consistent signal: the write/storage side is mature, but the
**read/recall** surface lags ("dump the corpus" more than "precise answer"). These are committed as post‑MVP
research/improvements. Sources are the tagged feedback memories [[feedback/mcp-retrieval-ux]] and
[[feedback/type-discipline]], re‑validated live on the SQLite + `sqlite-vec` + bge‑small path.

### Self‑describing relevance score
**Why:** the `search` `score` is a reciprocal‑rank‑fusion value (k=60, ≈`1/(60+rank)` per channel), so every hit
sits in a narrow ~0.016–0.033 band and is **not** a similarity / confidence. A consuming agent naturally misreads
it as relevance confidence — in dogfooding it caused a *false* diagnosis ("weak embedder") when the embedder was
fine. The sharpest single recall wart: opaque **and** misleadable.
**What:** make the signal interpretable — label `score` as RRF in the tool/result schema, and/or also return the
raw per‑channel similarity + rank, and/or normalise to a documented [0,1]. Research which form a consuming agent
actually uses as a threshold.

### Graph navigation at the MCP surface (`get` / `neighbors`)
**Why:** memory is densely linked (`[[topic_key]]` wikilinks) and a typed `links` table now exists in storage
(`add_link` / `links_for`), but there is no way to *traverse* it from the tools — reaching a linked memory means
another semantic search and hope. The graph lives in the data, not the interface.
**What:** add `get(id | topic_key)` (exact fetch, including the supersede chain) and `neighbors(id)` (the typed
edges out/in, single hop — not multi‑hop inference) MCP tools that read the existing `links` table. Pure
interface, no new storage; stays within the deterministic typed‑edge graph (not a knowledge graph).

### Query‑less browse / list mode
**Why:** `search` requires a `query`, so "all `type=decision` in this project, newest first" can't be expressed
without inventing a query that itself biases ranking. Retrieving a *category* (e.g. `tags=["feedback"]`) shouldn't
need a semantic guess.
**What:** allow `search` with an empty/optional `query` — a pure filter (type / tags / scope / recency) ordered by
recency, with no relevance ranking. A browse path distinct from semantic search.

### Ranking quality on the hybrid path
**Why:** a generic "status/next‑steps" memory floats to the top of unrelated queries (observed again on the
SQLite + bge‑small path: a storage‑engine query returned the status note above the ADR). At a small `limit` this
displaces the right answer.
**What:** investigate the cause (document length, or one over‑broad catch‑all memory dominating an RRF channel)
and tune — per‑channel weighting, length normalisation, or down‑weighting catch‑all memories.

### Retrieval robustness to write‑time mis‑typing
**Why:** a normative memory mis‑stored as `type=learning` instead of `rule` is silently invisible to the precise
`type=rule` query ([[feedback/type-discipline]]). The root cause is a write‑time data error, not a retrieval bug,
and a query‑side workaround ("also search `type=learning`") would only pollute the access pattern.
**What:** fix at the source via a **re‑type op in the background consolidation worker**: under the determinism
axiom, re‑typing is a cheap, reindex‑free, reversible filter‑facet correction — lean auto‑apply with
`provenance=llm` and a before→after log, rather than flag‑only (which would create review noise for every typo).
Settle the policy when the worker's op set is designed.

### Deferred indefinitely
**Why:** out of the local, single‑user, lightweight scope mnemo targets.
**What (not doing unless the scope changes):** knowledge graph / multi‑hop traversal; web dashboard; document/PDF
ingestion; multi‑user / RBAC / cloud sync.
