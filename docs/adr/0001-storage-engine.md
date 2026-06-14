# ADR 0001 — Storage engine: SQLite + sqlite-vec + FTS5

**Status:** Accepted (2026-06-14). Supersedes the earlier choice of LanceDB.

## Context

mnemo is, at its core, a **mutable, evolving typed store** that also needs vector search — not an
append-only vector warehouse. Its hot path and its evolution model are dominated by relational/transactional
operations, with similarity search as *one* feature among them:

- **In-place updates on every write.** Exact-duplicate writes bump a counter (`duplicate_count`,
  `last_seen_at`); a `topic_key` upsert flips the prior record to `status: superseded`. These are routine,
  not occasional.
- **Atomic two-record evolution.** "Mark the old record superseded **and** insert the successor" must be one
  atomic unit, or a crash leaves a torn state (old superseded, successor missing).
- **Point lookups on the hot path.** Every write does an exact lookup by content-hash and (for upsert) by
  `topic_key` + project.
- **Relationships.** Supersede chains today; deterministic typed edges + provenance next; the background
  worker will *flag* associative/contradiction candidates later. Light, but relational.
- **Hybrid retrieval.** Dense (vector) + lexical (BM25) fused by reciprocal-rank fusion.

This shape is **OLTP**: frequent point reads/writes, in-place mutation, small atomic transactions, an edges
table — at modest scale (single user; thousands to low-hundred-thousands of records).

LanceDB (the previous choice) is the opposite shape: an embedded **analytical** vector store on an
append-only, columnar, MVCC format. It is excellent at vector ANN + full-text at scale, but for mnemo's
mutable/relational core it is structurally awkward:

- **No multi-row atomic transaction** — a transaction carries one operation, so supersede-then-insert is two
  separate commits (torn-state risk on crash).
- **In-place updates rewrite fragments and accumulate versions**, dropping rows out of indexes and requiring a
  periodic `optimize()`/compaction regime.
- **No native joins / edges / traversal** — relationships must be modeled in app code.
- Point lookups rely on scalar indexes that are not live until `optimize()`.

(These are documented behaviours of the Lance format and LanceDB OSS, not speculation.)

## Decision

Use **SQLite + the `sqlite-vec` extension + FTS5** as the single embedded store, behind the existing
`MemoryRepositoryPort`.

- **Relational core, edges, transactions, point lookups, in-place updates** — native to SQLite.
- **Vector search** — `sqlite-vec`, brute-force: the embedding is a **`BLOB` column on the `memories` row**
  (with a `CHECK(vec_length(...))` dimension guard), ranked by the `vec_distance_cosine` scalar over a
  `WHERE`-filtered scan — **not** a `vec0` virtual table (rationale below). At mnemo's scale this is well under
  ~100 ms; ANN is unnecessary.
- **Lexical search** — FTS5 (BM25), built into SQLite, as an external-content table over `content` kept in sync
  by triggers.
- **Hybrid** — reciprocal-rank fusion (k=60) computed **in the adapter** over the two ranked result lists
  (dense `vec_distance_cosine` + FTS5 BM25).
- **One embedded file**, no daemon, ~0 idle RAM — fits the on-demand / strictly-offline axioms even better than
  LanceDB.

### Why a `BLOB` column + scalar distance, not a `vec0` virtual table

`sqlite-vec` officially supports both, framing it as a trade-off (`vec0` is faster/more compact but needs JOINs
and a restricted filter set; the manual `BLOB` + `vec_distance_*` scan is more flexible). For mnemo's *own* shape
the manual scan is the better fit on **correctness** and **convenience**, and the only thing `vec0` buys —
speed at scale — is precisely what we don't need (single user; thousands–low-hundred-thousands of records; no
hot-path latency budget):

- **Filters are correct by construction.** Our retrieval filters include **list-valued** ones (`tags` ALL,
  `related_files` ANY). `vec0` metadata columns only support scalar operators (`= != < <= > >=`) — no arrays,
  no `json_each`, no `LIKE` — so array/recency filters can't be pushed into the KNN; `vec0` users resort to
  over-fetch-then-post-filter (which can silently return fewer than *k*) or a single `rowid IN (...)` subquery.
  With the embedding as a column on `memories`, **every** filter — scalar and array (`json_each`) — is a plain
  `WHERE` in the same SELECT that computes the distance, so ranking happens over exactly the filtered set.
- **Single-table atomicity.** The embedding living on the `memories` row means supersede+insert, the duplicate
  counter bump, and hard delete are all **one table's** transaction. A `vec0` table is a *separate* table that
  must be kept in `rowid`-sync through every one of those mutations (the `rowid = id` contract, a `SAVEPOINT`
  around row+vector writes, and a soft-/hard-delete asymmetry are exactly the complications this avoids). This
  directly serves the OLTP-correctness thesis above.

This was decided after surveying seven embedded sqlite-vec/SQLite memory projects and the official `sqlite-vec`
and SQLite docs; the closest mutable-memory analogs (the two `engram` projects) independently use the same
`BLOB` + scalar-cosine + external-content-FTS + RRF(k=60) shape. The choice rests on fit for our shape, not on
which option is more common (the field is split ~3:2 toward `vec0`).

## Why (correctness + convenience — not "what others do")

The decision rests on fit for mnemo's *own* shape, not on ecosystem popularity:

| Criterion (already exercised in the MVP) | LanceDB | SQLite + sqlite-vec |
|---|---|---|
| **Correctness:** atomic supersede + insert | two commits, torn-state on crash | one transaction |
| **Convenience:** in-place update (dup counter, status flip) | fragment rewrite + version churn + `optimize()` | trivial `UPDATE` |
| **Convenience:** point lookup (hash / topic_key) on write path | scan / index not live until optimize | indexed, O(log n) |
| Relationships (supersede pointer, later typed edges) | denormalized, no joins | native edge table (no mandatory traversal) |
| Hybrid (dense + lexical) | built-in | RRF in SQL (standard, small) |
| Vector ANN at millions | mature | brute-force (not needed at our scale) |

Migration cost and prior investment are **not** weighed: the project is new and carries no data baggage.

## Alternatives considered

- **Stay on LanceDB.** Rejected: its only edges over SQLite (mature ANN, built-in hybrid) are unneeded at our
  scale or trivially reproduced, while its append/analytical shape is wrong for a mutable evolving core.
- **LanceDB + a second relational store (two stores).** Rejected: dual writes with no cross-store transaction,
  and it breaks co-location (pushed-down filters and hybrid require vectors + metadata + FTS in one place).
- **DuckDB + VSS.** Rejected: OLAP/columnar (poor for in-place OLTP writes); no multi-process access; the VSS
  index is experimental (no WAL recovery, "not for production").
- **Postgres + pgvector.** Rejected: a resident server/daemon — violates the on-demand, no-daemon, ~0-idle
  axiom.
- **libSQL / Turso (SQLite fork with native DiskANN ANN, MIT).** Kept as the **upgrade path**: a drop-in if we
  ever outgrow brute-force (~1M+ vectors), preserving every SQLite advantage. We start on stock SQLite +
  sqlite-vec (standard `sqlite3` + one extension; simpler).

## Consequences

- **Contained to the adapter.** Domain, use cases, MCP/CLI, and the parametrized store-contract tests do not
  change — the port pays off. Only the store adapter (`LanceDb…` → `SqliteVec…`), the hybrid step (→ RRF in
  the adapter over two SQL queries), and dependencies change.
- **`links` becomes a native table** — the typed-edges work (roadmap 1.9) folds into this re-platform instead
  of being built on LanceDB.
- **Concurrency is out of scope here.** The write-concurrency model depends on the *target architecture*
  (shared on-demand process + thin per-agent shim vs. today's process-per-agent), which is a later phase. It
  moves to the architecture section; this ADR does not decide it.
- **Accepted risk:** `sqlite-vec` is pre-v1 (breaking changes possible) and brute-force only. Mitigated by our
  scale and the libSQL escape hatch.

## Not a knowledge graph

This store gives mnemo a **graph** (typed edges between memories) but deliberately not a **knowledge graph**:
edges are created only on explicit signals (supersede / `derived_from`) or human/agent-confirmed worker flags —
never auto-inferred-and-trusted. That boundary is a project axiom, independent of the storage engine.
