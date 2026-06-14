# Sub‑phase — Re‑platform the store to SQLite

Between the memory layer ([phase-1-memory-layer.md](phase-1-memory-layer.md)) and the architecture work: swap
the store engine from LanceDB to **SQLite + `sqlite-vec` + FTS5**. Rationale and alternatives are in
[adr/0001-storage-engine.md](../adr/0001-storage-engine.md) — in short, mnemo's mutable typed core is OLTP‑shaped
(in‑place updates, atomic supersede, point lookups, edges) and SQLite fits it on **correctness and convenience**;
the vector index is just one feature, and brute‑force `sqlite-vec` is ample at our scale.

Each step is **Why** · **What** · **Done when**. Everything stays behind `MemoryRepositoryPort` — domain, use
cases, MCP/CLI, and the parametrized store‑contract tests do **not** change.

---

### S.1 SQLite store adapter behind the port

**Why.** The whole point of the port: replace the engine without touching the core.
**What.** A `sqlite-vec`‑backed repository: a single relational `memories` table carrying the payload (type,
scope, project, tags, related_files, hash, status, topic_key, session_id, timestamps, counters) **and** the
embedding as a `BLOB` column on the same row (`CHECK(vec_length(...))` dimension guard) — ranked by the
`vec_distance_cosine` scalar, not a `vec0` virtual table (see [the ADR](../adr/0001-storage-engine.md#why-a-blob-column--scalar-distance-not-a-vec0-virtual-table)); FTS5 over `content` as an external‑content table
synced by triggers. Exact lookups by hash / topic_key use indexed columns; `UPDATE` for supersede/duplicate‑counter;
the in‑memory store stays the offline/test backend.
**Done when.** The SQLite store passes the **same store‑contract tests** as in‑memory (both backends remain
parametrized); selecting it is one config switch.

### S.2 Hybrid search = RRF in SQL

**Why.** Dense + lexical retrieval, the same behaviour as before — just expressed in SQL instead of LanceDB's
built‑in hybrid.
**What.** Two queries — dense (`vec_distance_cosine` over the `WHERE`‑filtered scan) and lexical (FTS5 BM25) —
fused by reciprocal‑rank fusion (k=60) **in the adapter** (a pure function over the two ranked id lists). The
structured filters (scope / type / tags / related_files / recency) are pushed down as `WHERE` into both legs;
list filters (`tags`, `related_files`) use `json_each`. (Every analog surveyed fuses in code, not in one SQL
statement — it's simpler and trivially unit‑testable.)
**Done when.** A paraphrase and an exact token both return the right memory; every filter narrows correctly;
the store‑contract search tests pass on SQLite.

### S.3 Typed links table (folds in 1.9)

**Why.** On SQLite, relationships are a native table — so the deferred "links + provenance" work lands here for
free, deterministically (no inferred knowledge graph; see the ADR).
**What.** A `links` table `(source_id, target_id, type, provenance, created_at)`. The **supersede** edge is
written automatically on a `topic_key` upsert, with provenance. The shape is generic so the background worker's
later flag‑only edges fit without a migration. `derived_from` / `source_ids` stay **post‑MVP**.
**Done when.** A `topic_key` upsert writes a `supersedes` edge with provenance; a memory's links are retrievable;
tests cover it.

### S.4 Wire as default, drop LanceDB, update config & dev script

**What.** Make SQLite the default store; remove the `lancedb` dependency, add `sqlite-vec`; replace
`MNEMO_LANCEDB_URI` with a single SQLite file path; update `dev.sh` and the env docs ([13-usage.md](../13-usage.md)).
**Done when.** Default runtime uses SQLite; `lancedb` is gone from dependencies; docs match the code.

### S.5 Migrate the dogfooding data

**What.** Move the existing live memories into the SQLite store (a one‑time, idempotent step) — or, since the
project is new and small, simply re‑seed. Never silently drop data.
**Done when.** The live store runs on SQLite with the existing memories intact (or knowingly re‑seeded).

### S.6 Integration tests (mandatory, same PRs)

**What.** Per [12-testing.md](../12-testing.md): the new adapter and the MCP/CLI exercised end‑to‑end on SQLite,
in the same PRs as the change — not a follow‑up.
**Done when.** Offline and heavy suites are green on the SQLite backend.

---

**Out of scope here:** concurrency (an *architecture* concern — [03-architecture.md](../03-architecture.md));
`recall` and `derived_from` (post‑MVP); ANN (libSQL/DiskANN is the future upgrade path).
