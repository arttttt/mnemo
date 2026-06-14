# 10 — Roadmap (index)

The MVP is **all of Phases 0–4**. Everything under Post‑MVP lands **after Phase 4**.

Each phase has its own file with small, independently verifiable steps. Every step is written as:

- **What** — the goal, in a line or two.
- **Depends on** — which earlier steps must be done first.
- **Done when** — a concrete, checkable condition (a passing test or an observable behavior).

A step is sized to ship and review on its own. No implementation/file details here — only behavior and verification.

## Phases

| Phase | Goal | File |
|---|---|---|
| 0 — Walking skeleton ✅ | store + search locally via MCP | [roadmap/phase-0-skeleton.md](roadmap/phase-0-skeleton.md) |
| 1 — Memory layer ✅ | typed, scoped, persistent memory; hybrid search; sessions; deletion; supersede | [roadmap/phase-1-memory-layer.md](roadmap/phase-1-memory-layer.md) |
| 1b — Store re‑platform | LanceDB → **SQLite + `sqlite-vec` + FTS5**; typed links table folds in | [roadmap/sqlite-migration.md](roadmap/sqlite-migration.md) |
| 2 — On‑demand lifecycle + concurrency | one shared service, spins up/down, ~0 idle RAM; the 10+‑agent write model | [roadmap/phase-2-lifecycle.md](roadmap/phase-2-lifecycle.md) |
| 3 — Background consolidation | small model improves memory off the hot path, concurrent from the start | [roadmap/phase-3-consolidation.md](roadmap/phase-3-consolidation.md) |
| 4 — Polish & optional upgrades | reranker/NER, export/import, packaging, air‑gapped | [roadmap/phase-4-polish.md](roadmap/phase-4-polish.md) |
| Post‑MVP | bi‑temporal, importance, revision tooling, session scope | [roadmap/post-mvp.md](roadmap/post-mvp.md) |

## Definition of done for the MVP
Phases 0–4 complete; runs on a 16 GB machine with 10+ agents, strictly offline, on‑demand (no resident
daemon, no Docker), on SQLite + `sqlite-vec` + FTS5; an agent works through `remember` / `search` / `delete` (`recall` is post‑MVP).
