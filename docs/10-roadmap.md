# 10 — Roadmap

Phased plan. Each phase is shippable and testable on its own.

## Phase 0 — Skeleton (walking skeleton)
**Goal:** an agent can store and search memory locally via MCP.
- [ ] Project scaffold (Python, `uv`, package layout, CLI via Typer).
- [ ] Embedded store wrapper (LanceDB) — create/upsert/query, payload schema from [04](04-data-model.md).
- [ ] Embedder wrapper (fastembed/ONNX, `bge-small-en-v1.5`), dimension pinned at init.
- [ ] FastMCP service with `store_memory` and `search` (semantic only).
- [ ] stdio shim → shared http service; manual start (no lifecycle yet).
- [ ] CLI: `mnemo store/search/stats`.
- **Done when:** Claude Code can `store_memory` and `search` against a local store, no cloud.

## Phase 1 — Memory layer (v1 core)
**Goal:** typed memory with scoping, dedup, session context.
- [ ] All `store_*` wrappers + typed payload, importance, tags, related_files.
- [ ] Per‑project scoping + `__global__`; `get_rules`/`store_rule`.
- [ ] Hot‑path dedup (hash + cosine threshold + `topic_key` upsert).
- [ ] Hybrid search (dense + lexical/FTS, RRF) + filters.
- [ ] `recall(project)` + `session_recap` + sessions table.
- [ ] `expand`, `update_memory`, `inactivate`, `supersede`.
- [ ] Internal write queue/lock; verify 10+ concurrent agents (no lost writes, no "db locked").
- **Done when:** the requirements FR‑1..FR‑13 hold; 10‑agent concurrency test passes.

## Phase 2 — On‑demand lifecycle
**Goal:** nothing resident; spins up/down per the [07](07-lifecycle-and-ram.md) scheme.
- [ ] Ref‑counting in the shim; grace‑timer shutdown.
- [ ] Socket activation: launchd plist (macOS) + systemd socket/service (Linux).
- [ ] `mnemo init <client>` writes MCP config + installs activation.
- [ ] RAM budget verification (idle ~0, active ~1 GB).
- **Done when:** NFR‑5..NFR‑8 verified on macOS and Linux.

## Phase 3 — Background consolidation
**Goal:** the small LLM improves memory off the hot path.
- [ ] Generator wrapper (llama.cpp, `Qwen3-4B-Instruct-2507` Q4) with on‑demand load/unload.
- [ ] GBNF/guided‑JSON contract; flat schema; `temperature=0`.
- [ ] Triggers (by volume / idle / manual); batch selection by embedding neighborhood.
- [ ] Operations: near‑dup merge, cluster summarize, insight extraction, staleness marking.
- [ ] Idempotency + failure isolation; consolidation log.
- [ ] `MNEMO_GENERATOR=off` degradation path (cosine‑only).
- **Done when:** FR‑14/FR‑15 hold; generator RAM is transient; consolidation never blocks writes.

## Phase 4 — Polish & optional upgrades
- [ ] Optional reranker (`Qwen3-Reranker-0.6B`) and GLiNER2 dedup.
- [ ] `tasks` API.
- [ ] Export/import (portability).
- [ ] Packaging (`uvx`/`pipx`), docs, install guides per IDE.
- [ ] Air‑gapped mode (pre‑seeded model cache).

## Explicitly deferred (post‑v1)
- `session` scope (transient session‑keyed memory) — cheap to add later; use `working-notes` for now.
- Knowledge graph / multi‑hop; web dashboard; document/PDF ingestion; multi‑user/RBAC/cloud sync.

## Definition of done for v1
FR‑1..FR‑16 (MUST/SHOULD) + NFR‑1..NFR‑13 satisfied; works on a 16 GB machine with 10+ agents,
strictly offline, no resident daemon, no Docker.
