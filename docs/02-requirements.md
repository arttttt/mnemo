# 02 — Requirements

Notation: **MUST** — required for v1, **SHOULD** — desirable, **MAY** — optional/later.

## Functional requirements

### Memory storage
- **FR‑1 (MUST).** Store typed "memories" (see [04-data-model.md](04-data-model.md)):
  `decision, debug, progress, feature, research, code-snippet, rule, learning, discussion, design, working-notes`.
- **FR‑2 (MUST).** Each memory holds: `content`, `type`, `project`, `related_files[]`, `tags[]`,
  `importance`, `session_id`, `created_at`, `hash`.
- **FR‑3 (MUST).** Scopes: `project` (local) and `global`. Isolation is **soft** — projects organize memory,
  they are not a search wall. (A `session` scope is deferred — post‑v1, cheap to add later.)
- **FR‑3b (MUST).** Cross‑project search: retrieval can span all projects (`scope="all"`). The default scope
  (current project + global) may also surface strongly‑relevant cross‑project hits, ranked lower and labeled.
- **FR‑4 (SHOULD).** Dedup on write: do not spawn near‑duplicates (hash + cosine threshold).
- **FR‑5 (MAY).** Soft delete/inactivation (recoverable), audit trail.

### Search and retrieval
- **FR‑6 (MUST).** Semantic search over local embeddings.
- **FR‑7 (SHOULD).** Hybrid: dense + lexical (BM25/sparse), merged via RRF.
- **FR‑8 (MUST).** Filtering by `scope`, `project`, `type`, `related_files`, `tags`, recency.
- **FR‑9 (SHOULD).** Rerank top candidates (optional local reranker).

### Sessions and context
- **FR‑10 (MUST).** `recall(project)` at session start: rules + recent activity + pending + recap.
- **FR‑11 (SHOULD).** `session_recap()` — where we left off last time.
- **FR‑12 (MAY).** Tasks linked to memories.

### Rules
- **FR‑13 (SHOULD).** Per‑project and global rules (`rule`), loaded at session start.

### Background processing
- **FR‑14 (SHOULD).** Periodic consolidation by a small LLM: dedup‑merge, summarization, insight extraction.
- **FR‑15 (MUST).** Consolidation is NOT on the write hot path and does NOT block agents.

### Integration
- **FR‑16 (MUST).** Connect via MCP to Claude Code / Cursor / Windsurf / any MCP client.
- **FR‑17 (SHOULD).** A CLI for debugging (`mnemo search/store/stats/consolidate`).

### API simplicity
- **FR‑18 (MUST).** Minimal, obvious agent‑facing API **driven by real tasks — not a fixed number of tools**.
  Prefer behavior via parameters (type/scope/filters) over a tool per type or per operation; add a tool only
  when a real task needs it (e.g. `recall`, `remember`, `search`, and later a revision affordance). Minimal ≠ exactly two.
- **FR‑19 (MUST).** Operational commands (`stats`, `consolidate`, `doctor`, export/import) live in the CLI,
  NOT in the agent‑facing MCP surface.

## Non‑functional requirements

### Privacy / locality
- **NFR‑1 (MUST).** Zero outbound network calls at runtime (except a one‑time model weights download at install).
- **NFR‑2 (MUST).** Embeddings and the LLM run locally only (ONNX / llama.cpp / Ollama). No cloud API keys.
- **NFR‑3 (SHOULD).** Air‑gapped mode: everything works without internet after the initial install.
- **NFR‑4 (MUST).** No telemetry.

### Lifecycle / resources (see [07-lifecycle-and-ram.md](07-lifecycle-and-ram.md))
- **NFR‑5 (MUST).** On‑demand: the service process exists only while ≥1 agent is working.
- **NFR‑6 (MUST).** Grace shutdown: after the last agent disconnects, the service exits after a configurable
  timeout (default ~5 min); if a new agent connects within it, the service stays up.
- **NFR‑7 (MUST).** No permanently running Docker daemon. Storage is embedded (in‑process).
- **NFR‑8 (MUST).** RAM while active ≤ ~1.5 GB (without the generator); ~0 when idle.
- **NFR‑9 (MUST).** The heavy generative model is loaded only for the consolidation window, then unloaded.
- **NFR‑10 (MUST).** Correct on a 16 GB machine, including when the coding agent itself runs alongside.

### Concurrency
- **NFR‑11 (MUST).** Correct simultaneous operation of 10+ agents against the shared service.
- **NFR‑12 (MUST).** Writes are cheap: embedding + upsert, no LLM. Concurrency is resolved inside one process.
- **NFR‑13 (SHOULD).** No lost writes under races (internal queue/lock, no "database is locked").

### Performance
- **NFR‑14 (SHOULD).** `store_*` latency (hot path) — tens of ms (embedding a short text on CPU).
- **NFR‑15 (SHOULD).** `search` latency — < ~200 ms at typical volume (tens to hundreds of thousands of records).

### Operations
- **NFR‑16 (SHOULD).** Install in 1–2 commands (`uvx` / `pipx` / `npx`), no manual DB setup.
- **NFR‑17 (SHOULD).** Data in a single directory (`~/.mnemo/`), easy to back up / move / delete.
- **NFR‑18 (MAY).** Export/import of memory (portability, no lock‑in).

### Engineering principles (code quality)
- **NFR‑19 (MUST).** Follow **SOLID**, **DRY**, **KISS** throughout. Prefer the simplest design that meets the
  requirement; do not over‑engineer.
- **NFR‑20 (MUST).** Separate architectural layers per **Clean Architecture**:
  - **domain** — entities & pure business rules (no framework/IO deps);
  - **application** — use cases + ports (interfaces); depends only on domain;
  - **adapters** — store / embedder / MCP / CLI implementations of the ports;
  - **infrastructure / composition root** — config, wiring (DI), entrypoints.
- **NFR‑21 (MUST).** Dependency rule: dependencies point **inward** (Dependency Inversion). Domain and use cases
  must NOT import MCP, LanceDB, fastembed, llama.cpp or any framework — those live behind ports.
- **NFR‑22 (SHOULD).** One reason to change per component (SRP); new stores/embedders/models are added as new
  adapters (OCP) without touching the core. Layout in [09-tech-stack.md](09-tech-stack.md).

### Testing (minimum bar)
- **NFR‑23 (MUST).** **Unit tests** cover the inner layers in isolation — domain rules and application use cases —
  using in‑memory/fake ports (no real MCP/DB/model).
- **NFR‑24 (MUST).** **Integration tests**, in separate suites, cover each adapter against its real boundary:
  store persistence (round‑trip), embedder, MCP controller, CLI, and the composition root.
- **NFR‑25 (SHOULD).** Tests run **offline and fast by default** (hash embedder + in‑memory/JSON store). Heavy or
  networked backends (fastembed, lancedb) sit behind an opt‑in marker. Strategy in [12-testing.md](12-testing.md).

## Core axiom constraints (what sets this project apart)

1. **No LLM on the write path.**
2. **No always‑on daemon / no Docker.** Embedded store + on‑demand service.
3. **Small model only** (≤4–8B), and even that one runs transiently, in the background.
4. **16 GB is the target bar**, not "32 GB + GPU recommended".
5. **Minimal, clear agent API** + **soft project isolation** with first‑class cross‑project search.
6. **Extra models are opt‑in.** Default = embedder + (optional) one generator; any additional small model
   must be justified by a measured gain or enabled explicitly.
7. **Clean layering.** SOLID/DRY/KISS; domain & use cases are framework‑agnostic behind ports.
