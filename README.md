# mnemo

**Local memory for AI coding agents. On‑demand. Strictly offline. Built for 10+ agents on a 16 GB machine.**

## What it is

A persistent memory layer for agents (Claude Code, Cursor, Windsurf, any MCP client) that
remembers decisions, bugs, progress, and rules across sessions. What sets it apart:

- **strictly local** — zero cloud calls; embeddings and LLM run only on your machine;
- **on‑demand** — nothing runs in the background; the service spins up under load and shuts down after a grace period;
- **no Docker daemon** — embedded storage inside a single process (SQLite + `sqlite-vec`);
- **lightweight** — ~1 GB RAM while active, ~0 when idle; a small model (Qwen3‑4B / Gemma 4) runs only during background consolidation;
- **concurrent** — one shared service serves 10+ agents; writes are cheap (embed + insert, no LLM on the hot path);
- **simple** — a small set of MCP tools (`remember` / `search` / `browse` / `recall` / `delete`, plus project management); project‑scoped reads with first‑class cross‑project search on request (`scope=all`).

## Core principles

1. **No LLM on the write path.** A write = local embedding + upsert. The LLM runs only in the background, in batches.
2. **One shared process, started on demand.** Not 10 stdio processes hitting one file — one service + thin shims.
3. **Heavy things are transient.** The generative model is loaded only for a consolidation window, then unloaded.
4. **Typed memory, project scoping.** `decision / debug / progress / rule / ...` across `project` / `global` / (optional) `session` scopes — reads isolate by project (plus always‑visible `global`); cross‑project search is first‑class but opt‑in (`scope=all`).
5. **Tiny, obvious API.** Two everyday verbs (`remember` / `search`) plus a query‑less `browse`; deletion (`delete` / `delete_project`) is on the agent surface and the CLI, while wiping everything (`purge`) is CLI‑only; type/scope are parameters, not extra tools.
6. **The default path works after one install.** The pplx and recall runtimes ship together; additional specialist models must earn their place.
