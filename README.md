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
- **simple** — 3 core MCP verbs (`recall` / `remember` / `search`); soft project scoping with first‑class cross‑project search.

## Core principles

1. **No LLM on the write path.** A write = local embedding + upsert. The LLM runs only in the background, in batches.
2. **One shared process, started on demand.** Not 10 stdio processes hitting one file — one service + thin shims.
3. **Heavy things are transient.** The generative model is loaded only for a consolidation window, then unloaded.
4. **Typed memory, soft scoping.** `decision / debug / progress / rule / ...` across `project` / `global` / (optional) `session` scopes — projects organize memory but never wall off search; cross‑project search is first‑class.
5. **Tiny, obvious API.** Three verbs the agent learns in seconds; type/scope are parameters, not extra tools. Ops commands live in the CLI.
6. **Extra models are opt‑in.** Default = embedder + one optional background generator; more small models only when justified.
