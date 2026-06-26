# mnemo

Local‑first memory for AI coding agents — typed, deterministic, project‑scoped.

## What it is

A persistent memory layer for AI coding agents (Claude Code, Cursor, Windsurf, any MCP client) that
remembers decisions, bugs, progress, and rules across sessions, so you don't re‑explain your project
every time.

- **local‑first** — the embedder and the optional `recall` model run on your machine; external model
  providers are a possible option, not a requirement;
- **deterministic** — a write is a local embedding + insert, with no LLM in the loop. mnemo never runs
  a model over your memories to extract, merge, or summarize them, and nothing rewrites them in the
  background; a stored memory changes only on an explicit `supersede` / `topic_key` signal;
- **on‑demand** — nothing runs in the background; the shared service starts under load and exits after
  an idle grace period;
- **no Docker, no external DB** — the whole store is one process over SQLite + `sqlite-vec` + FTS5;
- **typed & project‑scoped** — `decision / progress / rule / learning / research / working-notes`,
  scoped per project, with first‑class cross‑project search on request (`scope=all`);
- **small MCP surface** — one write (`remember`) and four reads (`search` by meaning, `browse` by
  filter, `get` by id/topic_key, `recall` for an LLM‑synthesized answer), plus `delete` and project
  tools.

## How writes work

A write is a local embedding + insert — no LLM on the path. Many memory tools run an LLM on every
write to extract or summarize what was said, and some keep rewriting it in the background; mnemo
doesn't, so what you store is what you get back. The only LLM in the system is the opt‑in `recall`
read tool: it loads a small model on demand to synthesize an answer over retrieved memories, then
unloads it, and it never changes what's stored (`recall` is gated behind an explicit `force` flag).

## Retrieval quality

Retrieval is tested in‑repo (`tools/eval/`) against public benchmarks (LoCoMo, LongMemEval) and a
real project‑fact set, where it compares favorably with other open‑source memory servers on Recall@k
and abstention.

## Install & use

See [docs/13-usage.md](docs/13-usage.md) for install, the CLI, and one‑command MCP client setup
(`mnemo setup`). The full tool surface is in [docs/05-mcp-api.md](docs/05-mcp-api.md).
