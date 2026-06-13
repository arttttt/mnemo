# 11 — Alternatives Research

Why build instead of adopting. Survey of existing local memory solutions (as of June 2026) against
our axioms: strictly local · on‑demand · no Docker daemon · no LLM on the write path · 16 GB · 10+ agents · typed layer.

## Summary verdict

Every "ready" option fails at least one hard axiom, or it's a bare store you'd build the memory layer
on top of anyway. The memory‑layer logic we want is little code; the heavy parts (store, embeddings,
inference) are commodity components. Hence a thin custom build.

## Disqualified by a hard axiom

| Solution | Killer for us |
|---|---|
| **cognee** | Needs a 32B+ model for a clean graph; heavy LLM‑driven `cognify` on **every** write; default file DBs serialize concurrent writes (SQLite deadlock history). Too complex; LLM‑per‑write. |
| **mem0 (OSS)** | LLM call on every `add()` (fact extraction) → bottleneck on one local engine with 10+ agents. PostHog telemetry by default. |
| **Letta (ex‑MemGPT)** | Memory is mediated by the agent's own LLM loop; heaviest to self‑host (server + Postgres). LLM‑per‑write in spirit; not "simpler". |
| **Eion** | Powerful (Postgres+pgvector + Neo4j knowledge graph, multi‑agent RBAC), but Neo4j (JVM) is a heavy **resident** process — exactly the RAM "hog" we avoid. |
| **Memary** | Effectively abandoned (no releases since Oct 2024), no MCP. |
| **Memobase** | Heavy Docker stack (Postgres+Redis), embedding API by default; user‑profile memory, not code‑oriented. |
| **Recallium** (the inspiration) | Closed‑core Docker image; almost certainly does LLM work on/after write by default; an always‑on container. |

## Resident‑daemon / Docker (conflict with on‑demand, no‑Docker)

| Solution | Note |
|---|---|
| **Qdrant + mcp‑server‑qdrant** | Excellent concurrent Rust server, local fastembed, no LLM on write — but a server/Docker daemon, and a **bare** store (just `qdrant-store`/`qdrant-find`, no typed layer). We borrow the architecture idea (single shared store) but embed it instead. |
| **Chroma server + chroma‑mcp** | Good concurrent writes (log‑batched), local ONNX embeddings, no LLM on write — but a resident server; bare store; the MCP wrapper is low‑velocity. |
| **txtai (service)** | Local, built‑in MCP, no LLM on write, cheap upserts — but writes aren't concurrency‑safe (single worker + serialize), and it's a bare store; needs pgvector for true multi‑writer. |
| **Annal** | Architecturally ideal (Qdrant daemon, ONNX, no‑LLM write) but installs as an always‑on systemd/launchd **service** (resident) and is alpha/0★. |
| **Ogham MCP** ⭐ | Closest to our ideal: Postgres+pgvector, shared SSE daemon, ONNX/Ollama local embeddings, **no LLM in the write path** (regex), typed store tools. We use it as a **reference**, but it's a resident Postgres daemon and young (~109★). |

## stdio‑per‑agent + single SQLite (concurrency caveat at 10+)

| Solution | Note |
|---|---|
| **engram (Gentleman‑Programming)** | Go binary, SQLite + FTS5, typed memory + sessions, no LLM on write — but **lexical only** (no embeddings), and N stdio processes on one file: WAL+busy_timeout+retry help, yet an open data‑loss bug (#477, shared‑WAL split‑brain) and stale cross‑process reads (#206) remain. Mitigation is to funnel all agents to one shared `engram serve` — i.e. our single‑shared‑process idea. |
| **claude‑mem‑lite** | Single SQLite, FTS5+TF‑IDF (lexical), but the **default auto‑capture calls Haiku** (LLM on write); Claude‑Code‑only; stdio‑per‑agent. |
| **llm‑wiki‑memory** | On‑device BGE embeddings, git‑markdown wiki, no LLM on the direct write tool — but the auto‑loop (flush/compile/consolidate) **calls an LLM**, and 10+ writers contend on the git index. |
| **official MCP `memory`** | Trivially local JSONL knowledge graph, **no embeddings**, no project scoping — a baseline, not a semantic memory. |

## What we borrow from the survey

- **Single shared store** (from Qdrant/Ogham) — but embedded + on‑demand instead of a resident daemon.
- **Typed memory + sessions + topic_key evolution** (from engram/Recallium/Ogham).
- **No‑LLM regex/cosine write path** (from Ogham) + **LLM only in background** (our addition for 16 GB / 10+ agents).
- **GBNF/guided JSON on a small model** (from the small‑model inference research) — reliability at 4B.

## One‑line conclusion

No existing tool simultaneously satisfies: strictly local + on‑demand (no resident daemon, no Docker) +
no LLM on the write path + typed memory layer + 16 GB + 10+ agents. The gap is exactly a thin layer over
commodity parts — so we build `mnemo`.
