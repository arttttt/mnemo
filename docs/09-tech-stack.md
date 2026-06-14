# 09 — Tech Stack

Concrete choices for v1, with the rationale and the alternatives considered.

## Language / runtime: **Python**

- **Why:** the richest "batteries‑included" path for this stack — MCP SDK (FastMCP), ONNX/`fastembed`,
  `sqlite-vec`, `llama-cpp-python`, model tooling.
- **Alternative:** TypeScript/Node (good MCP SDK, single‑binary‑ish via Bun). Viable, but the local‑ML
  ecosystem (embedders, llama.cpp bindings, GBNF) is smoother in Python. Pick TS only if the team is TS‑first.

## MCP layer: **FastMCP** (official `mcp` Python SDK)

- Exposes the tools from [05-mcp-api.md](05-mcp-api.md).
- Transport: **streamable‑http** for the shared service + a thin **stdio shim** for client compatibility.

## Store (embedded): **SQLite + `sqlite-vec` + FTS5**

mnemo's core is a **mutable, evolving typed store** (in‑place updates, atomic supersede, point lookups, edges)
that also needs vector search — an **OLTP** shape, not an append‑only vector warehouse. So the engine is a
relational one *with* a vector index, not a vector DB with bolted‑on relational features. One embedded file
under `~/.mnemo/data/`, no server, ~0 idle RAM. **One backend only — we don't mix stores.**

- **Relational core, edges, transactions, point lookups, in‑place updates:** native SQLite.
- **Vector search:** `sqlite-vec` — the embedding is a `BLOB` column on the `memories` row, ranked by the
  `vec_distance_cosine` scalar over a `WHERE`‑filtered scan (brute‑force; **no `vec0` virtual table**, so every
  structured filter — including the list‑valued `tags`/`related_files` via `json_each` — is a plain `WHERE`).
  Well under ~100 ms at our scale (single user; thousands–low‑hundred‑thousands of records); ANN is unnecessary.
  See [the ADR](adr/0001-storage-engine.md#why-a-blob-column--scalar-distance-not-a-vec0-virtual-table) for why
  `BLOB`+scalar over `vec0`.
- **Lexical search:** FTS5 (BM25), built into SQLite (external‑content over `content`, trigger‑synced).
- **Hybrid:** dense + lexical fused by reciprocal‑rank fusion (k=60) in the adapter.

Full rationale and the alternatives weighed (LanceDB, a two‑store split, DuckDB, Postgres+pgvector, libSQL) are
in [adr/0001-storage-engine.md](adr/0001-storage-engine.md). In short: LanceDB (the earlier choice) is an
append‑optimized analytical vector store — strong on vector ANN but structurally awkward for a mutable
relational core (no multi‑row transaction, no joins/edges, costly in‑place updates). **libSQL/Turso** (a SQLite
fork with native DiskANN ANN) is the upgrade path if we ever outgrow brute‑force.

## Embeddings: **ONNX Runtime** via `fastembed` (or `onnxruntime` directly)

- Local, CPU, no API key. The specific model is **not chosen yet** — it is selected against the requirements in [06-models.md](06-models.md).
- One‑time weights download at install; afterwards fully offline.

## Generator inference: **llama.cpp** via `llama-cpp-python`

- On‑demand load/unload, GGUF quants, **GBNF grammar** for guaranteed‑valid JSON.
- **Alternative:** Ollama with `keep_alive=0` (simpler install, auto‑unload).
- **Rejected for the core: vLLM** — built for sustained concurrent serving; our generator is a single background
  batch job, and a resident vLLM violates the on‑demand/RAM goals.

## On‑demand lifecycle: **socket activation** (launchd/systemd) + ref‑counting shim

- See [07-lifecycle-and-ram.md](07-lifecycle-and-ram.md). Default A (socket activation), fallback B (userland supervisor).

## Packaging / install

- Distribute via **`uvx` / `pipx`** (Python) — run without a manual venv.
- `mnemo init <client>` writes the MCP config for Claude Code / Cursor / etc. and sets up socket activation.
- Single data dir `~/.mnemo/` (data + run/ + logs).

## Dependency shortlist (Python)

```
mcp                      # FastMCP server + stdio shim
sqlite-vec               # vector search inside SQLite (FTS5 is built in)
fastembed      OR  onnxruntime + tokenizers
llama-cpp-python         # generator (optional at runtime)
pydantic                 # schemas / config
typer                    # CLI
```

## Code architecture (Clean Architecture layers)

Dependencies point **inward** (domain ← application ← adapters ← infrastructure). The core is framework‑agnostic
(SOLID/DRY/KISS — see [02-requirements.md](02-requirements.md) NFR‑19..22).

```
src/mnemo/
├── domain/            # entities & pure rules: Memory, types, scopes, hashing/normalize. No deps.
├── application/       # use cases + ports (interfaces)
│   ├── ports.py         EmbedderPort, MemoryRepositoryPort
│   └── use_cases.py     RememberMemory, SearchMemory
├── adapters/          # implement ports / handle I/O
│   ├── store/           InMemoryMemoryRepository (offline/tests), SqliteVecMemoryRepository
│   ├── embedding/       HashEmbedder (offline/tests), FastEmbedEmbedder
│   ├── mcp/             FastMCP controller exposing remember/search/delete
│   └── cli/             Typer controller
└── infrastructure/    # composition root: config, wiring (DI), entrypoints
    ├── config.py
    └── container.py     builds use cases from config (selects adapters)
```

- **Ports live in `application`**; concrete adapters implement them (Dependency Inversion).
- Domain and use cases import **nothing** from `mcp`, `sqlite`, `fastembed`, `llama.cpp`.
- Adding a backend = a new adapter; the core stays untouched (Open/Closed). Adapters are swappable (Liskov).

## What we deliberately do NOT build

- A vector store / ANN index (use `sqlite-vec`).
- An embedding model or inference engine (use ONNX / llama.cpp).
- A knowledge graph DB (out of scope — a deterministic typed‑edge table covers the need; we do **not** build an
  inferred knowledge graph; see [adr/0001-storage-engine.md](adr/0001-storage-engine.md)).

## Reuse as a starting skeleton (don't start from zero)

- **`mcp-server-qdrant`** (official) — a clean MCP→store skeleton to learn the FastMCP + tool wiring from.
- **Ogham MCP** — reference for "no‑LLM (regex) write path + typed store tools" patterns.
- We graft our typed memory layer + on‑demand lifecycle + background consolidation on top.
