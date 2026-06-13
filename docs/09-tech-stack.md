# 09 — Tech Stack

Concrete choices for v1, with the rationale and the alternatives considered.

## Language / runtime: **Python**

- **Why:** the richest "batteries‑included" path for this stack — MCP SDK (FastMCP), ONNX/`fastembed`,
  `lancedb`/`sqlite-vec`, `llama-cpp-python`, model tooling.
- **Alternative:** TypeScript/Node (good MCP SDK, single‑binary‑ish via Bun). Viable, but the local‑ML
  ecosystem (embedders, llama.cpp bindings, GBNF) is smoother in Python. Pick TS only if the team is TS‑first.

## MCP layer: **FastMCP** (official `mcp` Python SDK)

- Exposes the tools from [05-mcp-api.md](05-mcp-api.md).
- Transport: **streamable‑http** for the shared service + a thin **stdio shim** for client compatibility.

## Vector store (embedded): **LanceDB** (primary) or **sqlite‑vec** (minimalist)

| Option | Pros | Cons | Pick when |
|---|---|---|---|
| **LanceDB** ⭐ | embedded (no server/Docker), dense + full‑text/hybrid, fast, handles concurrent reads, columnar, scales well | a heavier dependency than sqlite‑vec | default — want hybrid search and headroom |
| **sqlite‑vec** | a single `.sqlite` file, tiny, trivially embeddable, FTS5 available for lexical | younger vector ext, manual hybrid wiring | want the absolute minimum footprint/deps |

- **Rejected: Qdrant/Chroma in server/Docker mode** — they are excellent concurrent servers, but require an
  always‑on daemon, which contradicts NFR‑5/NFR‑7 (on‑demand, no Docker). With a single shared service process,
  the multi‑process single‑writer problem disappears, so an embedded store is the right call.
- **Rejected: Qdrant embedded (`local path`)** — a simplified client‑side impl with an exclusive file lock; fine
  for one process but feature‑limited and not worth it vs LanceDB.

## Embeddings: **ONNX Runtime** via `fastembed` (or `onnxruntime` directly)

- Local, CPU, no API key. Default model per [06-models.md](06-models.md) (`bge-small-en-v1.5` or `Qwen3-Embedding-0.6B`).
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
lancedb        OR  sqlite-vec + sqlite-utils
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
│   ├── store/           InMemoryMemoryRepository (Phase 0), LanceMemoryRepository (Phase 1+)
│   ├── embedding/       HashEmbedder (offline/tests), FastEmbedEmbedder
│   ├── mcp/             FastMCP controller exposing recall/remember/search
│   └── cli/             Typer controller
└── infrastructure/    # composition root: config, wiring (DI), entrypoints
    ├── config.py
    └── container.py     builds use cases from config (selects adapters)
```

- **Ports live in `application`**; concrete adapters implement them (Dependency Inversion).
- Domain and use cases import **nothing** from `mcp`, `lancedb`, `fastembed`, `llama.cpp`.
- Adding a backend = a new adapter; the core stays untouched (Open/Closed). Adapters are swappable (Liskov).

## What we deliberately do NOT build

- A vector store / ANN index (use LanceDB/sqlite‑vec).
- An embedding model or inference engine (use ONNX / llama.cpp).
- A knowledge graph DB (out of scope; payload + hybrid search covers the need).

## Reuse as a starting skeleton (don't start from zero)

- **`mcp-server-qdrant`** (official) — a clean MCP→store skeleton to learn the FastMCP + tool wiring from.
- **Ogham MCP** — reference for "no‑LLM (regex) write path + typed store tools" patterns.
- We graft our typed memory layer + on‑demand lifecycle + background consolidation on top.
