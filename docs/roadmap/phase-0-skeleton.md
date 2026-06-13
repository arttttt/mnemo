# Phase 0 — Walking skeleton ✅ (done, merged)

**Goal:** the smallest end‑to‑end slice — an agent can store and search memory locally via MCP.

Step format: **Why** (the requirement) · **What** (exactly what to do) · **Done when** (verifiable).

### 0.1 Clean‑Architecture scaffold ✅
**Why:** the codebase must stay framework‑agnostic and swappable (NFR‑19..22).
**What:** four layers (domain / application / adapters / infrastructure), `uv`, CLI entry point.
**Done when:** domain & application import no framework; the package builds and runs.

### 0.2 Local store + embedders ✅
**Why:** the core must run and be tested offline, with no cloud (NFR‑1/2).
**What:** in‑memory + JSON store behind `MemoryRepositoryPort`; `hash` (offline) and `fastembed` embedders.
**Done when:** store round‑trips; search ranks by similarity; tests pass offline.

### 0.3 MCP `remember` / `search` ✅
**Why:** agents must reach memory through MCP (FR‑16).
**What:** the two write/read tools with typed (enum) parameters.
**Done when:** an agent can `remember` then `search`; the tool schema advertises valid types.

### 0.4 CLI `store` / `search` / `stats` ✅
**Why:** humans need a way to drive and debug memory (FR‑17).
**What:** CLI over the same use cases.
**Done when:** the commands work end‑to‑end offline.

### 0.5 Tests (unit + integration) ✅
**Why:** testing is a first‑class requirement (NFR‑23..25).
**What:** unit (domain, use cases) + integration (store, CLI, MCP); heavy embedder behind a marker.
**Done when:** `uv run pytest` green and offline by default.

**Phase 0 done:** ✅ shipped and merged.
