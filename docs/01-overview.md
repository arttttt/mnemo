# 01 — Overview

## Problem

AI coding agents forget context between sessions: architecture decisions, the reasoning behind
choices, debugged bugs, agreements, project rules. The developer has to re‑explain the same things.
Existing memory solutions are either:

- tied to the cloud / a SaaS account (no privacy, offline impossible);
- heavy (cognee needs a 32B model and builds a knowledge graph on every write);
- call an LLM on **every** write (mem0) — a bottleneck on a single local inference engine with many agents;
- keep permanently resident processes/daemons (Postgres, Neo4j, Qdrant, vLLM) — RAM "hog" 24/7;
- or they are a "bare" vector store with no memory layer (types, scope, sessions) — you build it all yourself.

A detailed survey is in [11-alternatives-research.md](11-alternatives-research.md).

## Goal

A thin custom memory system tuned exactly to our constraints:

- **Strictly local.** Embeddings and the LLM run only on the developer's machine. Zero outbound cloud calls.
- **On‑demand.** No resident daemons. The service spins up when ≥1 agent is working and shuts down after a
  grace period following the last one. The heavy model is loaded only for the background processing window.
- **Built for 10+ parallel agents.** One shared service process, cheap concurrent writes.
- **16 GB RAM friendly.** ~1 GB while active, ~0 when idle, a brief spike during consolidation.
- **Typed memory + per‑project scoping + session recap** — Recallium‑like UX, but open, local, and simple.

## Non‑goals (explicitly out of scope for v1)

- Cloud sync, multi‑user, RBAC, team sharing.
- A knowledge graph with multi‑hop traversal (Neo4j‑class). Hybrid vector search + payload filters cover ~90% of a coding agent's needs.
- A web dashboard, PDF/document ingestion, analytics. Possible later, not in the core.
- Running a "large" model (32B+). We deliberately work with a small one (≤4–8B).
- LLM on the write hot path. A hard ban (see [03-architecture.md](03-architecture.md)).

## Target user

A solo developer or a small team who:

- runs several agents in parallel (or one agent with many sub‑sessions);
- wants privacy/offline (code and decisions never leave the machine);
- works on a regular machine (16 GB), often without a powerful GPU;
- values that the tool does not run in the background and does not eat resources when unused.

## What it looks like in practice

```
Agent (needs context):  mcp call search(query="how do we handle auth errors", project="checkout-api")
mnemo:                  → spins up if not running
                        → returns: matching memories (rules, decisions, notes), ranked

Agent (after work):     mcp call remember(content="JWT with refresh rotation; httpOnly cookies; ...",
                            type="decision", project="checkout-api",
                            related_files=["src/auth/jwt.ts"])
mnemo:                  → embedding (local, ms) → upsert into store. NO LLM.

Agent (cross-project):  mcp call search("connection pool limits", scope="all")
mnemo:                  → returns relevant hits from every project (soft isolation)

Background (every N writes / on idle):
mnemo:                  → loads Qwen3‑4B → dedup/merge/summary/insights → unloads the model

No active agents for N minutes:
mnemo:                  → persists state, the process exits. RAM ~0.
```
