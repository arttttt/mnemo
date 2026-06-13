# Phase 2 — On‑demand lifecycle

**Goal:** nothing resident — one shared service spins up under load and exits on idle; ~0 RAM when unused, no Docker.

Each step is **Why** (the requirement and reasoning) · **What** (exactly what to build) · **Done when** · **Depends on**.

---

### 2.1 One shared service + thin per‑agent connector

**Why.** For 10+ agents the wrong model is "each agent spawns its own server" — that means N copies of the embedder
in RAM and N processes contending on one store, the exact problem we set out to avoid. The right model is **one
shared service** all agents talk to, so the embedder and store load once. But most MCP clients launch a small stdio
process, so we front the shared service with a **thin per‑agent connector** that simply proxies into it — the
connector is tiny (≈0 RAM), the heavy parts are shared.

**What.** A single service that owns the store and embedder, plus a thin per‑agent connector (what goes in each
client's config) that forwards calls to the shared service.

**Done when.**
- Several agents reach one service process through the connector.
- Only one embedder instance is loaded regardless of agent count.

**Depends on:** Phase 0 (the service/use cases already exist).

---

### 2.2 Ref‑count + grace shutdown

**Why.** The hard requirement is "nothing resident": the service should exist only while ≥1 agent is working, and
otherwise use ~0 RAM. But it must not thrash — restarting for every brief gap between calls. So it tracks how many
clients are connected and lingers for a short grace period after the last one leaves.

**What.** Count connected clients; when the last disconnects, start a grace timer; if it expires with no new client,
persist state and exit; a client that connects within the window cancels the shutdown.

**Done when.**
- With no clients, the service exits after the configured grace period.
- A connection within the window cancels the pending shutdown.
- Tested.

**Depends on:** 2.1.

---

### 2.3 On‑demand start without a resident daemon

**Why.** We need start‑on‑demand and idle‑exit but explicitly **without a daemon that runs forever** and **without
Docker** — that resident‑process "RAM hog" is precisely what we disliked in other tools. The OS can hold a listening
socket for ~0 RAM and start the service only on the first connection; this is the idiomatic way to get the exact
behavior we want.

**What.** OS‑level socket activation on macOS and Linux: the OS listens; the first connection starts the service;
on idle the service exits; the OS keeps (re‑)listening.

**Done when.**
- The first connection starts the service, idle exits it, the OS re‑listens — verified on macOS and Linux.
- Only the listening socket is resident (not the service).

**Depends on:** 2.2.

---

### 2.4 One‑command setup per client

**Why.** Setup must be one or two commands, not hand‑editing JSON config, or people won't adopt it.

**What.** A command that wires a given client (Claude Code / Cursor) to mnemo and installs the activation.

**Done when.** Running it yields a working connection with no manual editing.

**Depends on:** 2.1, 2.3.

---

### 2.5 RAM budget verification

**Why.** The whole lifecycle exists to fit a 16 GB machine — ~0 idle and ~1 GB active — even next to the developer's
IDE and agents.

**What.** Measure idle and active RAM of the running system.

**Done when.** ~0 RAM idle and ~1 GB active measured on a 16 GB machine.

**Depends on:** 2.1–2.3.

---

**Phase done when:** on‑demand start + idle exit verified on macOS and Linux; nothing resident; no Docker; the RAM
budget is met.
