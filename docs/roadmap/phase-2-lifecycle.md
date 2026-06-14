# Phase 2 — On‑demand lifecycle

**Goal:** nothing resident — one shared service spins up under load and exits on idle; ~0 RAM when **no agent is
connected**, no Docker.

Each step is **Why** (the requirement and reasoning) · **What** (exactly what to build) · **Done when** · **Depends on**.

> **Status (v0.2.0 — complete).** All steps built. 2.1: one shared `mnemo-service` over streamable-http owning a
> single embedder + thread-safe store; a thin `mnemo-mcp` connector that proxies into it, **spawns it on demand**
> (no socket activation — see 2.3), and **owns the run's session id** (sent to the service as request metadata, so
> the service never invents one). The embedder is shared, so the footprint is `S + c·N`, not `S·N` — validated on
> real hardware (~170 MB service + ~40 MB per connector). 2.2: the service **idle-exits** when no connector is
> alive. 2.4: `mnemo setup` wires six MCP clients (one command, no hand-editing). 2.5: the `S + c·N` footprint is
> confirmed by live observation. The 10+-agent concurrency model is taken as covered by the store-concurrency test
> + a live multi-agent run.

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

### 2.2 Ref‑count + grace shutdown — **built**

**Why.** The hard requirement is "nothing resident": the service should exist only while ≥1 agent is working, and
otherwise use ~0 RAM. But it must not thrash — restarting for every brief gap between calls. So it tracks how many
clients are connected and lingers for a short grace period after the last one leaves.

**What (built).** Liveness is keyed to the connector, not to traffic or a transport session (a busy agent can be
silent between calls, and a crashed connector never sends a clean close). Each connector holds an exclusive
`flock` on `~/.mnemo/run/connectors/<session_id>.lock` for its whole life; the **kernel frees it on death (clean
or SIGKILL)**, so the service counts live connectors by which markers are still locked — no PID tracking, immune
to PID reuse. A background sweep (a daemon thread, every `MNEMO_IDLE_CHECK_INTERVAL_SECONDS`) checks them and is
the **sole cleaner** of dead markers (it removes a marker exactly when it can take the lock). When none are live a
grace timer starts (`MNEMO_IDLE_GRACE_SECONDS`, default 5 min); a connector appearing within it cancels the
shutdown, otherwise the service exits (committed data is durable in the SQLite WAL). Boot counts as "the last one
just left", so an orphan spawn also exits.

**Done when.** ✅
- With no clients, the service exits after the configured grace period (incl. a crashed/SIGKILLed connector).
- A connection within the window cancels the pending shutdown.
- Tested at the real boundary (`tests/integration/test_idle_exit.py`: orphan exit, live-keeps-alive then
  exit-on-release, and SIGKILL crash recovery).

**Depends on:** 2.1.

---

### 2.3 On‑demand start — DROPPED (folded into the connector)

Socket activation is **dropped**. It keeps a standing OS unit + listening socket registered even when nothing
runs — apparatus we don't want, and "idiomatic" is not our criterion. Instead the **connector starts the service
itself** when it is not up (a single‑spawn file lock so a burst of connectors spawns exactly one; it then polls
until ready) — built as part of 2.1. So when no agent runs, **nothing** is registered, not even a socket, and the
launchd‑vs‑systemd portability cost disappears. Idle‑**exit** stays the service's own job (2.2).

---

### 2.4 One‑command setup per client — **built**

**Why.** Setup must be one or two commands, not hand‑editing JSON config, or people won't adopt it.

**What (built).** `mnemo setup` resolves the connector launch command (absolute `mnemo-mcp`, or
`uv run --directory <repo> mnemo-mcp` from a checkout — absolute so GUI clients' PATH resolves it) and
wires a client to it. Six clients, two integration styles behind one `ClientInstaller` port: those that
ship an official `mcp add` (**claude-code**, **codex**, **kimi-code**) are shelled out to; the rest
(**cursor**, **windsurf**, **opencode**) have their MCP config written directly (idempotent upsert,
other entries preserved). `mnemo setup` with no client **detects** installed clients, lists them, and
wires the picked subset; `mnemo setup <client>` wires one; `--all` wires all detected; `--dry-run`
writes nothing. No socket/launchd activation to install (the connector spawns the service on demand).

**Done when.** ✅ Running it yields a working connection with no manual editing. Covered by unit tests
(selection parsing, command resolution) and real-boundary integration (file writes per client, the CLI
installer's exact argv via a fake runner, and the `setup` command through the real CLI app).

**Depends on:** 2.1, 2.3.

---

### 2.5 Footprint check — minimal necessary — **verified (observed)**

**Why.** The lifecycle exists to add the **minimum** to the machine, not to hit a fixed budget (once local LLMs
run they dominate RAM anyway). The thing to verify is the *shape*, not a number.

**What.** Confirm the footprint is `S + c·N` — one shared embedder regardless of agent count, thin connectors —
that the shared service leaves ~0 resident when no agent is connected, and that the generator is transient.

**Done when.** ✅ The embedder is loaded once for N agents (not N times); the service is gone when no agent is
connected; a connector alone is tens of MB. This is a **passive observation**, not a CI assertion: the RAM gap
only appears with the real embedder (offline/CI uses the hash embedder, where the service and a connector are
near-equal). Observed live: one ~170 MB service + ~40 MB per connector. (The generator is transient by design —
verified once it lands in Phase 3.)

**Depends on:** 2.1, 2.2.

---

**Phase done.** ✅ The connector starts the service on demand; the service idle‑exits after grace; nothing resident
when no agent is connected; no Docker; the footprint is minimal (`S + c·N`, one shared embedder). The 10+‑agent
concurrency model is taken as covered by the store-concurrency test (8 concurrent writers + readers, zero lost
writes) plus a live multi-agent run sharing one service — no dedicated ≥10-connector stress was written.
