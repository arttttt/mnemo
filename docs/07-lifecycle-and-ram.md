# 07 — Lifecycle (on‑demand) and RAM budget

The key non‑functional requirement: **nothing runs in the background**. The service exists only
while ≥1 agent is working; after the last one, it shuts down after a grace period. The heavy model is transient.

## Desired behavior

```
first agent starts               → service spins up
agents working (1..N)            → service alive, serving all
last agent disconnects           → a grace timer starts (default 5 min)
   ├─ a new agent arrives in time → timer reset, service stays
   └─ nobody arrives              → service persists state and exits. RAM ~0.
background consolidation         → separately loads the generator and unloads it
```

## How on‑demand works

Socket activation (a standing launchd/systemd unit holding a listening socket) was considered and **dropped**:
even at ≈0 RAM it keeps OS apparatus registered when nothing runs, and it splits across launchd vs systemd. The
start is folded into the connector instead:

- **Start — the connector's job.** Each agent's `mnemo-mcp` connector, on launch, checks whether the service is
  up; if not, it spawns it under a **single‑spawn file lock** in `~/.mnemo/run/` (so a burst of connectors spawns
  exactly one), then polls until it accepts connections. A pidfile records the process; the service is detached
  (`start_new_session`), so it outlives the connector.
- **Exit — the service's job.** Each connector holds an exclusive `flock` on a per‑run marker in
  `~/.mnemo/run/connectors/` for its whole life; the **kernel frees the lock when the connector dies — for any
  reason, clean exit or crash/SIGKILL** — so the service counts live connectors by which markers are still locked
  (no PID tracking, immune to PID reuse). A background sweep checks every few seconds and is the **sole cleaner**
  of dead markers. When none are live it starts a grace timer (default 5 min, `MNEMO_IDLE_GRACE_SECONDS`); a
  connector that appears within it cancels the shutdown, otherwise the service exits (committed data is already
  durable in the SQLite WAL). Boot counts as "the last one just left", so an orphan spawn no connector ever uses
  also exits.

So when no agent runs, **nothing** is registered — not even a socket. While agents run, each carries a thin
connector **in its own process tree** (not a daemon — it dies with the agent's MCP client); the one shared service
stays up until the last connector leaves.

## MCP transport and connectors

- The agent config points at the **thin stdio connector** `mnemo-mcp`, which proxies to the shared service
  (`http://127.0.0.1:<port>/mcp`) and starts it on first use. The agent‑facing command is unchanged; only its
  internals became a proxy.
- A client that speaks streamable‑http directly can be given the URL instead — but then nothing starts the
  service for it, so it must already be running.

## RAM — minimal necessary, not a budget

The goal is to add the **minimum**, not to hit a number. The footprint is `S + c·N` for N connected agents —
the heavy parts load **once** in the shared service, with thin connectors — never `S·N`.

| State | What's in memory | RAM |
|---|---|---|
| **No agent connected** (after grace) | nothing — the service has exited | **~0** |
| **N agents, no generator** | one shared service `S` + one thin connector `c` per agent | **`S + c·N`** |
| **Consolidation window** | + the generator, transient | **+ GBs** (then freed) |

Measured on real hardware (single‑user, real bge‑small embedder): **service `S` ≈ 170 MB** (Python runtime +
embedded SQLite store + the loaded embedder), **connector `c` ≈ 40 MB** (Python + the MCP SDK, no embedder/store).
The store is **brute‑force `sqlite-vec` + FTS5** — no ANN/HNSW index. So 10 agents ≈ `170 + 40·10 = 570 MB`, vs
`170·10 = 1.7 GB` if each agent ran its own server — the shared embedder is the decisive saving; the `c·N` tail
is cheap.

The number is a guide, not a ceiling: once a **local model for the coding agent itself** runs, it is the main
consumer and the machine is in the GBs regardless — mnemo's job is just to stay the minimum, and we schedule
consolidation for machine‑idle.

## Settings (env / config)
```
MNEMO_PORT=8765
MNEMO_DATA_DIR=~/.mnemo/data
MNEMO_IDLE_GRACE_SECONDS=300        # grace before shutdown after the last connector leaves
MNEMO_IDLE_CHECK_INTERVAL_SECONDS=5 # how often the service sweeps for live connectors
MNEMO_EMBEDDER=<tbd>                 # production model not chosen yet — see 06-models.md
MNEMO_GENERATOR=qwen3-4b-instruct-2507-q4   # or "off"
MNEMO_GENERATOR_ENGINE=llama.cpp            # llama.cpp | ollama
MNEMO_CONSOLIDATE_EVERY=50          # new records before a trigger
```
