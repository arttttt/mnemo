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

## Two ways to implement on‑demand

### Option A — socket activation (recommended)
The OS (launchd on macOS / systemd on Linux) holds **only a listening socket** (≈0 RAM).
- First connection → the OS launches `mnemo service`.
- `IdleTimeout` (launchd) / `RuntimeMaxSec`+idle logic (systemd) → the process exits on idle; the OS listens on the socket again.

> This is **NOT** a resident service. Unlike a "service that runs forever" (the very downside you disliked),
> here the process lives only under load. This is the idiomatic way to get exactly the desired scheme.

launchd (plist fragment):
```xml
<key>Sockets</key>   <!-- activating socket -->
<key>TimeOut</key><integer>300</integer>   <!-- idle → exit -->
```
systemd: `mnemo.socket` (Accept=no) + `mnemo.service` with idle‑exit on a timer.

### Option B — userland supervisor (no systemd/launchd)
If you don't want OS units:
- Each agent's shim takes a **lock + ref‑count** in `~/.mnemo/run/` at start.
- The first shim spawns `mnemo service` (background) and writes the PID.
- Each shim decrements the counter on exit; when it reaches 0 — a grace timer is armed.
- Timer elapsed and the counter is still 0 → the service receives a signal and exits.
- Races are resolved with a file lock + an atomic counter.

**Default:** A (more robust, less runtime code). B — a fallback / cross‑platform option.

## MCP transport and shims

- Not all MCP clients can do socket‑activated HTTP. So the agent config points at a **thin stdio shim**
  (`command: mnemo-shim`) that proxies to the shared service (`http://127.0.0.1:<port>/mcp`) and starts it on first use.
- Clients that speak streamable‑http directly can be given the URL and rely on socket activation without the shim.

## RAM budget (target — 16 GB)

| State | What's in memory | RAM |
|---|---|---|
| **Idle** (after grace) | only the socket/shim (or nothing) | **~0** |
| **Active, no generator** | service (runtime) + embedded store + HNSW index (~50–100k records) + embedder | **~0.5–1 GB** |
| **Consolidation window** | + the Qwen3‑4B Q4 generator, transient | **+~3–4 GB** (then freed) |
| **Economy mode** | generator = Qwen3‑1.7B Q4 or disabled | **+~1.2 GB / +0** |

Rough breakdown of the active state:
- service runtime (Python/Node): ~100–250 MB;
- embedded store + index for ~50–100k memories: ~150–400 MB;
- embedder (a small ONNX model, e.g. Qwen3‑Embedding‑0.6B or bge‑m3, Q8 — not chosen yet): ~150–500 MB.

Conclusion: ~1 GB while active, ~0 when idle, a brief on‑demand spike for consolidation.
Comfortable on 16 GB even next to an IDE and agents. If a **local model for the coding agent itself**
also runs on the machine — it is the main consumer; the mnemo part stays cheap, and we schedule
consolidation for machine‑idle.

## Settings (env / config)
```
MNEMO_PORT=8765
MNEMO_DATA_DIR=~/.mnemo/data
MNEMO_IDLE_GRACE_SECONDS=300        # grace before shutdown
MNEMO_EMBEDDER=<tbd>                 # production model not chosen yet — see 06-models.md
MNEMO_GENERATOR=qwen3-4b-instruct-2507-q4   # or "off"
MNEMO_GENERATOR_ENGINE=llama.cpp            # llama.cpp | ollama
MNEMO_CONSOLIDATE_EVERY=50          # new records before a trigger
```
