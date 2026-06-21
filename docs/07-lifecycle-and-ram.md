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
| **Consolidation window** | + reranker + NLI (cheap) + the generator (transient) | **+ ~2–3 GB** (then freed) |

Measured on real hardware (single‑user): with the chosen **pplx int8** embedder, **service `S` ≈ 1.2 GB**
(Python runtime + embedded SQLite store + the loaded embedder ~1.1 GB), **connector `c` ≈ 40 MB** (Python +
the MCP SDK, no embedder/store). The store is **brute‑force `sqlite-vec` + FTS5** — no ANN/HNSW index. So 10
agents ≈ `1200 + 40·10 = 1.6 GB`, vs `1200·10 = 12 GB` if each agent ran its own server — the shared
embedder is the decisive saving; the `c·N` tail is cheap.

**The embedder is a pool of `MNEMO_EMBED_WORKERS` independent instances** (default **1**, so the `S` above
is the single‑instance figure). Raising it gives *real* parallel encoding — the background embed workers and
`search` requests each lease their own instance, run with no lock, and return it. The RAM cost: the model
weights are largely shared (the `.onnx` file is mmap'd, so the OS page cache backs every session), but each
instance adds its own session + activations, so `S`'s embedder portion grows roughly with the worker count.
So `MNEMO_EMBED_WORKERS` is the one knob for parallelism **and** RAM — set it to what the machine allows; when
all instances are busy a `search` simply waits for a free one (about one encode). The reranker and generator
are single‑instance pools (recall is single‑threaded).

The number is a guide, not a ceiling: once a **local model for the coding agent itself** runs, it is the main
consumer and the machine is in the GBs regardless — mnemo's job is just to stay the minimum, and we schedule
consolidation for machine‑idle.

## Settings (env / config)
```
MNEMO_PORT=8765
MNEMO_DATA_DIR=~/.mnemo/data
MNEMO_IDLE_GRACE_SECONDS=300        # grace before shutdown after the last connector leaves
MNEMO_IDLE_CHECK_INTERVAL_SECONDS=5 # how often the service sweeps for live connectors
MNEMO_EMBEDDER=pplx                  # default (pplx-embed-v1-0.6b int8); also: fastembed | hash
MNEMO_MODELS_DIR=~/.mnemo/models     # where models are cached (pplx -> ~/.mnemo/models/pplx)
MNEMO_EMBED_MAX_TOKENS=2048          # embedder window cap; over it a memory is rejected (split it)
MNEMO_EMBED_WORKERS=1                # embed worker threads = embedder instance-pool size = max parallel encodes (the RAM knob)
# Recall synthesis (now) + consolidation (Phase 3) generator — multilingual; see 06-models.md, 08-consolidation.md.
MNEMO_RERANKER=off                          # cross-encoder reranker repo, or "off" (default: off — none beat the embedder)
MNEMO_NLI=<model>                           # contradiction NLI (cross-encoder); not yet chosen (Phase 3)
MNEMO_GENERATOR=unsloth/gemma-4-E2B-it-qat-GGUF  # synthesis generator GGUF repo/path, or "off"
MNEMO_GENERATOR_FILE=*UD-Q4_K_XL.gguf       # GGUF glob in the repo (QAT, near-lossless Q4)
MNEMO_GENERATOR_CONTEXT=65536               # generator n_ctx (holds the recall bundle + answer)
MNEMO_GENERATOR_MAX_TOKENS=2048             # synthesis output token cap
MNEMO_MAX_MEMORY_TOKENS=512                 # per-memory content cap (stricter of this and the embedder window)
MNEMO_GENERATOR_ENGINE=llama.cpp            # llama.cpp | ollama
MNEMO_CONSOLIDATE_EVERY=50                  # new records before a consolidation trigger (Phase 3)
```
