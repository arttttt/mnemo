# 13 — Build, install, use, update

A practical flow for running and testing mnemo. Commands below are verified against the Phase 0 build
(Python 3.14, uv 0.10). `uv` is the primary path; plain `pip`/`venv` works too.

## Quick commands — `dev.sh`

Run the helper from the repo and pick from the menu:

```bash
./dev.sh
```
```
  1) install        create .venv + editable install (dev + embed)
  2) update         git pull + reinstall + run tests
  3) test           run the test suite (offline)
  4) test (heavy)   real-embedder tests (downloads model)
  5) demo           quick offline CLI demo
  6) mcp            print the Claude Code MCP add command
  7) clean          remove .venv, caches, build artifacts
  8) purge-data     delete the memory data dir (destructive)
  0) quit
```

(For scripting/CI you can also pass a choice directly, e.g. `./dev.sh test -m heavy`.)
The sections below explain the same steps manually (and the `pip` alternative).

## Prerequisites

- **Python 3.10+** (tested on 3.14). Note: a fresh macOS ships `/usr/bin/python3` at **3.9**, which is too
  old — a plain `pip install -e .` against the system interpreter fails before mnemo runs.
- **[uv](https://docs.astral.sh/uv/)** — effectively required, not just convenient: it provisions a
  new‑enough interpreter automatically. With plain `pip`/`venv` you must first install Python 3.10+ yourself.
- The default install includes the runtime for the pplx embedder and recall generator. Model weights are
  fetched once into `~/.mnemo/models`, then loaded cache-first without a network round-trip.

## 1. Get the code

```bash
git clone https://github.com/arttttt/mnemo.git
cd mnemo
```

## 2. Build & install (dev)

**uv:**
```bash
uv venv                          # create .venv
uv pip install -e ".[dev]"       # default pplx + recall runtimes, plus pytest
```

**pip:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Editable install (`-e`) means code changes take effect immediately — no reinstall after a `git pull`.

## 3. Run the tests

```bash
uv run pytest                    # unit + offline integration (in‑memory + SQLite backends)
uv run pytest tests/unit         # only unit
uv run pytest -m heavy           # real model integration tests (downloads weights)
```

Covered: unit (domain, use cases, rank fusion) + integration (store contract on in‑memory and SQLite,
DI container, CLI, MCP, migration). Offline and fast by default — the `hash` embedder is used so nothing is
downloaded, and the SQLite backend runs offline (`sqlite-vec` is a small extension). The real‑embedder and the
legacy LanceDB tests are marked `heavy` and skipped unless requested.

## 4. Use it — CLI

The default embedder is `pplx`; its runtime is part of the normal install. For a quick model-free check,
set the `hash` embedder.

```bash
export MNEMO_EMBEDDER=hash        # offline; drop this once you installed .[embed]

uv run mnemo store "Use JWT with refresh rotation; httpOnly cookies" --type decision --project checkout-api
uv run mnemo store "Always confirm destructive DB operations" --type rule --scope global
uv run mnemo search "how is auth done" --project checkout-api     # current project + global
uv run mnemo search "destructive operations" --scope all          # cross-project
uv run mnemo stats
```

(If the venv is activated, drop the `uv run` prefix: `mnemo store ...`.)

All state lives in **one directory** — back up / move / wipe by copying or deleting it.

### Config (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MNEMO_DATA_DIR` | `~/.mnemo/data` | data directory |
| `MNEMO_SQLITE_PATH` | `<data>/memory.db` | SQLite store file (SQLite + `sqlite-vec` + FTS5 — the store) |
| `MNEMO_EMBEDDER` | `pplx` | `pplx` (default, pplx-embed-v1-0.6b int8) · `hash` (model-free tests) |
| `MNEMO_MODELS_DIR` | `~/.mnemo/models` | model cache (pplx → `~/.mnemo/models/pplx`) |
| `MNEMO_EMBED_MAX_TOKENS` | `2048` | embedder window cap; over it a memory is rejected (split it) |

## 5. Use it — MCP (one-command setup)

`mnemo setup` wires a client to the connector for you — no hand-editing JSON. It resolves the right
launch command (an absolute `mnemo-mcp`, or `uv run --directory <repo> mnemo-mcp` from a checkout) and
either runs the client's own `mcp add` or writes its config file.

```bash
mnemo setup                 # detect installed clients, list them, pick which to wire
mnemo setup cursor          # wire one explicitly (no prompt)
mnemo setup --all           # wire every detected client
mnemo setup --dry-run       # show what it would do, write nothing
```

Supported clients: **claude-code**, **codex**, **kimi-code** (via each one's official `mcp add`), and
**cursor**, **windsurf**, **opencode** (by writing their MCP config). The agent then has the mnemo
memory tools — **`remember`**, **`search`**, **`browse`**, **`get`**, **`recall`**, **`delete`**, and the project
tools (`create-project` / `update-project` / `list-projects` / `delete-project`). Wiping everything
(`purge`) is intentionally **CLI‑only**.

Prefer to wire it by hand? Point the client's MCP config (stdio transport) at `mnemo-mcp`, e.g. for
Claude Code:
```bash
claude mcp add --scope user mnemo -- mnemo-mcp
# or, from a checkout without a global install:
claude mcp add --scope user mnemo -- uv run --directory /ABS/PATH/to/mnemo mnemo-mcp
```

> The on‑demand lifecycle is live: the `mnemo-mcp` connector **auto‑starts** the shared service on first use and
> the service **idle‑exits** after a grace period once no connector is alive — see
> [07-lifecycle-and-ram.md](07-lifecycle-and-ram.md). No manual start/stop needed (`./dev.sh stop` remains as a
> manual override).

## 6. Update (later)

```bash
cd mnemo
git checkout main
git pull
uv pip install -e ".[dev]"         # only needed when dependencies changed
uv run pytest                       # sanity check
```

- Code updates apply on `git pull` (editable install) — reinstall only when deps change.
- Global tool install: `uv tool install --force --from . mnemo` after pulling.

## 7. Contributing flow (gitflow: feature → main)

```bash
git checkout main && git pull
git checkout -b feature/<short-name>
# ... code + unit/integration tests ...
uv run pytest
git push -u origin feature/<short-name>
# open a PR into main; merge when green
```

No `develop` branch — feature branches target `main` directly.
