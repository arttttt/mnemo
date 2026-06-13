# 13 — Build, install, use, update

A practical flow for running and testing mnemo. Commands below are verified against the Phase 0 build
(Python 3.14, uv 0.10). `uv` is the primary path; plain `pip`/`venv` works too.

## Quick commands — `dev.sh`

A helper script at the repo root wraps the common flows:

```bash
./dev.sh install      # create .venv + editable install (dev + embed)
./dev.sh test         # run the test suite (pass args: ./dev.sh test -m heavy)
./dev.sh update       # git pull + reinstall deps + run tests
./dev.sh demo         # quick offline CLI demo
./dev.sh mcp          # print the Claude Code MCP add command
./dev.sh clean        # remove .venv, caches, build artifacts (uninstall)
./dev.sh purge-data   # delete the memory data dir (destructive)
./dev.sh help
```

The sections below explain the same steps manually (and the `pip` alternative).

## Prerequisites

- **Python 3.10+** (tested on 3.14).
- **[uv](https://docs.astral.sh/uv/)** (recommended) — or `pip` + `venv`.
- Optional **`embed`** extra for real local semantic search (fastembed downloads a small ONNX model once).
  Without it, use the offline `hash` embedder (lexical only — good for testing).

## 1. Get the code

```bash
git clone https://github.com/arttttt/mnemo.git
cd mnemo
```

## 2. Build & install (dev)

**uv:**
```bash
uv venv                          # create .venv
uv pip install -e ".[dev]"       # core deps + pytest
# optional — real local embeddings:
uv pip install -e ".[dev,embed]"
```

**pip:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ,embed for fastembed
```

Editable install (`-e`) means code changes take effect immediately — no reinstall after a `git pull`.

## 3. Run the tests

```bash
uv run pytest                    # unit + offline integration  -> 16 passed, 1 skipped
uv run pytest tests/unit         # only unit
uv run pytest -m heavy           # real fastembed (downloads the model)
```

Covered: unit (domain, use cases) + integration (store persistence, DI container, CLI, MCP).
Offline and fast by default — the `hash` embedder is used so nothing is downloaded; the real‑embedder
test is marked `heavy` and skipped unless requested.

## 4. Use it — CLI

The default embedder is `fastembed` (needs the `embed` extra). For a quick offline check, set the `hash` embedder.

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
| `MNEMO_STORE_PATH` | `<data>/memory.json` | store file (Phase 0 JSON) |
| `MNEMO_EMBEDDER` | `fastembed` | `fastembed` (real, local) or `hash` (offline) |
| `MNEMO_STORE` | `memory` | `memory` (Phase 0); `lancedb` arrives in Phase 1 |

## 5. Use it — MCP (Claude Code / Cursor)

Point your MCP client at the `mnemo-mcp` entry point. Simplest with uv (no global install — runs in the project venv):

```bash
claude mcp add --scope user mnemo -- uv run --directory /ABS/PATH/to/mnemo mnemo-mcp
```

Or install it as a global tool and reference the bare command:
```bash
uv tool install --from . "mnemo[embed]"          # provides `mnemo` and `mnemo-mcp` on PATH
claude mcp add --scope user mnemo -- mnemo-mcp
```

The agent then has two tools: **`remember`** and **`search`**. For Cursor/Windsurf, put the same
`mnemo-mcp` command into the client's MCP config (stdio transport).

> Phase 0 is launched per session / manually. The on‑demand lifecycle (auto start + grace shutdown)
> is Phase 2 — see [07-lifecycle-and-ram.md](07-lifecycle-and-ram.md).

## 6. Update (later)

```bash
cd mnemo
git checkout main
git pull
uv pip install -e ".[dev,embed]"   # only needed when dependencies changed
uv run pytest                       # sanity check
```

- Code updates apply on `git pull` (editable install) — reinstall only when deps change.
- Global tool install: `uv tool install --force --from . "mnemo[embed]"` after pulling.

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
