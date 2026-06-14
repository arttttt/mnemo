#!/usr/bin/env bash
# mnemo dev helper. Run it from the repo:  ./dev.sh  -> shows a menu.
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
DATA_DIR="${MNEMO_DATA_DIR:-$HOME/.mnemo}"

require_uv() {
  command -v uv >/dev/null 2>&1 || {
    echo "error: 'uv' not found — install it: https://docs.astral.sh/uv/" >&2
    exit 1
  }
}

cmd_install() {
  require_uv
  uv venv
  uv pip install -e ".[dev,embed]"
  echo
  echo "Installed (editable: dev + embed)."
}

cmd_update() {
  # Re-sync the environment to the current checkout (you manage git/branches yourself).
  require_uv
  uv pip install -e ".[dev,embed]"
}

cmd_test() {
  require_uv
  uv run pytest "$@"
}

cmd_test_heavy() {
  require_uv
  uv run pytest -m heavy
}

cmd_demo() {
  require_uv
  # subshell so the demo env does not leak into the menu session
  (
    tmp="$(mktemp -d)"
    export MNEMO_EMBEDDER=hash MNEMO_DATA_DIR="$tmp"
    echo "--- store ---"
    uv run mnemo store "Use JWT with refresh rotation; httpOnly cookies" --type decision --project demo
    uv run mnemo store "Always confirm destructive DB operations" --type rule --scope global
    echo "--- search (project + global) ---"
    uv run mnemo search "destructive operations" --project demo
    echo "--- stats ---"
    uv run mnemo stats
    rm -rf "$tmp"
  )
}

cmd_mcp() {
  echo "Add mnemo to Claude Code (runs in this checkout's venv):"
  echo
  echo "  claude mcp add --scope user mnemo -- uv run --directory \"$(pwd)\" mnemo-mcp"
}

cmd_stop() {
  # Stop the on-demand service. Temporary until idle-exit lands; the connector
  # spawns the service, which has no self-shutdown yet, so it lingers.
  pid_file="$DATA_DIR/run/service.pid"
  if [ -f "$pid_file" ] && kill "$(cat "$pid_file")" 2>/dev/null; then
    echo "Stopped mnemo-service (pid $(cat "$pid_file"))."
    rm -f "$pid_file"
  else
    echo "No running mnemo-service."
  fi
}

cmd_clean() {
  rm -rf "$VENV" build dist .pytest_cache ./*.egg-info src/*.egg-info
  find . -type d -name __pycache__ -prune -exec rm -rf {} +
  echo "Removed venv, caches and build artifacts."
}

cmd_purge_data() {
  read -r -p "Delete ALL memory data in '$DATA_DIR' ? [y/N] " ans
  case "$ans" in
    y | Y) rm -rf "$DATA_DIR" && echo "Deleted $DATA_DIR" ;;
    *) echo "Cancelled." ;;
  esac
}

print_menu() {
  cat <<EOF

  mnemo dev helper
  ----------------
  1) install        create .venv + editable install (dev + embed)
  2) update         reinstall deps (no git, no tests)
  3) test           run the test suite (offline)
  4) test (heavy)   real-embedder tests (downloads model)
  5) demo           quick offline CLI demo
  6) mcp            print the Claude Code MCP add command
  7) stop           stop the running mnemo-service (until idle-exit lands)
  8) clean          remove .venv, caches, build artifacts
  9) purge-data     delete memory data ($DATA_DIR)  [destructive]
  0) quit
EOF
}

dispatch() {
  case "$1" in
    1 | install) cmd_install ;;
    2 | update) cmd_update ;;
    3 | test) shift || true; cmd_test "$@" ;;
    4 | test-heavy) cmd_test_heavy ;;
    5 | demo) cmd_demo ;;
    6 | mcp) cmd_mcp ;;
    7 | stop) cmd_stop ;;
    8 | clean) cmd_clean ;;
    9 | purge-data | purge_data) cmd_purge_data ;;
    *) echo "invalid choice: $1" >&2; return 1 ;;
  esac
}

# Escape hatch for scripting/CI:  ./dev.sh test -m heavy
if [ "$#" -gt 0 ]; then
  dispatch "$@"
  exit $?
fi

# Default: interactive menu
while true; do
  print_menu
  read -r -p "> " choice || break
  case "$choice" in
    0 | q | quit | exit) break ;;
    "") continue ;;
  esac
  set +e
  dispatch "$choice"
  set -e
  echo
done
