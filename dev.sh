#!/usr/bin/env bash
# mnemo dev helper — one entry point for the common flows.
# Usage: ./dev.sh <command> [args]
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
  echo "Installed (editable: dev + embed). Try:  ./dev.sh test   |   ./dev.sh demo"
}

cmd_update() {
  require_uv
  git pull --ff-only
  uv pip install -e ".[dev,embed]"
  cmd_test
}

cmd_test() {
  require_uv
  # pass-through args, e.g.:  ./dev.sh test -m heavy   |   ./dev.sh test tests/unit
  uv run pytest "$@"
}

cmd_demo() {
  require_uv
  local tmp
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
}

cmd_mcp() {
  echo "Add mnemo to Claude Code (runs in this checkout's venv):"
  echo
  echo "  claude mcp add --scope user mnemo -- uv run --directory \"$(pwd)\" mnemo-mcp"
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

cmd_help() {
  cat <<EOF
mnemo dev helper

Usage: ./dev.sh <command> [args]

  install       create .venv and install editable (dev + embed)
  update        git pull, reinstall deps, run tests
  test [args]   run pytest  (e.g. ./dev.sh test -m heavy  |  ./dev.sh test tests/unit)
  demo          quick offline CLI demo (hash embedder, temp data dir)
  mcp           print the Claude Code MCP add command for this checkout
  clean         remove .venv, caches and build artifacts (uninstall)
  purge-data    delete the memory data directory ($DATA_DIR)   [destructive]
  help          show this help
EOF
}

case "${1:-help}" in
  install) cmd_install ;;
  update) cmd_update ;;
  test) shift; cmd_test "$@" ;;
  demo) cmd_demo ;;
  mcp) cmd_mcp ;;
  clean) cmd_clean ;;
  purge-data | purge_data) cmd_purge_data ;;
  help | -h | --help) cmd_help ;;
  *) echo "unknown command: $1" >&2; echo; cmd_help; exit 1 ;;
esac
