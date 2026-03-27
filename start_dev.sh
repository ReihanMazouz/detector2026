#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_MODULE="backend.app:app"
PYTHON_BIN="$ROOT_DIR/.conda-env/bin/python"
NPM_BIN="${NPM_BIN:-}"
BACKEND_PORT="${BACKEND_PORT:-8001}"
FRONTEND_PORT="${FRONTEND_PORT:-5174}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache}"

mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found at $PYTHON_BIN" >&2
  echo "Create it with: conda env create --prefix ./.conda-env -f environment.yml" >&2
  exit 1
fi

if [[ -z "$NPM_BIN" ]]; then
  if command -v npm >/dev/null 2>&1; then
    NPM_BIN="$(command -v npm)"
  elif [[ -x /usr/local/bin/npm ]]; then
    NPM_BIN="/usr/local/bin/npm"
  elif [[ -x /opt/homebrew/bin/npm ]]; then
    NPM_BIN="/opt/homebrew/bin/npm"
  fi
fi

if [[ -z "$NPM_BIN" ]]; then
  echo "npm is not available in PATH and no standard npm binary was found" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

cd "$ROOT_DIR"
"$PYTHON_BIN" -m uvicorn "$BACKEND_MODULE" --host 0.0.0.0 --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!

cd "$FRONTEND_DIR"
"$NPM_BIN" run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

echo "Backend running on http://127.0.0.1:$BACKEND_PORT"
echo "Frontend running on http://127.0.0.1:$FRONTEND_PORT"
echo "Press Ctrl+C to stop both processes."

wait "$BACKEND_PID" "$FRONTEND_PID"
