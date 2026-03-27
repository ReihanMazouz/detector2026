#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$ROOT_DIR/.conda-env/bin/python"
BACKEND_PORT="${BACKEND_PORT:-8001}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache}"

mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found at $PYTHON_BIN" >&2
  echo "Create it with: conda env create --prefix ./.conda-env -f environment.yml" >&2
  exit 1
fi

cd "$ROOT_DIR"
"$PYTHON_BIN" -m uvicorn backend.app:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload
