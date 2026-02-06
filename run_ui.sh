#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/Scripts/python.exe"

if [ ! -x "$PY" ]; then
  echo "ERROR: Expected venv python at: $PY" >&2
  echo "Create the shared venv at the workspace root (.venv) first." >&2
  exit 1
fi

echo "Starting Streamlit UI..."
cd "$ROOT/home-insurance-mcp-client"
exec "$PY" -m streamlit run src/client/app.py
