#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/Scripts/python.exe"

if [ ! -x "$PY" ]; then
  echo "ERROR: Expected venv python at: $PY" >&2
  echo "Create the shared venv at the workspace root (.venv) first." >&2
  exit 1
fi

echo "Starting MCP server..."
cd "$ROOT/home-insurance-mcp/src"
exec "$PY" -m server.main
