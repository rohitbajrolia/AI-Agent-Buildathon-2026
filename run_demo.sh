#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/Scripts/python.exe"

if [ ! -x "$PY" ]; then
  echo "ERROR: Expected venv python at: $PY" >&2
  echo "Create the shared venv at the workspace root (.venv) first." >&2
  exit 1
fi

SERVER_URL="${MCP_SERVER_URL:-http://127.0.0.1:4200/mcp/}"
HEALTH_URL="${SERVER_URL%/}/health"

# Used for fast port checks on Windows.
SERVER_PORT="${MCP_PORT:-}"
if [ -z "$SERVER_PORT" ]; then
  SERVER_PORT="$(echo "$SERVER_URL" | sed -n 's#^[a-zA-Z]\+://[^:/]\+:\([0-9]\+\)/.*#\1#p')"
fi
SERVER_PORT="${SERVER_PORT:-4200}"

QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
QDRANT_HEALTH_URL="${QDRANT_URL%/}/healthz"

check_qdrant() {
  # Preflight: require Qdrant before starting server/UI.
  if ! curl -fsS --max-time 2 "$QDRANT_HEALTH_URL" >/dev/null 2>&1; then
    echo "ERROR: Qdrant is not reachable at $QDRANT_URL" >&2
    echo "Start Qdrant first (Docker Desktop), then re-run ./run_demo.sh" >&2
    echo "Example:" >&2
    echo "  docker run --rm -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.9.3" >&2
    exit 1
  fi
}

check_mcp_health() {
  # Health check via MCP tool call; a plain HTTP GET may return 406.
  "$PY" - <<'PY'
import asyncio
import json
import os
import sys

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:4200/mcp/")
if not url.endswith("/"):
  url += "/"

async def go() -> bool:
  async with streamable_http_client(url) as (read_stream, write_stream, _):
    async with ClientSession(read_stream, write_stream) as session:
      await session.initialize()
      result = await session.call_tool("health", {})
      for part in result.content:
        if isinstance(part, types.TextContent):
          payload = json.loads(part.text)
          return payload.get("status") == "ok"
  return False

try:
  ok = asyncio.run(asyncio.wait_for(go(), timeout=2.5))
except Exception:
  ok = False

raise SystemExit(0 if ok else 1)
PY
}

port_is_listening() {
  # netstat.exe is available on Windows; Git Bash can run it.
  netstat.exe -ano 2>/dev/null | grep -E ":${SERVER_PORT} " | grep -i LISTENING >/dev/null 2>&1
}

cleanup() {
  if [ "${STARTED_SERVER:-0}" = "1" ]; then
    pid="$(netstat.exe -ano 2>/dev/null | grep -E ":${SERVER_PORT} " | grep -i LISTENING | awk '{print $NF}' | head -n 1)"
    if [ -n "$pid" ]; then
      echo "Stopping MCP server on port $SERVER_PORT (pid $pid)..."
      taskkill.exe /PID "$pid" /T /F >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT

STARTED_SERVER=0

echo "Checking Qdrant: $QDRANT_HEALTH_URL"
check_qdrant

echo "Checking MCP server health (MCP tool call) via $SERVER_URL"
if check_mcp_health; then
  echo "MCP server is already running."
  SERVER_PID=""
else
  if port_is_listening; then
    echo "ERROR: Port $SERVER_PORT is already in use, but the MCP health check did not respond." >&2
    echo "Stop the process using port $SERVER_PORT (or set MCP_PORT and MCP_SERVER_URL to a different port), then re-run." >&2
    echo "Tip (PowerShell): netstat -ano | findstr :$SERVER_PORT" >&2
    exit 1
  fi

  echo "Starting MCP server in background..."
  (
    cd "$ROOT/home-insurance-mcp/src"
    exec "$PY" -m server.main
  ) &
  SERVER_PID=$!
  STARTED_SERVER=1

  echo "Waiting for MCP server to become ready..."
  for i in {1..30}; do
    if check_mcp_health; then
      echo "MCP server is up."
      break
    fi

    # If the process exits early, stop and surface a likely cause.
    if [ -n "${SERVER_PID:-}" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "ERROR: MCP server process exited during startup." >&2
      echo "Most common cause: port $SERVER_PORT is already in use." >&2
      echo "Tip (PowerShell): netstat -ano | findstr :$SERVER_PORT" >&2
      exit 1
    fi

    sleep 1
    if [ "$i" -eq 30 ]; then
      echo "ERROR: MCP server did not become healthy in time." >&2
      echo "Tip: check that Qdrant is running and port $SERVER_PORT is free." >&2
      exit 1
    fi
  done
fi

echo "Starting Streamlit UI (Ctrl+C to stop both)..."
cd "$ROOT/home-insurance-mcp-client"
exec "$PY" -m streamlit run src/client/app.py
