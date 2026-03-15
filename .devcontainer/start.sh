#!/usr/bin/env bash
# Run this inside the Codespace terminal to start all three services.
# Usage: bash .devcontainer/start.sh
set -euo pipefail

WORKSPACE="/workspaces/AI-Agent-Buildathon-2026"
QDRANT_BIN="$HOME/.qdrant/qdrant"
QDRANT_STORAGE="$HOME/.qdrant/storage"

mkdir -p "$QDRANT_STORAGE"

# ── 1. Start Qdrant ──────────────────────────────────────────────────────────
echo ""
echo "=== Starting Qdrant ==="
QDRANT_URL=http://localhost:6333

if curl -fsS --max-time 2 "$QDRANT_URL/healthz" >/dev/null 2>&1; then
  echo "Qdrant is already running."
else
  nohup "$QDRANT_BIN" \
    --storage-path "$QDRANT_STORAGE" \
    > /tmp/qdrant.log 2>&1 &
  echo "Waiting for Qdrant to start..."
  for i in {1..20}; do
    if curl -fsS --max-time 1 "$QDRANT_URL/healthz" >/dev/null 2>&1; then
      echo "Qdrant is up."
      break
    fi
    sleep 1
    if [ "$i" -eq 20 ]; then
      echo "ERROR: Qdrant did not start in time. Check /tmp/qdrant.log"
      exit 1
    fi
  done
fi

# ── 2. Start MCP server ───────────────────────────────────────────────────────
echo ""
echo "=== Starting MCP server ==="
MCP_SERVER_URL="${MCP_SERVER_URL:-http://localhost:4200/mcp/}"

check_mcp() {
  python - <<'PY'
import asyncio, json, os, sys
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
url = os.getenv("MCP_SERVER_URL", "http://localhost:4200/mcp/")
async def go():
  async with streamable_http_client(url) as (r, w, _):
    async with ClientSession(r, w) as s:
      await s.initialize()
      result = await s.call_tool("health", {})
      for p in result.content:
        if isinstance(p, types.TextContent):
          return json.loads(p.text).get("status") == "ok"
  return False
try:
  ok = asyncio.run(asyncio.wait_for(go(), timeout=2.5))
except Exception:
  ok = False
raise SystemExit(0 if ok else 1)
PY
}

if check_mcp; then
  echo "MCP server is already running."
else
  nohup python -m server.main \
    --app-dir "$WORKSPACE/home-insurance-mcp/src" \
    > /tmp/mcp_server.log 2>&1 &
  # python -m server.main does not support --app-dir; use working directory instead
  kill $! 2>/dev/null || true

  nohup bash -c "cd '$WORKSPACE/home-insurance-mcp/src' && python -m server.main" \
    > /tmp/mcp_server.log 2>&1 &

  echo "Waiting for MCP server to start..."
  for i in {1..30}; do
    if check_mcp; then
      echo "MCP server is up."
      break
    fi
    sleep 1
    if [ "$i" -eq 30 ]; then
      echo "ERROR: MCP server did not start in time. Check /tmp/mcp_server.log"
      cat /tmp/mcp_server.log
      exit 1
    fi
  done
fi

# ── 3. Start Streamlit ────────────────────────────────────────────────────────
echo ""
echo "=== Starting Streamlit UI ==="
echo "Open the Ports tab in VS Code and click the globe icon next to port 8501."
echo ""

cd "$WORKSPACE/home-insurance-mcp-client"
exec python -m streamlit run src/client/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
