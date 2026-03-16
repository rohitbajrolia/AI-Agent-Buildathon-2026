#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

MCP_PORT="${MCP_PORT:-4200}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

kill_port() {
  local port="$1"
  local label="$2"

  # netstat.exe output is Windows-style; strip CR to keep parsing stable.
  local pids
  pids="$(netstat.exe -ano 2>/dev/null | tr -d '\r' | grep -E ":${port} " | grep -i LISTENING | awk '{print $NF}' | sort -u)"

  if [ -z "$pids" ]; then
    echo "No process is listening on port ${port} (${label})."
    return 0
  fi

  echo "Stopping ${label} on port ${port}..."
  while read -r pid; do
    [ -z "$pid" ] && continue
    echo "- Killing PID ${pid}"
    taskkill.exe /PID "$pid" /T /F >/dev/null 2>&1 || true
  done <<<"$pids"
}

echo "Stopping processes..."
kill_port "$STREAMLIT_PORT" "Streamlit UI"
kill_port "$MCP_PORT" "MCP server"

echo "Done."
echo "Note: Qdrant is a Docker container; stop it from Docker Desktop, or run:"
echo "  docker ps --filter ancestor=qdrant/qdrant"
echo "  docker stop <container_id>"
