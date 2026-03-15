#!/usr/bin/env bash
# Runs once when the Codespace is first created.
# Installs Python packages and downloads the Qdrant binary.
set -euo pipefail

WORKSPACE="/workspaces/AI-Agent-Buildathon-2026"

echo ""
echo "=== Installing Python packages ==="
pip install --upgrade pip --quiet
pip install -e "$WORKSPACE/home-insurance-mcp" --quiet
pip install -e "$WORKSPACE/home-insurance-mcp-client" --quiet
echo "Python packages installed."

echo ""
echo "=== Downloading Qdrant binary ==="
QDRANT_VERSION="v1.9.3"
QDRANT_DIR="$HOME/.qdrant"
mkdir -p "$QDRANT_DIR"

curl -fsSL \
  "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-musl.tar.gz" \
  -o /tmp/qdrant.tar.gz

tar -xzf /tmp/qdrant.tar.gz -C "$QDRANT_DIR"
chmod +x "$QDRANT_DIR/qdrant"
rm /tmp/qdrant.tar.gz
echo "Qdrant binary ready at $QDRANT_DIR/qdrant"

echo ""
echo "=== Creating docs folder ==="
mkdir -p "$WORKSPACE/docs"
echo "Drop your policy PDFs into the docs/ folder, then run Index to Qdrant from the sidebar."

echo ""
echo "=== Setup complete ==="
echo "Run:  bash .devcontainer/start.sh"
echo "Then open the Streamlit UI from the Ports tab."
