
# Home Insurance MCP (Server)

This is the MCP server. It exposes a few tools over Streamable HTTP:

- `health`: quick connectivity check
- `ingest_folder`: scan a local docs folder and summarize what’s inside
- `index_folder_qdrant`: chunk + embed + store the docs in Qdrant
- `retrieve_clauses`: semantic search over indexed clauses
- `index_status`: report Qdrant/collection status for demo reliability

Goal: a clean Buildathon demo that stays grounded in the actual policy packet and can explain itself.

## Architecture

See the end-to-end architecture and request flows in [../ARCHITECTURE.md](../ARCHITECTURE.md).

## Security + PII

- Don’t commit real keys. Use `.env.example` as the template and keep `.env` local.
- Put local policy PDFs under the workspace `docs/` folder — it’s git-ignored on purpose (no accidental “oops I pushed the policy packet”).
- Keep the demo grounded in the policy packet; don’t answer beyond retrieved evidence.

## Guardrails

- **Docs scope**: ingest/index is restricted to the configured docs root.
- **PII safety**: error text is scrubbed of secrets before returning to the client.
- **Key validation**: `index_status` can optionally verify the OpenAI key to prevent demo surprises.

## Prereqs

- Python 3.12+
- A running Qdrant instance
- An OpenAI API key (needed for embeddings/indexing and retrieval)
- Optional OCR for scanned PDFs: Tesseract installed and available on PATH

### Start Qdrant (Docker)

If you don’t already have Qdrant running locally:

```bash
docker run --rm -p 6333:6333 qdrant/qdrant
```

## Configure

1) Copy the example env file:

```bash
copy .env.example .env
```

2) Fill in values in `.env`:

- `OPENAI_API_KEY`
- `QDRANT_URL` (default is fine for local)
- `QDRANT_COLLECTION` (default is fine)
- `MCP_PORT` (default 4200)
- `MCP_HOST` (default `127.0.0.1`; use `0.0.0.0` if you need LAN access)
- `CHECK_OPENAI_ON_INDEX_STATUS` (default 1)
- `OPENAI_INDEX_STATUS_CACHE_SECONDS` (default 60)
- `ENABLE_PDF_OCR_FALLBACK` (default 1)

## Install + run

Use whatever workflow you prefer (uv/pip/poetry). For this workspace layout, the most predictable approach is a single shared virtual environment at the workspace root.

From the workspace root (`C:\AI-Agent-Buildathon-2026`):

```bash
./.venv/Scripts/python.exe -m pip install -e ./home-insurance-mcp
```

Start the server (Git Bash):

```bash
cd /c/AI-Agent-Buildathon-2026/home-insurance-mcp/src
../../.venv/Scripts/python.exe -m server.main
```

Or use the workspace helper script (starts server only):

```bash
cd /c/AI-Agent-Buildathon-2026
bash ./run_server.sh
```

Server starts at:

- `http://127.0.0.1:4200/mcp/`

If you set `MCP_HOST=0.0.0.0`, the server is reachable on your LAN.

## Quick sanity checks

Once the server is running, the client UI can call these tools, but you can also sanity-check via the client smoke script.

What you should see when things are healthy:

- `health` returns `status: ok`
- `index_status` shows Qdrant reachable and whether the collection exists / has points

## Notes

- This server is stateless on purpose. The client keeps the audit trail so text previews can be redacted before saving/exporting anything.
- If `OPENAI_API_KEY` is missing, the server still starts, but indexing and retrieval will fail.
- `index_status` can optionally validate the OpenAI key to prevent live-demo surprises. Use `CHECK_OPENAI_ON_INDEX_STATUS=0` to disable, and tune `OPENAI_INDEX_STATUS_CACHE_SECONDS` to reduce repeated calls.

## Start / stop (workspace scripts)

If you want a one-command launcher for the full demo (server + UI), use:

```bash
cd /c/AI-Agent-Buildathon-2026
bash ./run_demo.sh
```

To stop the UI + server (kills port 8501 + 4200):

```bash
cd /c/AI-Agent-Buildathon-2026
bash ./stop_demo.sh
```

