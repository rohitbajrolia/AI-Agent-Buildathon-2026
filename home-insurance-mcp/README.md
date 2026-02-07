# Home Insurance MCP (Server)

This repo runs the MCP tool server for the demo. It exposes Streamable HTTP tools that ingest policy docs, index them in Qdrant, and retrieve evidence for the client UI.

## What it provides

- Tool API: `health`, `ingest_folder`, `index_folder_qdrant`, `retrieve_clauses`, `index_status`, `normalize_quote_snapshot`.
- OCR fallback for scanned PDFs (Tesseract + PyMuPDF), when enabled.
- Stable chunk IDs and metadata (file, page, chunk) for citations.
- Docs-root restriction and error redaction for safety.

## Flow (server side, in order)

1. `health` confirms the server is reachable.
2. `ingest_folder` scans docs, extracts text, and returns file summaries.
3. `index_folder_qdrant` chunks text, embeds with OpenAI, and upserts to Qdrant.
4. `retrieve_clauses` embeds a query and returns top matches with snippets.
5. `index_status` reports readiness (Qdrant, collection, OpenAI, OCR).

## Architecture

See `../ARCHITECTURE.md`.

## Security + PII

- Keep real keys in `.env` and out of git.
- Store PDFs under `docs/` only; it is git-ignored.
- Keep answers grounded in retrieved evidence.

## Guardrails

- Docs scope: ingest and index are restricted to the configured docs root.
- PII safety: error text is scrubbed of secrets before returning to the client.
- Key validation: `index_status` can optionally verify the OpenAI key.

## Prereqs

- Python 3.12+
- Qdrant running
- OpenAI API key
- Optional OCR: Tesseract on PATH and PyMuPDF installed

## Start Qdrant (Docker)

Recommended (persistent, Docker Desktop-friendly):

```bash
cd /c/AI-Agent-Buildathon-2026
docker compose up -d
```

This uses a named Docker volume, so your index survives restarts.

Quick one-off (not persistent):

```bash
docker run --rm -p 6333:6333 qdrant/qdrant
```

## Configure

Copy the example env file:

```bash
copy .env.example .env
```

Set values in `.env`. Required: `OPENAI_API_KEY`. Optional: `QDRANT_URL`, `QDRANT_COLLECTION`, `MCP_PORT`, `MCP_HOST`, `CHECK_OPENAI_ON_INDEX_STATUS`, `OPENAI_INDEX_STATUS_CACHE_SECONDS`, `ENABLE_PDF_OCR_FALLBACK`.

## Install + run

From the workspace root (`C:\AI-Agent-Buildathon-2026`):

```bash
./.venv/Scripts/python.exe -m pip install -e ./home-insurance-mcp
```

Start the server (Git Bash):

```bash
cd /c/AI-Agent-Buildathon-2026/home-insurance-mcp/src
../../.venv/Scripts/python.exe -m server.main
```

Or use the workspace helper script:

```bash
cd /c/AI-Agent-Buildathon-2026
bash ./run_server.sh
```

Server URL: `http://127.0.0.1:4200/mcp/`. If you set `MCP_HOST=0.0.0.0`, the server is reachable on your LAN.

## Quick sanity checks

Healthy means:

- `health` returns `status: ok`.
- `index_status` shows Qdrant reachable and a collection with points (after indexing).

## Notes

- The server is stateless by design; the client keeps the audit trail.
- If `OPENAI_API_KEY` is missing, the server still starts, but indexing and retrieval will fail.
- `index_status` can validate the OpenAI key. Set `CHECK_OPENAI_ON_INDEX_STATUS=0` to disable.

## Start / stop (workspace scripts)

For the full demo (server + UI):

```bash
cd /c/AI-Agent-Buildathon-2026
bash ./run_demo.sh
```

To stop the UI + server:

```bash
cd /c/AI-Agent-Buildathon-2026
bash ./stop_demo.sh
```
