# Home Insurance MCP (Server)

This is the MCP tool server for the application. It exposes Streamable HTTP tools that ingest policy docs, index them in Qdrant, and retrieve evidence for the client UI.

Important: the server does not draft the final answer. The client drafts the answer. The server's job is: ingest -> index -> retrieve (plus readiness + utility tools).

## What it provides

Tools (core):
- `health`
- `ingest_folder`
- `index_folder_qdrant`
- `retrieve_clauses`
- `index_status`
- `normalize_quote_snapshot`

Tools (jobs, for progress):
- `start_ingest_job`, `start_index_job`, `job_status`

Tools (handoff tickets):
- `create_handoff_ticket`
- `list_handoff_tickets`

Other behavior:
- OCR fallback for scanned PDFs (Tesseract + PyMuPDF), when enabled.
- Stable chunk IDs and metadata (file, doc type, page, chunk) for citations.
- Docs-root restriction (folder path must be under the configured docs root).
- Error redaction so secrets do not leak through stack traces.
- Handoff tickets are also written to disk under `.demo_state/` when local persistence succeeds.

## Flow (server side, in order)

1. `health` confirms the server is reachable.
2. `ingest_folder` scans docs, extracts text, and returns file summaries.
3. `index_folder_qdrant` chunks text, embeds with OpenAI, and upserts to Qdrant.
4. `retrieve_clauses` embeds a query and returns top matches with snippets + metadata.
5. `index_status` reports readiness (Qdrant, collection, OpenAI, OCR).

For long-running operations, the client UI uses `start_ingest_job` / `start_index_job` and polls `job_status` to show progress.

## Architecture

See `../ARCHITECTURE.md`.

Transport:
- Streamable HTTP endpoint mounted at `/mcp` (client typically uses `http://127.0.0.1:4200/mcp/`)

## Security + PII

- Keep real keys in `.env` and out of git.
- Store PDFs under the server docs root (default: `../docs/`), which is git-ignored.
- Use redacted packets for demos and avoid pasting PII.

## Guardrails

- Docs scope: ingest and index are restricted to the configured docs root (`MCP_DOCS_ROOT`).
- PII safety: error text is scrubbed of secrets before returning to the client.
- Key validation: `index_status` can optionally verify the OpenAI key.

## Prereqs

- Python 3.12+
- Qdrant running
- OpenAI API key (used for embeddings)
- Optional OCR: Tesseract on PATH and PyMuPDF installed

## Start Qdrant (Docker)

Recommended (persistent, Docker Desktop-friendly):

```bash
cd /c/AI-Agent-Buildathon-2026
docker compose up -d
```

This uses a named Docker volume, so your index survives restarts.

Ad hoc run (not persistent):

```bash
docker run --rm -p 6333:6333 qdrant/qdrant
```

## Configure

Copy the example env file:

PowerShell:
```powershell
copy .env.example .env
```

Git Bash:
```bash
cp .env.example .env
```

Set values in `.env`.

Required:
- `OPENAI_API_KEY`

Common options:
- `QDRANT_URL`, `QDRANT_COLLECTION`
- `MCP_PORT`, `MCP_HOST`
- `MCP_DOCS_ROOT` (docs root enforced by the server)
- `CHECK_OPENAI_ON_INDEX_STATUS`
- `OPENAI_INDEX_STATUS_CACHE_SECONDS`
- `ENABLE_PDF_OCR_FALLBACK`

Optional (handoff ticket persistence path):
- `MCP_TICKETS_STORE` (default: `.demo_state/handoff_tickets.jsonl`)

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

Server URL: `http://127.0.0.1:4200/mcp/`.
If you set `MCP_HOST=0.0.0.0`, the server is reachable on your LAN.

## Sanity Checks

Healthy means:
- `health` returns `status: ok`.
- `index_status` shows Qdrant reachable and a collection with points (after indexing).

## Notes

- The server is tool-first and stateless in the "no chat history" sense. The client keeps the run-level audit log.
- If `OPENAI_API_KEY` is missing, the server still starts, but indexing and retrieval will fail.
- `index_status` can validate the OpenAI key. Set `CHECK_OPENAI_ON_INDEX_STATUS=0` to disable.

## Start / stop (workspace scripts)

For the full run (server + UI):

```bash
cd /c/AI-Agent-Buildathon-2026
bash ./run_demo.sh
```

To stop the UI + server:

```bash
cd /c/AI-Agent-Buildathon-2026
bash ./stop_demo.sh
```
