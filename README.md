# Home Insurance Coverage Concierge (MCP + Qdrant + Streamlit)

Coverage Concierge is a local-first homeowners insurance Q&A assistant. You point it at a folder of policy documents (policy booklet, declarations, endorsements, notices). It indexes them into Qdrant and answers questions using only retrieved snippets from that packet.

It runs as three parts:
- Qdrant (Docker): persistent vector index for retrieval
- MCP server (Python): ingest/index/retrieve tools over Streamable HTTP at `/mcp/`
- Streamlit client (Python): LangGraph workflow + citation verification + audit UI

## What this application provides (brief)

- Evidence-backed Q&A with citations tied to retrieved policy snippets.
- Explicit ingest and index steps so you control when docs are processed.
- Fail-closed mode: when citations are required, answers are blocked unless sources are present and citations verify.
- Planned multi-retrieve (not just one search): the workflow breaks the question into retrieval topics, runs multiple retrieval calls, then dedupes/merges evidence.
- Endorsement precedence signals: retrieved endorsements are surfaced as potential overrides and flagged for verification.
- Audit trail: retrieval plan, evidence summary, precedence signals, and a one-click JSON download.
- Redacted snippet previews by default (with an explicit local-only toggle to view unredacted).
- Handoff tickets: generate a structured "send to a human reviewer" payload with citations.
- Optional utilities in the UI: quote/rating normalization, and an "impact snapshot" JSON export.
- Local-first docs: policy PDFs stay under `docs/` and are git-ignored.

## Flow (correct order)

1. Start Qdrant (vector DB). It stores embeddings so retrieval is fast and persistent across restarts.
2. Start the MCP server. It exposes tools like `health`, `ingest_folder`, `index_folder_qdrant`, `retrieve_clauses`, and `index_status`.
3. Start the Streamlit UI. The UI calls MCP tools and runs the LangGraph workflow.
4. (Optional) Run the sidebar "Self-check (quick)". It validates `health`, `index_status`, and a small retrieval call when the index is ready.
5. Ingest Folder (UI button). The server scans PDFs/images under your docs folder, extracts text (with OCR fallback for scanned pages when enabled), and returns file summaries. No embeddings yet.
6. Index to Qdrant (UI button). The server chunks text, generates embeddings, and upserts them to Qdrant in batches. The index status becomes ready.
7. Ask Concierge (UI button). The workflow runs:
   `plan -> retrieve -> precedence_check -> validate -> answer -> citation_verify`
   If evidence is weak or citations are invalid, the run is blocked. Otherwise the UI shows the answer, sources, and an audit trace download.

Indexing note: if a Qdrant collection already exists, the UI skips re-indexing unless you confirm docs changed (checkbox: "I added/updated documents - re-index").

Docs folder note (important):
- The server enforces a docs root for safety (default: `<workspace>/docs`).
- The "Docs folder" you enter in the UI must be under that server docs root.
- If you want to use a different folder, set `MCP_DOCS_ROOT` on the server to match.

## UI guardrails (exact behavior)

- "Ask Concierge" is disabled until all conditions are true:
  - Index is ready (Qdrant collection exists and has points; index status is OK)
  - OpenAI is configured and looks healthy (from `index_status`)
  - You check: "I confirm I'm using redacted/non-sensitive data."
- "Require citations (block answers without sources)" is wired with `on_change=_clear_last_run`.
  - Practical effect: the moment you toggle it, the previously produced answer/sources/audit panel is cleared immediately.
- Snippet display is redacted by default.
  - Toggle: "Show unredacted snippets (local only)".
  - Downloads (audit trace / handoff ticket) include redacted snippet previews only.
- The UI shows an always-on disclaimer: educational use only; not legal advice, underwriting, or a binding coverage decision.

## What leaves your machine (important)

This app is "local-first" in the sense that your PDFs stay on disk under `docs/` (git-ignored).
But the workflow does send text to OpenAI:

- Server: policy text chunks (indexing) and query text (retrieval embeddings)
- Client: your question + selected policy snippets (answer drafting)

Bottom line: use redacted documents for demos and avoid pasting PII.

## Workflow internals (LangGraph)

High-level sequence:
1. `plan`: create a retrieval plan (topics + doc types) from the question.
2. `retrieve` (multi-retrieve): execute multiple retrieval calls, then merge/dedupe results.
3. `precedence_check`: scan retrieved matches for endorsement/override signals and emit a structured summary for operator review.
4. `validate`: compute evidence strength metrics (count, unique files/doc types, top score). If it is too weak (and citations are required), the run is blocked.
5. `answer`: draft a response from retrieved evidence.
6. `citation_verify`: verify citation formatting/required sections; if citations are required and verification fails, the run is blocked.

What you will see in the UI:
- Evidence strength (Weak/Medium/Strong) derived from validation stats
- "Endorsement override check" expander when endorsements are present
- Sources table + redacted snippets + audit trace download
- Handoff ticket creation + download

## Optional UI utilities

- "Quote / rating summary": calls an MCP tool to normalize pasted quote text into key fields.
- "Impact & metrics": lets you record demo/pilot assumptions and export an impact snapshot JSON.
- Feedback buttons ("Helpful" / "Needs work") are saved locally to `.demo_state/feedback.jsonl`.

## Prerequisites (machine setup)

Required:
- Python 3.12+
- Docker Desktop (for Qdrant)
- Git Bash (recommended) or any shell that can run `bash` scripts

Optional:
- Tesseract OCR (only needed if your PDFs are scanned images and you enable OCR fallback)

Ports (defaults):
- Qdrant: `6333`
- MCP server: `4200`
- Streamlit UI: `8501`

## Quick start (Windows)

### 0) Start Qdrant (persistent)

From the workspace root:

```bash
docker compose up -d
```

### 1) Create one shared virtualenv at the workspace root

```bash
python -m venv .venv
```

### 2) Install both packages into that venv

```bash
./.venv/Scripts/python.exe -m pip install -U pip
./.venv/Scripts/python.exe -m pip install -e ./home-insurance-mcp
./.venv/Scripts/python.exe -m pip install -e ./home-insurance-mcp-client
```

### 3) Configure environment files

You need two `.env` files (kept local and ignored by git):
- Server env: `home-insurance-mcp/.env` (set `OPENAI_API_KEY`, and optionally `MCP_DOCS_ROOT`, `ENABLE_PDF_OCR_FALLBACK`, etc.)
- Client env: `home-insurance-mcp-client/.env` (set `MCP_SERVER_URL` and `OPENAI_API_KEY`)

See:
- `home-insurance-mcp/README.md`
- `home-insurance-mcp-client/README.md`

### 4) One-command run (server + UI)

```bash
bash ./run_demo.sh
```

Stop everything:

```bash
bash ./stop_demo.sh
```

## Where to look next

- Overall diagrams and flows: `ARCHITECTURE.md`
- Server setup + Qdrant notes: `home-insurance-mcp/README.md`
- Client UI + demo steps: `home-insurance-mcp-client/README.md`

## Testing (MCP)

- MCP URL (default): `http://127.0.0.1:4200/mcp/`
- MCP Inspector: connect to the URL above and run `health`, then `index_status`.

Smoke test (tool calls):

```bash
cd home-insurance-mcp-client
python scripts/client_smoke.py
```

## Impact (demo targets)

These are demo/pilot targets and assumptions. Replace them with measured results when you have them:
- Reduce policy clause lookup time from 12 minutes to 7 minutes (~42% reduction)
- Improve first-contact resolution by 10-15% for common coverage questions
- Reduce escalations by 5-10% by making evidence and "what to verify" explicit
