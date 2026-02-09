# Home Insurance Coverage Concierge (MCP + Qdrant + Streamlit)

This workspace is a local application that answers homeowners insurance coverage questions grounded in your policy PDFs.
It runs three parts: Qdrant for vector search, an MCP tool server for ingest/index/retrieve, and a Streamlit UI that runs a LangGraph workflow to produce a cited answer.

## What this application provides (brief)

- Policy-grounded Q&A with citations tied to retrieved snippets.
- Explicit ingest and index steps so you control when docs are processed.
- An audit trail with redacted previews for transparency.
- Local-first docs: policy PDFs stay under `docs/` and are git-ignored.

## Flow (correct order, what each step does)

1. Start Qdrant (vector DB). It stores embeddings so retrieval is fast and persistent across restarts.
2. Start the MCP server. It exposes tools like `health`, `ingest_folder`, `index_folder_qdrant`, `retrieve_clauses`, and `index_status`.
3. Start the Streamlit UI. The UI calls MCP tools and runs the LangGraph workflow.
4. Ingest Folder (UI button). The server scans PDFs/images under `docs/`, extracts text (with OCR fallback for scanned pages), and returns file summaries. No embeddings yet.
5. Index to Qdrant (UI button). The server chunks text, generates embeddings, and upserts them to Qdrant in batches. The index status becomes ready.
6. Ask Concierge (UI button). LangGraph runs `retrieve -> validate -> answer -> citation_verify`. If evidence is weak or citations are invalid, the run is blocked. Otherwise the UI shows the answer, sources, and an audit trace download.

## Quick start (Windows + Git Bash)

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

- Server env: `home-insurance-mcp/README.md` -> copy `.env.example` to `.env` and set `OPENAI_API_KEY`
- Client env: `home-insurance-mcp-client/README.md` -> copy `.env.example` to `.env` and set `OPENAI_API_KEY`

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
