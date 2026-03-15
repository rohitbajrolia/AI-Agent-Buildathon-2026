# Home Insurance MCP (Client)

This is the client app. It runs a Streamlit UI and a LangGraph workflow that calls the MCP server for evidence, then produces a cited answer (or blocks if the evidence/citations are not defensible).

## What it provides

- Streamlit UI for evidence-backed Q&A and controls.
- LangGraph pipeline:
  `plan -> retrieve -> precedence_check -> validate -> answer -> citation_verify`
- Audit trail with redacted previews and a downloadable JSON trace.
- Manual ingest/index controls and a readiness indicator.
- Job-based ingest/index with progress bars (polls job status).
- Re-index gating: if the index already exists, re-indexing is disabled unless you confirm docs changed.
- Endorsement override check: shows precedence risk signals when endorsements are retrieved.
- Handoff tickets: generate a structured "send to a human reviewer" JSON payload with citations.
- Optional utilities: quote/rating normalization and an "impact snapshot" JSON export.

## Flow (client side, in order)

1. Sidebar -> Self-check to confirm `health`, `index_status`, and (if ready) a small `retrieve_clauses` call.
2. Sidebar -> Refresh Index Status to see whether indexing is needed.
3. If the collection is missing or empty, run Index to Qdrant.
4. If the collection already exists, indexing is skipped unless you check "I added/updated documents - re-index".
5. Click Review Coverage. The workflow runs:
   `plan -> retrieve -> precedence_check -> validate -> answer -> citation_verify`
6. The UI shows the answer, sources, and audit trace, or blocks the run if evidence is weak or citations are invalid.

## Guardrails

- Privacy: the UI warns against PII and defaults to redacted snippets in the audit views.
- Readiness: Q&A is disabled until the index is ready and OpenAI looks healthy (from `index_status`).
- Consent: Q&A is disabled until the user confirms they are using redacted data.
- Grounding: answers are instructed to use only the provided SOURCES.
- Verification: when "Require citations" is on, citations are checked against retrieved chunks; invalid citations are rejected and the run can be blocked.
- The UI shows an always-on disclaimer: educational use only; not legal advice, underwriting, or a binding coverage decision.

## Prereqs

- Python 3.12+
- The MCP server running (`home-insurance-mcp`)
- Qdrant running
- OpenAI API key (used by the client answer step)

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

Set values in `.env`: `MCP_SERVER_URL` and `OPENAI_API_KEY`.

## Install + run

From the workspace root (`C:\AI-Agent-Buildathon-2026`):

```bash
./.venv/Scripts/python.exe -m pip install -e ./home-insurance-mcp-client
```

Start the UI (Git Bash):

```bash
cd /c/AI-Agent-Buildathon-2026/home-insurance-mcp-client
../.venv/Scripts/python.exe -m streamlit run src/client/app.py
```

Or use the workspace helper script:

```bash
cd /c/AI-Agent-Buildathon-2026
bash ./run_ui.sh
```

For a single-command launcher that starts the MCP server and then the UI:

```bash
cd /c/AI-Agent-Buildathon-2026
bash ./run_demo.sh
```

To stop the UI + server:

```bash
cd /c/AI-Agent-Buildathon-2026
bash ./stop_demo.sh
```

## Preflight checks

Run:

```bash
python scripts/demo_preflight.py
python scripts/client_smoke.py
python scripts/graph_smoke.py
```

Good looks like this:
- `demo_preflight.py` prints `PASS (ready)`
- `client_smoke.py` shows `openai_ok: true` and a non-zero `points_count`
- `graph_smoke.py` exits `0`

## Walkthrough

1. Sidebar -> Self-check -> Run self-check.
2. Sidebar -> Refresh Index Status.
3. If not indexed yet: run Index to Qdrant.
4. If indexed already: only re-index after changing docs (check "I added/updated documents - re-index").
5. Optional: Ingest Folder (verifies docs are found and shows progress).
6. Pick a question preset and run it.
7. Open Audit log and download the JSON trace.
8. Optional: Create a handoff ticket and download it.

## Troubleshooting

- Qdrant unreachable: start Qdrant, then refresh index status.
- Not indexed yet or empty: run Index to Qdrant from the sidebar.
- Docs updated but answers look stale: check "I added/updated documents - re-index" and run Index to Qdrant.
- Blocked due to weak evidence: re-index the correct folder or ask a more specific question.

## Notes

- If you need access from another device, run Streamlit with `--server.address 0.0.0.0` and start the MCP server with `MCP_HOST=0.0.0.0`. Then set `MCP_SERVER_URL` to your machine's LAN IP (for example, `http://192.168.1.10:4200/mcp/`).
- Keep documents local and preferably redacted. The audit trace stores short, redacted previews of text.
- The system is designed around citations. If you turn off "Require citations", you are choosing a less strict mode.

## Local state files

The UI writes small local-only files under `.demo_state/`:
- `docs_fingerprint.json`: remembers the last indexed docs snapshot (used to detect likely doc changes)
- `feedback.jsonl`: optional feedback clicks saved locally from the UI

Note: the MCP server also persists handoff tickets under `.demo_state/` by default.
