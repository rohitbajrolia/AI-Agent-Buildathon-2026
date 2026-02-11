# Home Insurance MCP (Client)

This repo is the client. It runs a Streamlit UI and a LangGraph workflow that retrieves evidence from the MCP server and produces a cited answer.

## What it provides

- Streamlit UI for evidence-backed Q&A and controls.
- LangGraph pipeline: retrieve -> validate -> answer -> citation_verify.
- Audit trail with redacted previews and a one-click JSON download.
- Manual ingest/index controls and a readiness indicator.
- Job-based ingest/index with progress bars (polls job status).
- Re-index gating: if the index already exists, re-indexing is disabled unless you confirm docs changed.
- A self-check panel to confirm backend readiness without clicking through multiple steps.

## Flow (client side, in order)

1. Sidebar -> Self-check (quick) to confirm `health`, `index_status`, and (if ready) a small `retrieve_clauses` call.
2. Sidebar -> Refresh Index Status to see whether indexing is needed.
3. If the collection is missing or empty, run Index to Qdrant.
4. If the collection already exists, indexing is skipped unless you check “I added/updated documents - re-index”.
5. Click Ask Concierge. The workflow runs `retrieve -> validate -> answer -> citation_verify`.
6. The UI shows the answer, sources, and audit trace, or blocks the run if evidence is weak or citations are invalid.

## Guardrails

- Privacy: the UI warns against PII and defaults to redacted snippets in the audit trace.
- Consent: Q&A is disabled until the user confirms they are using redacted data.
- Grounding: answers require citations, and the system blocks responses when evidence is weak or missing.
- Verification: citations are checked against retrieved chunks; invalid citations are rejected.

## Prereqs

- Python 3.12+
- The MCP server running (`home-insurance-mcp`)
- Qdrant running
- OpenAI API key (for the answer step in the client)

## Configure

Copy the example env file:

```bash
copy .env.example .env
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

For a one-command launcher that starts the MCP server and then the UI:

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

Good looks like this: `demo_preflight.py` prints `PASS (ready)`, `client_smoke.py` shows `openai_ok: true` and a non-zero `points_count`, and `graph_smoke.py` exits `0`.

## Quick walkthrough

1. Sidebar -> Self-check (quick) -> Run self-check.
2. Sidebar -> Refresh Index Status.
3. If not indexed yet: run Index to Qdrant.
4. If indexed already: only re-index after changing docs (check “I added/updated documents - re-index”).
5. Optional: Ingest Folder (verifies docs are found and shows progress).
6. Pick a Quick question preset and ask.
7. Open Audit trail and download the JSON trace.

## Troubleshooting

- Qdrant unreachable: start Qdrant, then refresh index status.
- Not indexed yet or empty: run Index to Qdrant from the sidebar.
- Docs updated but answers look stale: check “I added/updated documents - re-index” and run Index to Qdrant.
- Blocked due to weak evidence: re-index the correct folder or ask a more specific question.

## Notes

- If you need access from another device, run Streamlit with `--server.address 0.0.0.0` and start the MCP server with `MCP_HOST=0.0.0.0`. Then set `MCP_SERVER_URL` to your machine's LAN IP (for example, `http://192.168.1.10:4200/mcp/`).
- Keep documents local and preferably redacted. The audit trace stores short, redacted previews of user text.
- The system is designed around citations. If you turn off Require citations, you are choosing a less strict mode.

## Local state files

The UI writes small local-only files under `.demo_state/`:

- `docs_fingerprint.json`: remembers the last indexed docs snapshot (used to detect likely doc changes).
- `feedback.jsonl`: optional feedback clicks saved locally from the UI.
