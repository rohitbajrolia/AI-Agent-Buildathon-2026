
# Home Insurance MCP (Client)

This repo is the demo client:

- Streamlit UI (policy-grounded Q&A)
- LangGraph workflow (retrieve → validate → answer)
- Audit trace (redacted previews + timings) with a one-click JSON download

## Architecture

See the end-to-end architecture and request flows in [../ARCHITECTURE.md](../ARCHITECTURE.md).

## Security + PII

- Don’t paste or type sensitive personal data into the UI.
- Use redacted docs for the demo.
- Put local policy PDFs under the workspace `docs/` folder — it’s git-ignored on purpose (no accidental “oops I pushed the policy packet”).

## What the UI does

- Lets you explicitly run `Ingest Folder` and `Index to Qdrant` (it will not auto-index on every run)
- Shows an `Index status` indicator so you know if the demo is ready before asking questions
- Enforces grounded outputs: if citations are required and the evidence is weak, it blocks answering and tells you what to do next

The workflow does retrieval → validation → answer, and it logs a redacted audit trace you can download.

## Guardrails

- **Privacy**: the UI warns against PII and defaults to redacted snippets in the audit trace.
- **Consent**: Q&A is disabled until the user confirms they are using redacted data.
- **Grounding**: answers require citations, and the system blocks responses when evidence is weak or missing.
- **Verification**: citations are checked against retrieved chunks; invalid citations are rejected.

## Prereqs

- Python 3.12+
- The MCP server running (`home-insurance-mcp`)
- Qdrant running
- OpenAI API key (needed for the LLM answer step in the client)

## Configure

1) Copy the example env file:

```bash
copy .env.example .env
```

2) Set values in `.env`:

- `MCP_SERVER_URL` (default: `http://127.0.0.1:4200/mcp/`)
- `OPENAI_API_KEY`

## Install + run

For this workspace layout, the most predictable approach is a single shared virtual environment at the workspace root.

From the workspace root (`C:\AI-Agent-Buildathon-2026`):

```bash
./.venv/Scripts/python.exe -m pip install -e ./home-insurance-mcp-client
```

Start the UI (Git Bash):

```bash
cd /c/AI-Agent-Buildathon-2026/home-insurance-mcp-client
../.venv/Scripts/python.exe -m streamlit run src/client/app.py
```

Or use the workspace helper script (starts UI only):

```bash
cd /c/AI-Agent-Buildathon-2026
bash ./run_ui.sh
```

For a one-command launcher that starts the MCP server and then the UI, use:

```bash
cd /c/AI-Agent-Buildathon-2026
bash ./run_demo.sh
```

To stop the UI + server (kills port 8501 + 4200):

```bash
cd /c/AI-Agent-Buildathon-2026
bash ./stop_demo.sh
```

## Demo preflight

Before you start the live Q&A, run:

```bash
python scripts/demo_preflight.py
python scripts/client_smoke.py
python scripts/graph_smoke.py
```

What “good” looks like:
- `demo_preflight.py` prints `PASS (demo-ready)`.
- `client_smoke.py` shows `openai_ok: true` and a non-zero `points_count`.
- `graph_smoke.py` exits `0` (retrieve → validate → answer → citation_verify).

## The 90-second demo script

1) Sidebar → `Server Health` (shows the server is reachable)
2) Sidebar → `Refresh Index Status`
	- If it says “Not indexed yet”, run `Index to Qdrant`
3) Sidebar → `Ingest Folder` (quickly confirms docs are being found)
4) Sidebar → `Index to Qdrant`
5) Pick a “Quick question preset” and ask
6) Open “Audit trail” and download the JSON trace

## Troubleshooting (the common stuff)

- **Qdrant unreachable**: start Qdrant, then refresh index status.
- **Not indexed yet / empty**: run `Index to Qdrant` from the sidebar.
- **Blocked due to weak evidence**: the retrieval step didn’t find enough; reindex the correct folder or ask a more specific question.

## Notes

- If you need access from another device, run Streamlit with `--server.address 0.0.0.0` and start the MCP server with `MCP_HOST=0.0.0.0`. Then set `MCP_SERVER_URL` to your machine’s LAN IP (e.g., `http://192.168.1.10:4200/mcp/`).
- Keep documents local and preferably redacted. The audit trace intentionally stores only short, redacted previews of user text.
- The demo is designed around citations. If you turn off “Require citations”, you’re choosing a less strict mode.

