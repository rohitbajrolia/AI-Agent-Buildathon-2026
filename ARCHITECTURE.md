# Home Insurance MCP - Architecture and Flow

**Purpose**
This project is a local, evidence-backed Q&A workflow for homeowners insurance. It combines a Streamlit UI, a LangGraph pipeline, an MCP server, Qdrant (vector search), and OpenAI (embeddings + answer generation).

**Key files to understand the application**
- UI and user flow: `home-insurance-mcp-client/src/client/app.py:18`
- LangGraph pipeline (retrieve -> validate -> answer -> citation_verify): `home-insurance-mcp-client/src/client/graph.py:682`
- MCP client wrapper (tool calls): `home-insurance-mcp-client/src/client/mcp_client.py:22`
- MCP server tools (ingest/index/retrieve/status): `home-insurance-mcp/src/server/mcp_server.py:290`
- Answer instruction template and format rules: `home-insurance-mcp-client/src/client/prompts.py:1`

**High-level architecture**
- Streamlit UI runs the LangGraph pipeline and renders results. `home-insurance-mcp-client/src/client/app.py:18`
- LangGraph nodes call MCP tools and OpenAI. `home-insurance-mcp-client/src/client/graph.py:95`
- MCP server exposes tools over Streamable HTTP at `/mcp`. `home-insurance-mcp/src/server/mcp_server.py:710`
- Qdrant stores embeddings for retrieval. `home-insurance-mcp/src/server/mcp_server.py:268`

**From user click to UI response (end-to-end flow)**
1. User clicks **Ask Concierge**; UI runs the graph. `home-insurance-mcp-client/src/client/app.py:308`
2. `retrieve_node` calls MCP `retrieve_clauses` and builds the SOURCES block. `home-insurance-mcp-client/src/client/graph.py:95` and `home-insurance-mcp-client/src/client/mcp_client.py:53`
3. `validate_node` blocks early if evidence is missing or weak. `home-insurance-mcp-client/src/client/graph.py:585`
4. `answer_node` calls OpenAI and generates the draft answer. `home-insurance-mcp-client/src/client/graph.py:171`
5. `verify_citations_node` checks citation correctness and may retry once. `home-insurance-mcp-client/src/client/graph.py:460`
6. UI renders either a blocked response with reasons or the final answer. `home-insurance-mcp-client/src/client/app.py:327`

**Retry path (citation verification -> final UI response)**
1. After the initial answer, `verify_citations_node` parses citations and checks them against retrieved chunks. `home-insurance-mcp-client/src/client/graph.py:475`
2. If citations are missing or invalid and retry is available, it rewrites using only allowed citation tags. `home-insurance-mcp-client/src/client/graph.py:478`
3. The rewrite is verified again. `home-insurance-mcp-client/src/client/graph.py:540`
4. If verification still fails, the run is blocked and the answer is cleared. `home-insurance-mcp-client/src/client/graph.py:546`
5. The UI shows the blocked state and next actions. `home-insurance-mcp-client/src/client/app.py:327`

**Where the UI shows the final response**
- Answer display: `home-insurance-mcp-client/src/client/app.py:345`
- Sources table + snippets: `home-insurance-mcp-client/src/client/app.py:350`
- Audit trail download: `home-insurance-mcp-client/src/client/app.py:403`

**MCP server tools and where they live**
- `health`: `home-insurance-mcp/src/server/mcp_server.py:296`
- `ingest_folder` (summaries only): `home-insurance-mcp/src/server/mcp_server.py:336`
- `index_folder_qdrant` (chunk + embed + store): `home-insurance-mcp/src/server/mcp_server.py:380`
- `retrieve_clauses` (vector search): `home-insurance-mcp/src/server/mcp_server.py:468`
- `index_status` (readiness + OpenAI validation): `home-insurance-mcp/src/server/mcp_server.py:507`
- `normalize_quote_snapshot` (quote extraction): `home-insurance-mcp/src/server/mcp_server.py:305`

**Guardrails in code (what they protect)**
- PII warnings + consent gate in the UI. `home-insurance-mcp-client/src/client/app.py:27` and `home-insurance-mcp-client/src/client/app.py:274`
- Redacted audit trail and snippet display. `home-insurance-mcp-client/src/client/app.py:62` and `home-insurance-mcp-client/src/client/graph.py:40`
- Strict citation rules in the instruction template. `home-insurance-mcp-client/src/client/prompts.py:5`
- Evidence-quality gating. `home-insurance-mcp-client/src/client/graph.py:585`
- Citation verification + retry. `home-insurance-mcp-client/src/client/graph.py:460`
- Docs root restriction on server ingest/index. `home-insurance-mcp/src/server/mcp_server.py:64`

**Operational scripts**
- One-command startup (preflight + server + UI): `run_demo.sh:1`
- MCP smoke test (server + index + retrieval): `home-insurance-mcp-client/scripts/client_smoke.py:30`
- LangGraph smoke test (retrieve -> validate -> answer -> verify): `home-insurance-mcp-client/scripts/graph_smoke.py:14`

**If new to MCP / LangGraph**
- Treat MCP as a tool protocol: the client sends `call_tool("retrieve_clauses", ...)` to the server. `home-insurance-mcp-client/src/client/mcp_client.py:53`
- Treat LangGraph as a small state machine: `build_graph()` wires function nodes in order with conditional edges. `home-insurance-mcp-client/src/client/graph.py:682`
