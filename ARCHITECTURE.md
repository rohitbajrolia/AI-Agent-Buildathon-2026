# Home Insurance MCP - Architecture and Flow

**Purpose**
This project is a local, evidence-backed Q&A workflow for homeowners insurance. It combines a Streamlit UI, a LangGraph pipeline, an MCP server, Qdrant (vector search), and OpenAI (embeddings + answer generation).

**Key files to understand the application**
- UI and user flow: `home-insurance-mcp-client/src/client/app.py`
- LangGraph pipeline: `home-insurance-mcp-client/src/client/graph.py`
- MCP client wrapper (tool calls): `home-insurance-mcp-client/src/client/mcp_client.py`
- MCP server tools (ingest/index/retrieve/status): `home-insurance-mcp/src/server/mcp_server.py`
- Answer instruction template and format rules: `home-insurance-mcp-client/src/client/prompts.py`

**High-level architecture**
- Streamlit UI runs the LangGraph pipeline and renders results.
- LangGraph nodes call MCP tools and OpenAI.
- MCP server exposes tools over Streamable HTTP at `/mcp`.
- Qdrant stores embeddings for retrieval.

**From user click to UI response (end-to-end flow)**
1. User clicks **Ask Concierge**; UI invokes the graph with the question, state code, and `require_citations` flag.
2. `plan_retrievals_node` runs a scope gate, then builds a retrieval plan: a list of targeted sub-queries covering coverage grant, definitions, exclusions, endorsements, and conditions.
3. `multi_retrieve_node` executes each sub-query against the MCP `retrieve_clauses` tool, merges results, de-dupes by stable key, and builds the SOURCES block.
4. `precedence_check_node` scans retrieved matches for endorsement override signals and stores a structured precedence summary.
5. `validate_node` gates the answer step — blocks early if evidence is clearly too weak (no matches, very low score, or weak + narrow evidence when citations are required).
6. `answer_node` calls OpenAI chat completions and drafts the response from the SOURCES block using the system prompt and citation rules.
7. `verify_citations_node` checks citation formatting and required-section coverage; if citations are required and verification fails, it retries once, then blocks if still failing.
8. UI renders either a blocked response with reasons, or the final answer with sources and audit trace.

**Scope gate (plan node)**
Before building the retrieval plan, `plan_retrievals_node` runs a keyword-based scope check:
- Questions with no insurance-domain terms at all: blocked immediately.
- Questions about a purely different line of business (e.g., auto coverage only, no homeowners context): blocked with a message directing the user to load the correct policy documents.
- Mixed-LOB questions (e.g., "my car was damaged in the garage during a hailstorm — what's covered?"): pass through to retrieval. The model answers only the portion the loaded homeowners documents support and explicitly defers the auto/other portion to the relevant policy type.

**Retry path (citation verification)**
1. `verify_citations_node` parses citations and checks them against retrieved chunk keys.
2. If citations are missing or invalid and no retry has been attempted yet, the model is asked to rewrite using only the allowed citation tag list.
3. The rewrite is verified again.
4. If verification still fails, the run is blocked and the answer is cleared.
5. UI shows the blocked state with next-action suggestions.

**MCP server tools**
- `health`: checks Qdrant connectivity and returns `status: ok` or `status: degraded` plus collection existence. Located in `_call_tool_impl` in `mcp_server.py`.
- `ingest_folder`: scans docs, extracts text, returns file summaries (no embeddings).
- `index_folder_qdrant`: chunks text, embeds with OpenAI, upserts to Qdrant.
- `retrieve_clauses`: embeds a query and returns top Qdrant matches with snippets and metadata.
- `index_status`: reports Qdrant readiness, collection point count, OpenAI key status, OCR availability.
- `normalize_quote_snapshot`: lightweight extraction of key numbers from pasted quote text.
- `start_ingest_job` / `start_index_job` / `job_status`: async job variants for progress polling.
- `create_handoff_ticket` / `list_handoff_tickets`: structured human-review payload creation and retrieval.

**Guardrails in code (what they protect)**
- PII warning and consent gate in the UI (`app.py`).
- Redacted audit trail and snippet display (`app.py`, `graph.py`).
- Strict citation rules in the instruction template (`prompts.py`).
- Scope gate blocking out-of-domain and wrong-LOB questions (`graph.py` — `plan_retrievals_node`).
- Evidence-quality gate before answering (`graph.py` — `validate_node`).
- Citation verification with retry and block (`graph.py` — `verify_citations_node`).
- Docs root restriction on server ingest/index (`mcp_server.py`).

**Operational scripts**
- One-command startup (preflight + server + UI): `run_demo.sh`
- MCP smoke test (server + index + retrieval): `home-insurance-mcp-client/scripts/client_smoke.py`
- LangGraph smoke test (retrieve -> validate -> answer -> verify): `home-insurance-mcp-client/scripts/graph_smoke.py`

**If new to MCP / LangGraph**
- Treat MCP as a tool protocol: the client sends `call_tool("retrieve_clauses", ...)` to the server over Streamable HTTP.
- Treat LangGraph as a small state machine: `build_graph()` in `graph.py` wires function nodes in order with conditional edges. The entry point is the `plan` node.
