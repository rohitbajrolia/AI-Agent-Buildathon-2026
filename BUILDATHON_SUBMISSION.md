# Coverage Concierge - Proposal (Agentic Insurance Assistant)

## 1) Problem statement
Homeowners insurance policy packets are long and fragmented (booklet, declarations, endorsements, change notices). Service and claims teams repeatedly answer the same coverage questions while manually searching PDFs, reconciling endorsements, and explaining conditions and exclusions. This increases handle time, produces inconsistent guidance, and raises the risk of incorrect answers when evidence is missing.

## 2) Proposed solution (single agent, tool-driven)
Coverage Concierge is a policy-grounded assistant that answers coverage questions using retrieved policy evidence. It uses an explicit workflow that can refuse to answer when evidence is weak, and it verifies citations so results stay defensible.

## 3) Workflow (end-to-end)
1. Ingest policy PDFs locally (extract text; summarize files/pages).
2. Index policy text into Qdrant (chunk -> embed -> upsert).
3. Ask a coverage question.
4. Retrieve top matching clauses/snippets via MCP tool.
5. Validate evidence quality (block early if weak or inconsistent).
6. Generate an answer with citations.
7. Verify citations against retrieved snippets; retry once if needed.
8. Present answer + sources + audit trace; optionally create a handoff ticket.

## 4) Technical architecture
- UI: Streamlit
- Workflow engine: LangGraph state machine
- Tool layer: MCP server (Streamable HTTP) exposing ingest/index/retrieve/status tools
- Vector store: Qdrant (Docker, persistent volume)
- LLM provider: OpenAI (embeddings + answer generation)

Reference: ARCHITECTURE.md

## 5) MCP tools (what they do)
- `health`: confirms tool server availability.
- `index_status`: reports Qdrant reachability, collection presence, and embedding-provider configuration.
- `ingest_folder`: extracts text and returns summaries/stats.
- `index_folder_qdrant`: chunks text, generates embeddings, and writes to Qdrant.
- `retrieve_clauses`: retrieves relevant snippets with metadata and scores.
- `start_ingest_job` / `start_index_job` + `job_status`: progress reporting for long-running operations.
- `create_handoff_ticket` / `list_handoff_tickets`: structured payload for human review (redacted previews).

## 6) Expected impact (quantified, with assumptions)
Pilot assumptions:
- Common coverage questions handled by a service team.
- Policy packet is already available and indexed for the relevant product/customer.
- The assistant is used for first-pass guidance with citations; humans retain final decision authority.

Target outcomes (example ranges):
- Reduce policy clause lookup time from 12 minutes to 7 minutes (~42% reduction).
- Improve first-contact resolution by 10-15% for common coverage questions.
- Reduce escalations by 5-10% by providing cited evidence and clear "what to verify" guidance.

## Category alignment
This use case aligns to **Insurance** (customer service and claims support). It focuses on policy-grounded answers, endorsement awareness, and auditability so teams can respond faster while remaining defensible.

## 7) Safety and guardrails
- Local-first documents: PDFs remain under `docs/` (ignored by git).
- Privacy guardrails: UI warns against pasting PII; audit previews redact common patterns.
- Evidence gating: blocks answers when retrieval is too weak to support a safe response.
- Citation verification: checks citations against retrieved snippets; invalid citations trigger a rewrite or block.
- Endorsement override awareness: when endorsements are retrieved, the workflow flags override risk.

## 8) Success metrics (production)
- Median handle time for coverage questions.
- First-contact resolution rate for top question categories.
- Escalation/reopen rate.
- Blocked-run rate and top reasons (used to improve document completeness/indexing).
- Audit trace usage and handoff ticket usage (as proxies for explainability/compliance utility).
