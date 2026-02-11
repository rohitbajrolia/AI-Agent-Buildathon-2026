# Coverage Concierge - Project Brief

## 1) Problem statement
Homeowners policy packets are long and fragmented (policy booklet, declarations, endorsements, change notices). Teams repeatedly answer the same coverage questions while manually searching PDFs, reconciling endorsements, and explaining conditions and exclusions. This increases handle time, creates inconsistent guidance across agents, and raises risk when an answer is given without strong policy evidence.

## 2) Proposed solution
Coverage Concierge is a system for answering coverage questions from policy content. It retrieves evidence, applies an explicit decision path, declines low-confidence questions, verifies citations, and produces defensible outputs grounded in the policy packet.

## 3) Workflow (step-by-step)
1. Start Qdrant (vector store).
2. Start the MCP server (tools for ingest, index, retrieve, status).
3. Start the Streamlit UI (operator-facing experience).
4. Ingest documents: extract text from PDFs (summaries only; no embeddings yet).
5. Index documents: chunk text, embed, and upsert to Qdrant (progress tracked by job status).
6. Ask a question: UI runs a workflow `retrieve -> validate -> answer -> citation_verify`.
7. If evidence is weak or citations do not verify, the run is blocked with clear next actions.
8. If successful, UI shows the answer, sources table, snippets, and an audit trace download.

## 4) Architecture (components)
- UI: Streamlit app (operator controls, status, results, audit export).
- Workflow engine: LangGraph state machine.
- Tool layer: MCP server (Streamable HTTP) exposing ingest/index/retrieve/status tools.
- Vector store: Qdrant (Docker, persistent volume).
- Text and embedding service: OpenAI (embeddings + response generation).

See the end-to-end diagram and tool list in ARCHITECTURE.md.

## 5) MCP tools (why they exist)
- `health`: confirms the tool server is available.
- `index_status`: reports whether Qdrant is reachable, whether a collection exists, and whether the embedding provider is configured.
- `ingest_folder`: extracts text and returns file/page/chunk stats to validate document intake.
- `index_folder_qdrant`: chunks, embeds, and stores vectors for retrieval.
- `retrieve_clauses`: returns the most relevant snippets for a question with metadata.
- `job_status`: supports progress reporting for long-running ingest/index operations.
- `create_handoff_ticket` / `list_handoff_tickets`: produces a structured payload for human review (with redacted previews).

## 6) Quantified impact (assumptions + metrics)
Pilot assumptions:
- Common coverage questions handled by a service team.
- Policy packet is indexed for the customer/product.
- The workflow is used for first-pass guidance with citations; humans retain final decision authority.

Target outcomes (example):
- Reduce policy lookup / clause-finding time from 12 minutes to 7 minutes (~42% reduction).
- Improve first-contact resolution by 10-15% for common coverage questions.
- Reduce escalations by 5-10% by providing cited evidence and clear "what to verify" guidance.

## 7) Risks and guardrails
- Privacy: UI warns against pasting PII; audit previews are redacted.
- Grounding: answers require retrieved evidence; weak evidence blocks the run.
- Citation correctness: citations are verified against retrieved snippets; invalid citations trigger a rewrite or block.
- Endorsement overrides: when endorsements are retrieved, the workflow flags override risk to prevent overconfident answers.

## 8) Success metrics (what to measure in production)
- Median handle time for coverage questions.
- First-contact resolution rate for the top 10 question types.
- Escalation/reopen rate.
- "Blocked-run rate" (too little evidence) and top reasons; used as a signal to improve document completeness/indexing.
- Audit trace export usage (as a proxy for explainability/compliance utility).
