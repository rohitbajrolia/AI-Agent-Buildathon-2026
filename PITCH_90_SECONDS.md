# 90-second pitch - Coverage Concierge (Policy-Grounded)

Insurance policy packets are long and fragmented - booklet, endorsements, declarations - and teams repeatedly answer the same coverage questions. Today that means agents waste time searching PDFs, answers vary by who handles the call, and there's real risk if guidance is given without strong policy evidence.

Coverage Concierge is a policy-grounded assistant that answers coverage questions with citations from the customer's policy packet.

It's deliberately tool-driven:
- Ingest and index the policy PDFs locally.
- Retrieve the most relevant clauses from Qdrant using MCP tools.
- Validate evidence quality and refuse to answer if evidence is too weak.
- Generate a cited response, then verify that the citations actually match the retrieved snippets.

What you get is a fast, defensible first pass:
- A grounded answer.
- The exact snippets used.
- An audit trace you can download.
- And a structured handoff ticket when a human reviewer needs to take over.

In a pilot, the target outcomes are measurable:
- Reduce clause lookup time from 12 minutes to 7 minutes (about a 42% reduction).
- Improve first-contact resolution by 10-15% for common questions.
- Reduce escalations by 5-10% by making evidence and next verification steps explicit.

The key point: this isn't a chatbot guessing about coverage - it's a workflow that retrieves evidence, validates it, and blocks when it can't be defended.
