import streamlit as st
import asyncio
import json
from pathlib import Path
import re
import hashlib
import time

# When Streamlit runs a file directly, relative imports can be flaky.
# We add the src/ root to sys.path so `client.*` imports work consistently.
import sys

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from client.graph import GraphState, build_graph
from client import mcp_client

st.set_page_config(page_title="Home Insurance Coverage Concierge", layout="wide")

# ---------- ALWAYS-VISIBLE COMPLIANCE HEADER ----------
st.title("🏠 Coverage Concierge (Policy-Grounded)")
st.warning(
    "Educational use only — not legal advice, underwriting, or a binding coverage decision. "
    "Coverage depends on your full policy, endorsements, declarations, and claim facts."
)

with st.expander("Privacy & what NOT to paste (PII guardrail)", expanded=False):
    st.markdown(
        "- Do **not** paste: SSN, bank/credit card numbers, full address, DOB, policy number, claim number.\n"
        "- If your documents contain PII, keep them local (as you are doing). Only the cited snippets are shown.\n"
        "- For the demo, use redacted documents whenever possible."
    )


def _default_docs_dir() -> str:
    # Default docs location for our demo. Override in the sidebar.
    return str((Path(__file__).resolve().parents[3] / "docs").resolve())


_DEMO_STATE_DIR = (Path(__file__).resolve().parents[3] / ".demo_state").resolve()
_DOCS_FINGERPRINT_FILE = _DEMO_STATE_DIR / "docs_fingerprint.json"


def _compute_docs_fingerprint(folder_path: str) -> dict:
    """Build a quick fingerprint of PDFs (names + mtimes + sizes).

    We intentionally do not hash full PDF bytes to keep this fast.
    """
    folder = Path(folder_path).expanduser().resolve()
    pdf_paths = sorted([p for p in folder.rglob("*.pdf") if p.is_file()])

    items: list[dict] = []
    for p in pdf_paths:
        stat = p.stat()
        items.append(
            {
                "path": str(p.relative_to(folder)).replace("\\", "/"),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )

    payload = {
        "root": str(folder).replace("\\", "/"),
        "pdf_count": len(items),
        "items": items,
    }
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()
    return {
        "root": payload["root"],
        "pdf_count": payload["pdf_count"],
        "hash": digest,
        "computed_from": "path+size+mtime_ns",
    }


def _load_saved_docs_fingerprint() -> dict | None:
    try:
        if _DOCS_FINGERPRINT_FILE.exists():
            return json.loads(_DOCS_FINGERPRINT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _save_docs_fingerprint(fp: dict) -> None:
    _DEMO_STATE_DIR.mkdir(parents=True, exist_ok=True)
    _DOCS_FINGERPRINT_FILE.write_text(json.dumps(fp, indent=2), encoding="utf-8")


def _run(coro):
    # Streamlit runs sync code; our MCP helpers are async.
    # Bridge the two with asyncio.run() for predictable execution.
    return asyncio.run(coro)


def _redact_display_text(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s

    # Emails
    s = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[REDACTED_EMAIL]", s)

    # Phone-ish numbers
    s = re.sub(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]", s)

    # Long digit runs (policy/claim/account-ish)
    s = re.sub(r"\b\d{6,}\b", "[REDACTED_NUMBER]", s)
    return s


def _sanitize_retrieved_matches_for_audit(raw_results: list[dict]) -> list[dict]:
    sanitized: list[dict] = []
    for r in raw_results or []:
        sanitized.append(
            {
                "file_name": r.get("file_name"),
                "doc_type": r.get("doc_type"),
                "page_number": r.get("page_number"),
                "chunk_index": r.get("chunk_index"),
                "score": r.get("score"),
                "snippet_redacted": _redact_display_text(r.get("snippet") or ""),
            }
        )
    return sanitized


def _render_next_steps(*, status_payload: dict | None) -> None:
    steps: list[str] = []

    if not status_payload:
        steps.append("Click ‘Refresh Index Status’ in the sidebar.")
        steps.append("If the collection doesn’t exist yet, run ‘Index to Qdrant’.")
    else:
        q_ok = status_payload.get("status") == "ok"
        collection_exists = bool(status_payload.get("collection_exists"))
        points_count = status_payload.get("points_count")
        openai_ok = status_payload.get("openai_ok")
        openai_configured = bool(status_payload.get("openai_configured"))

        if openai_configured and openai_ok is False:
            steps.append("Your OPENAI_API_KEY is present but invalid — update it and refresh index status.")
        elif not openai_configured:
            steps.append("Set OPENAI_API_KEY (server + client), then refresh index status.")
        elif not q_ok:
            steps.append("Make sure Qdrant is running, then refresh index status.")
        elif not collection_exists:
            steps.append("Run ‘Index to Qdrant’ (first-time setup).")
        elif points_count == 0:
            steps.append("Index exists but is empty — run ‘Index to Qdrant’. ")

    steps.append("If results still look thin, try a more specific question (peril + coverage + location).")

    st.info("Next steps:\n- " + "\n- ".join(steps))


with st.sidebar:
    with st.expander("Demo checklist", expanded=True):
        st.caption("A quick path that keeps the live demo predictable.")

        last_health = st.session_state.get("last_health")
        index_status_payload = st.session_state.get("index_status")
        last_ingest = st.session_state.get("last_ingest")
        last_index = st.session_state.get("last_index")

        server_ok = bool(last_health and last_health.get("status") == "ok")
        qdrant_ok = bool(index_status_payload and index_status_payload.get("status") == "ok")
        index_ready = bool(
            index_status_payload
            and index_status_payload.get("status") == "ok"
            and index_status_payload.get("collection_exists")
            and (
                index_status_payload.get("points_count") is None
                or (index_status_payload.get("points_count") or 0) > 0
            )
        )
        openai_ready = bool(index_status_payload and index_status_payload.get("openai_configured"))
        ingest_ok = bool(last_ingest and (last_ingest.get("files_total") or 0) > 0)
        index_ok = bool(last_index and (last_index.get("chunks_indexed") or 0) > 0)

        demo_ready = bool(server_ok and qdrant_ok and index_ready and openai_ready)

        st.write(f"1) Server health: {'OK' if server_ok else 'Run Health'}")
        st.write(f"2) Index status: {'Ready' if index_ready else 'Refresh Status'}")
        st.write(f"3) OpenAI configured: {'OK' if openai_ready else 'Missing OPENAI_API_KEY'}")
        if index_ready:
            st.write("4) Ingest docs: Optional")
            st.write("5) Index docs: Optional")
        else:
            st.write(f"4) Ingest docs: {'Done' if ingest_ok else 'Run Ingest'}")
            st.write(f"5) Index docs: {'Done' if index_ok else 'Run Index'}")
        st.write("6) Ask a question: include the peril + what you want covered")

        if demo_ready:
            st.success("Demo is ready to run")
        else:
            st.warning("Demo not ready yet")

    with st.expander("Policy docs (local) + chunking settings", expanded=False):
        st.caption("Choose where the PDFs live and how we chunk them for retrieval.")
        docs_dir = st.text_input("Docs folder (absolute path)", value=_default_docs_dir())
        max_pages = st.number_input("Max PDF pages (per file)", min_value=1, max_value=200, value=25, step=1)
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1200, step=50)
        overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=150, step=25)

    st.divider()
    st.subheader("Indexing")

    status_col_a, status_col_b = st.columns([1, 1])
    with status_col_a:
        refresh_status = st.button("Refresh Index Status")
    with status_col_b:
        show_status = st.checkbox("Show status details", value=False)

    if refresh_status:
        try:
            st.session_state["index_status"] = _run(mcp_client.index_status())
        except Exception as e:
            st.session_state["index_status"] = {"status": "degraded", "error": str(e)}

    status_payload = st.session_state.get("index_status")
    if status_payload:
        q_ok = status_payload.get("status") == "ok"
        collection_exists = bool(status_payload.get("collection_exists"))
        points_count = status_payload.get("points_count")

        if q_ok and collection_exists and (points_count is None or points_count > 0):
            st.success("Index ready")
        elif q_ok and collection_exists:
            st.warning("Index exists, but empty")
        elif q_ok:
            st.warning("Not indexed yet")
        else:
            st.error("Qdrant unreachable")

        if show_status:
            st.json(status_payload)

    # Indexing intent:
    # - If the collection does NOT exist yet: indexing is required.
    # - If the collection already exists: only index again if the user says they added/changed docs.
    status_payload = st.session_state.get("index_status")
    status_known = bool(status_payload)
    q_ok = bool(status_payload and status_payload.get("status") == "ok")
    collection_exists = bool(status_payload and status_payload.get("collection_exists"))
    index_has_points = bool(
        status_payload
        and status_payload.get("collection_exists")
        and (status_payload.get("points_count") is None or (status_payload.get("points_count") or 0) > 0)
    )

    wants_reindex = False
    docs_changed = False
    if status_known and q_ok and collection_exists:
        if "saved_docs_fingerprint" not in st.session_state:
            st.session_state["saved_docs_fingerprint"] = _load_saved_docs_fingerprint()

        saved_fp = st.session_state.get("saved_docs_fingerprint")
        current_fp = None
        try:
            if Path(docs_dir).exists():
                current_fp = _compute_docs_fingerprint(docs_dir)
        except Exception:
            current_fp = None

        if saved_fp and current_fp and saved_fp.get("root") == current_fp.get("root"):
            docs_changed = bool(saved_fp.get("hash") and current_fp.get("hash") and saved_fp["hash"] != current_fp["hash"])

        if docs_changed:
            st.warning("Docs look different since the last index. Re-index to pick up the changes.")
            if "wants_reindex" not in st.session_state:
                st.session_state["wants_reindex"] = True
        elif index_has_points:
            st.caption("Index already exists. Skipping indexing unless docs changed.")

        wants_reindex = st.checkbox(
            "I added/updated documents — re-index",
            key="wants_reindex",
            value=bool(st.session_state.get("wants_reindex", False)),
            help="Use this only when you changed PDFs under the docs folder.",
        )

    col_a, col_b = st.columns(2)
    with col_a:
        do_ingest = st.button("Ingest Folder")
    with col_b:
        index_disabled = True
        if not status_known:
            index_disabled = True
        elif not q_ok:
            index_disabled = True
        elif not collection_exists:
            index_disabled = False
        else:
            index_disabled = not wants_reindex

        do_index = st.button(
            "Index to Qdrant",
            disabled=index_disabled,
            help=(
                "First refresh index status. Indexing runs automatically only when the collection is missing; "
                "otherwise enable re-index after adding new docs."
            ),
        )

    # Quick health check before running the demo.
    if st.button("Server Health"):
        try:
            payload = _run(mcp_client.health())
            st.session_state["last_health"] = payload
            st.success("Server is up and running")
            with st.expander("Health details", expanded=False):
                st.json(payload)
        except Exception as e:
            st.error(f"Health check failed: {e}")

    if do_ingest:
        try:
            status_line = st.empty()
            bar = st.progress(0)
            details = st.caption("")

            status_line.info("Starting ingest…")
            job = _run(
                mcp_client.start_ingest_job(
                    folder_path=docs_dir,
                    max_pages=int(max_pages),
                    chunk_size=int(chunk_size),
                    overlap=int(overlap),
                )
            )
            job_id = (job or {}).get("job_id")
            if not job_id:
                raise RuntimeError("Failed to start ingest job")

            while True:
                s = _run(mcp_client.job_status(job_id))
                if s.get("status") == "error":
                    raise RuntimeError(s.get("error") or "Job status error")

                status_line.info(s.get("message") or "Ingesting…")
                prog = (s.get("progress") or {})
                files_total = int(prog.get("files_total") or 0)
                files_done = int(prog.get("files_done") or 0)
                chunks_total = prog.get("chunks_total")
                pct = (files_done / files_total) if files_total else 0.0
                bar.progress(min(1.0, max(0.0, pct)))
                extra = f"Files: {files_done}/{files_total}"
                if chunks_total is not None:
                    extra += f" • Chunks counted: {int(chunks_total)}"
                details.caption(extra)

                if s.get("status") in {"completed", "failed"}:
                    if s.get("status") == "failed":
                        raise RuntimeError(s.get("error") or "Ingest failed")
                    payload = s.get("result") or {}
                    st.session_state["last_ingest"] = payload
                    status_line.success("Ingest completed")
                    bar.progress(1.0)
                    break

                time.sleep(0.35)
        except Exception as e:
            st.error(f"Ingest failed: {e}")

    if do_index:
        try:
            status_line = st.empty()
            bar = st.progress(0)
            details = st.caption("")

            status_line.info("Starting indexing…")
            job = _run(
                mcp_client.start_index_job(
                    folder_path=docs_dir,
                    max_pages=int(max_pages),
                    chunk_size=int(chunk_size),
                    overlap=int(overlap),
                    batch_size=64,
                )
            )
            job_id = (job or {}).get("job_id")
            if not job_id:
                raise RuntimeError("Failed to start index job")

            while True:
                s = _run(mcp_client.job_status(job_id))
                if s.get("status") == "error":
                    raise RuntimeError(s.get("error") or "Job status error")

                status_line.info(s.get("message") or "Indexing…")
                prog = (s.get("progress") or {})
                batches_total = int(prog.get("batches_total") or 0)
                batches_done = int(prog.get("batches_done") or 0)
                points_upserted = int(prog.get("points_upserted") or 0)
                files_total = int(prog.get("files_total") or 0)
                files_done = int(prog.get("files_done") or 0)

                pct = (batches_done / batches_total) if batches_total else ((files_done / files_total) if files_total else 0.0)
                bar.progress(min(1.0, max(0.0, pct)))

                extra = f"Files: {files_done}/{files_total} • Batches: {batches_done}/{batches_total or '?'} • Points upserted: {points_upserted}"
                details.caption(extra)

                if s.get("status") in {"completed", "failed"}:
                    if s.get("status") == "failed":
                        raise RuntimeError(s.get("error") or "Index failed")
                    payload = s.get("result") or {}
                    st.session_state["last_index"] = payload
                    status_line.success("Index completed")
                    bar.progress(1.0)

                    # Record the current docs snapshot as the "indexed" baseline.
                    try:
                        fp = _compute_docs_fingerprint(docs_dir)
                        st.session_state["saved_docs_fingerprint"] = fp
                        _save_docs_fingerprint(fp)
                        st.session_state["wants_reindex"] = False
                    except Exception:
                        pass

                    # Keep the sidebar honest right after indexing.
                    try:
                        st.session_state["index_status"] = _run(mcp_client.index_status())
                    except Exception:
                        pass

                    break

                time.sleep(0.35)
        except Exception as e:
            st.error(f"Index failed: {e}")

# ---------- INPUTS ----------
left, right = st.columns([2, 1])

DEMO_QUESTION_PRESETS = {
    "Water backup / sump pump": "Does my policy cover water backup or drainage system issues?",
    "Pipe burst + resulting damage": "If a pipe bursts inside the home, is the resulting water damage covered?",
    "Mold after water loss": "Is mold remediation covered after a covered water loss, and what limits apply?",
    "Roof leak from storm": "If wind or hail damages my roof and rain enters, what is covered and what is excluded?",
    "Loss of use": "If my home is uninhabitable after a covered loss, does the policy cover temporary living expenses?",
}


def _apply_preset_question() -> None:
    choice = st.session_state.get("preset_choice")
    if choice and choice != "(custom)":
        st.session_state["question_text"] = DEMO_QUESTION_PRESETS.get(choice, "")

with left:
    if "question_text" not in st.session_state:
        st.session_state["question_text"] = DEMO_QUESTION_PRESETS["Water backup / sump pump"]

    st.selectbox(
        "Quick question presets (demo)",
        options=["(custom)"] + list(DEMO_QUESTION_PRESETS.keys()),
        index=0,
        key="preset_choice",
        on_change=_apply_preset_question,
    )

    question = st.text_area(
        "Ask a homeowners insurance coverage question",
        key="question_text",
        height=90,
    )

with right:
    st.caption("Demo controls (for compliance + transparency)")
    state = st.selectbox("Jurisdiction / State (demo)", ["IL", "CA", "NY", "TX", "FL"], index=0)
    require_citations = st.checkbox("Require citations (block answers without sources)", value=True)
    consent = st.checkbox(
        "I confirm I’m using redacted/non-sensitive data for this demo.",
        value=False
    )

    with st.expander("Quote / rating summary", expanded=False):
        st.caption("Paste a quote or rating summary. This normalizes it into key fields for the demo.")
        quote_text = st.text_area("Quote / rating text", key="quote_text", height=120)
        if st.button("Normalize quote / rating summary"):
            try:
                st.session_state["quote_snapshot"] = _run(mcp_client.normalize_quote_snapshot(raw_text=quote_text))
                st.success("Normalized")
            except Exception as e:
                st.error(f"Normalize failed: {e}")
        if st.session_state.get("quote_snapshot"):
            st.json(st.session_state["quote_snapshot"])

# ---------- ACTION / PREFLIGHT ----------
status_payload = st.session_state.get("index_status")
demo_ready = bool(
    status_payload
    and status_payload.get("status") == "ok"
    and status_payload.get("collection_exists")
    and (status_payload.get("points_count") is None or (status_payload.get("points_count") or 0) > 0)
    and status_payload.get("openai_configured")
    and (status_payload.get("openai_ok") is not False)
)

if not demo_ready:
    _render_next_steps(status_payload=status_payload)

ask = st.button("Ask Concierge", type="primary", disabled=not (consent and demo_ready))

# ---------- RUN GRAPH ----------
if ask:
    graph = build_graph()

    with st.spinner("Retrieving policy clauses (MCP) + generating answer..."):
        graph_input: GraphState = {"question": question, "state": state, "require_citations": require_citations}
        out = graph.invoke(graph_input)

    answer = out.get("answer", "")
    sources = out.get("sources", "")
    raw_results = out.get("raw_results", [])
    trace = out.get("trace", [])
    run_id = out.get("run_id")
    blocked = bool(out.get("blocked", False))
    validation = out.get("validation", None)

    st.session_state["last_trace"] = trace
    st.session_state["last_run_id"] = run_id

    # ---------- VALIDATION / ENFORCEMENT ----------
    # Graph can block either due to weak evidence (pre-answer) or failed citation verification (post-answer).
    if blocked:
        st.error(
            "Blocked: The agent could not retrieve strong enough evidence to answer safely. "
            "Try ingest/indexing more docs or rephrasing the question."
        )
        if validation:
            st.subheader("Why it was blocked")
            st.json(validation)

            next_actions = validation.get("next_actions") if isinstance(validation, dict) else None
            if next_actions:
                st.subheader("What to do next")
                st.markdown("- " + "\n- ".join(next_actions))
        st.subheader("What the agent found")
        st.code(sources or "(no sources found)")
        _render_next_steps(status_payload=status_payload)

    else:
        # ---------- ANSWER ----------
        st.subheader("Answer (grounded)")
        st.write(answer)

        # ---------- SOURCE TRANSPARENCY ----------
        st.subheader("Sources used (snippets)")
        show_unredacted = st.checkbox("Show unredacted snippets (local demo only)", value=False)
        if raw_results:
            rows = []
            for r in raw_results:
                rows.append(
                    {
                        "file": r.get("file_name"),
                        "type": r.get("doc_type"),
                        "page": r.get("page_number"),
                        "chunk": r.get("chunk_index"),
                        "score": (round(float(r["score"]), 3) if isinstance(r.get("score"), (int, float)) else None),
                    }
                )

            st.dataframe(rows, use_container_width=True, hide_index=True)

            with st.expander("View snippets", expanded=False):
                for i, r in enumerate(raw_results, start=1):
                    file_name = r.get("file_name") or "(unknown file)"
                    doc_type = r.get("doc_type") or "(unknown type)"
                    page = r.get("page_number")
                    chunk = r.get("chunk_index")
                    header = f"{i}. {file_name} | {doc_type}"
                    if page is not None:
                        header += f" | p. {page}"
                    if chunk is not None:
                        header += f" | chunk {chunk}"
                    st.markdown(f"**{header}**")
                    st.caption("Snippet")
                    snippet = (r.get("snippet") or "").strip()
                    st.write(snippet if show_unredacted else _redact_display_text(snippet) or "(empty)")
                    st.divider()
        else:
            st.code(sources)

    # ---------- HANDOFF / ESCALATION ----------
    if not blocked:
        st.subheader("Handoff to human agent")
        st.caption("Useful for compliance: summarize findings without making a binding decision.")
        if st.button("Generate handoff summary"):
            summary = out.get("handoff_summary", None)
            if not summary:
                # fallback: create a simple summary from available fields
                summary = (
                    f"User question: {question}\n"
                    f"State: {state}\n"
                    f"Top sources used:\n{sources}\n"
                    "Request: Please review the full policy packet and confirm coverage and any exclusions."
                )
            st.text_area("Handoff summary", value=summary, height=180)

    # ---------- AUDIT TRAIL ----------
    with st.expander("Audit trail", expanded=True):
        st.caption(
            "This log is designed for demo/audit. It does not store full prompts, and it redacts common PII patterns in text previews."
        )

        sanitized_matches = _sanitize_retrieved_matches_for_audit(raw_results)
        trace_bundle = {
            "run_id": run_id,
            "inputs": {"question": question, "state": state, "require_citations": require_citations},
            "validation": validation,
            "trace": trace,
            "retrieved_matches": sanitized_matches,
        }

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Workflow steps (trace)**")
            st.json(trace)
        with col2:
            st.markdown("**Inputs + retrieved matches**")
            st.json({"inputs": trace_bundle["inputs"], "validation": validation, "retrieved_matches": sanitized_matches})

        st.download_button(
            "Download audit trace (JSON)",
            data=json.dumps(trace_bundle, indent=2, ensure_ascii=False),
            file_name=f"audit_trace_{run_id or 'run'}.json",
            mime="application/json",
        )


with st.expander("Ingestion / indexing results", expanded=False):
    ingest = st.session_state.get("last_ingest")
    index = st.session_state.get("last_index")

    if not ingest and not index:
        st.caption("Run Ingest Folder or Index to Qdrant from the sidebar.")
    if ingest:
        st.markdown("**Last ingest**")
        st.json(ingest)
    if index:
        st.markdown("**Last index**")
        st.json(index)
