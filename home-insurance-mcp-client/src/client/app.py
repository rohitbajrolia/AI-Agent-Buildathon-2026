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


def _apply_ui_theme() -> None:
    try:
        css_path = Path(__file__).with_name("ui_theme.css")
        if css_path.exists():
            st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    except Exception:
        return


_apply_ui_theme()

# ---------- ALWAYS-VISIBLE COMPLIANCE HEADER ----------
st.title("Coverage Concierge (Policy-Grounded)")
st.warning(
    "Educational use only - not legal advice, underwriting, or a binding coverage decision. "
    "Coverage depends on your full policy, endorsements, declarations, and claim facts."
)

with st.expander("Privacy: what not to paste (PII)", expanded=False):
    st.markdown(
        "- Do **not** paste: SSN, bank/credit card numbers, full address, DOB, policy number, claim number.\n"
        "- If your documents contain PII, keep them local (as you are doing). Only the cited snippets are shown.\n"
        "- Use redacted documents whenever possible."
    )


def _default_docs_dir() -> str:
    # Default docs location. Override in the sidebar.
    return str((Path(__file__).resolve().parents[3] / "docs").resolve())


_DEMO_STATE_DIR = (Path(__file__).resolve().parents[3] / ".demo_state").resolve()
_DOCS_FINGERPRINT_FILE = _DEMO_STATE_DIR / "docs_fingerprint.json"
_FEEDBACK_FILE = _DEMO_STATE_DIR / "feedback.jsonl"


def _append_feedback(record: dict) -> None:
    try:
        _DEMO_STATE_DIR.mkdir(parents=True, exist_ok=True)
        with _FEEDBACK_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        return


def _compute_docs_fingerprint(folder_path: str) -> dict:
    """Build a quick fingerprint of PDFs (names + mtimes + sizes).

    We avoid hashing full PDFs here. It is slow and no fun.
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


def _evidence_strength_from_validation(validation: dict | None) -> tuple[str, str, list[str]]:
    if not isinstance(validation, dict):
        return ("Unknown", "info", ["No validation stats"])

    stats_obj = validation.get("stats")
    stats: dict = stats_obj if isinstance(stats_obj, dict) else {}
    result_count = stats.get("result_count")
    unique_files = stats.get("unique_files")
    unique_doc_types = stats.get("unique_doc_types")
    max_score = stats.get("max_score")

    rc = int(result_count) if isinstance(result_count, (int, float)) else None
    uf = int(unique_files) if isinstance(unique_files, (int, float)) else None
    ud = int(unique_doc_types) if isinstance(unique_doc_types, (int, float)) else None
    score = float(max_score) if isinstance(max_score, (int, float)) else None

    reasons: list[str] = []
    if score is not None:
        reasons.append(f"Top relevance {score:.2f}")
    if rc is not None:
        reasons.append(f"Matches {rc}")
    if uf is not None:
        reasons.append(f"Files {uf}")
    if ud is not None:
        reasons.append(f"Doc types {ud}")

    if rc == 0 or (score is not None and score < 0.10):
        return ("Weak", "error", reasons)
    if (score is not None and score < 0.25) or (uf is not None and uf < 2) or (ud is not None and ud < 2):
        return ("Medium", "warning", reasons)
    if score is None and rc is None:
        return ("Unknown", "info", reasons)
    return ("Strong", "success", reasons)


def _render_evidence_strength(label: str, kind: str, reasons: list[str]) -> None:
    msg = f"Evidence strength: {label}"
    if kind == "success":
        st.success(msg)
    elif kind == "warning":
        st.warning(msg)
    elif kind == "error":
        st.error(msg)
    else:
        st.info(msg)
    if reasons:
        st.caption("Signals: " + " | ".join(reasons))


def _render_run_summary(*, validation: dict | None, run_seconds: float | None) -> None:
    stats = validation.get("stats") if isinstance(validation, dict) else {}
    if not isinstance(stats, dict):
        stats = {}

    run_time = f"{run_seconds:.1f}s" if isinstance(run_seconds, (int, float)) else "—"
    match_count = stats.get("result_count") or "—"
    file_count = stats.get("unique_files") or "—"

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Run time", run_time)
    with col_b:
        st.metric("Matches", match_count)
    with col_c:
        st.metric("Files", file_count)


def _render_next_steps(*, status_payload: dict | None) -> None:
    steps: list[str] = []

    if not status_payload:
        steps.append("Click 'Refresh Index Status' in the sidebar.")
        steps.append("If the collection doesn't exist yet, run 'Index to Qdrant'.")
    else:
        q_ok = status_payload.get("status") == "ok"
        collection_exists = bool(status_payload.get("collection_exists"))
        points_count = status_payload.get("points_count")
        openai_ok = status_payload.get("openai_ok")
        openai_configured = bool(status_payload.get("openai_configured"))

        if openai_configured and openai_ok is False:
            steps.append("Your OPENAI_API_KEY is present but invalid - update it and refresh index status.")
        elif not openai_configured:
            steps.append("Set OPENAI_API_KEY (server + client), then refresh index status.")
        elif not q_ok:
            steps.append("Make sure Qdrant is running, then refresh index status.")
        elif not collection_exists:
            steps.append("Run 'Index to Qdrant' (first-time setup).")
        elif points_count == 0:
            steps.append("Index exists but is empty - run 'Index to Qdrant'.")

    steps.append("If results still look thin, try a more specific question (peril + coverage + location).")

    st.info("Next steps:\n- " + "\n- ".join(steps))


with st.sidebar:
    with st.expander("Setup checklist", expanded=True):
        st.caption("A quick path that keeps setup predictable.")

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
            st.success("Ready to run")
        else:
            st.warning("Not ready yet")

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
            "I added/updated documents - re-index",
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

    # Quick health check before running.
    if st.button("Server Health"):
        try:
            payload = _run(mcp_client.health())
            st.session_state["last_health"] = payload
            st.success("Server is up and running")
            with st.expander("Health details", expanded=False):
                st.json(payload)
        except Exception as e:
            st.error(f"Health check failed: {e}")

    with st.expander("Self-check (quick)", expanded=False):
        st.caption("Runs in order: health -> index status -> retrieve sample")

        if "self_check_query" not in st.session_state:
            st.session_state["self_check_query"] = "water damage coverage"

        self_check_query = st.text_input(
            "Sample retrieve query",
            key="self_check_query",
            help="Use a generic phrase. Avoid personal or claim-specific info.",
        )

        run_self_check = st.button("Run self-check")

        if run_self_check:
            result: dict = {
                "ran_unix": int(time.time()),
                "health": None,
                "index_status": None,
                "retrieve": None,
                "errors": [],
            }

            # 1) Health
            try:
                result["health"] = _run(mcp_client.health())
            except Exception as e:
                result["errors"].append(f"health: {e}")

            # 2) Index status
            try:
                result["index_status"] = _run(mcp_client.index_status())
            except Exception as e:
                result["errors"].append(f"index_status: {e}")

            # 3) Retrieve (only if index looks ready)
            try:
                status_payload = result.get("index_status") or {}
                ready = bool(
                    status_payload
                    and status_payload.get("status") == "ok"
                    and status_payload.get("collection_exists")
                    and (
                        status_payload.get("points_count") is None
                        or (status_payload.get("points_count") or 0) > 0
                    )
                )
                if ready:
                    result["retrieve"] = _run(mcp_client.retrieve_clauses(query=str(self_check_query), top_k=2))
                else:
                    result["retrieve"] = {"skipped": True, "reason": "Index not ready"}
            except Exception as e:
                result["errors"].append(f"retrieve_clauses: {e}")

            st.session_state["self_check_result"] = result

        sc = st.session_state.get("self_check_result")
        if isinstance(sc, dict):
            health_ok = bool((sc.get("health") or {}).get("status") == "ok")
            idx = sc.get("index_status") or {}
            idx_ok = bool(idx.get("status") == "ok")
            idx_ready = bool(
                idx_ok
                and idx.get("collection_exists")
                and (idx.get("points_count") is None or (idx.get("points_count") or 0) > 0)
            )

            st.write(f"1) Health: {'OK' if health_ok else 'Not OK'}")
            st.write(f"2) Index status: {'OK' if idx_ok else 'Not OK'}")
            st.write(f"3) Index ready: {'Yes' if idx_ready else 'No'}")

            r = sc.get("retrieve")
            if isinstance(r, dict) and r.get("skipped"):
                st.info("Retrieve skipped (index not ready)")
            elif isinstance(r, dict):
                results = r.get("results") or []
                if results:
                    st.success(f"Retrieve: OK ({len(results)} matches)")
                    first = results[0]
                    st.caption("First match (redacted preview)")
                    st.json(
                        {
                            "file_name": first.get("file_name"),
                            "doc_type": first.get("doc_type"),
                            "page_number": first.get("page_number"),
                            "score": first.get("score"),
                            "snippet_redacted": _redact_display_text(first.get("snippet") or ""),
                        }
                    )
                else:
                    st.warning("Retrieve returned 0 matches")

            errs = sc.get("errors") or []
            if errs:
                st.error("Self-check errors")
                st.code("\n".join(str(e) for e in errs))

            with st.expander("Self-check details", expanded=False):
                st.json(sc)

    if do_ingest:
        try:
            status_line = st.empty()
            bar = st.progress(0)
            details = st.caption("")

            status_line.info("Starting ingest...")
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

                status_line.info(s.get("message") or "Ingesting...")
                prog = (s.get("progress") or {})
                files_total = int(prog.get("files_total") or 0)
                files_done = int(prog.get("files_done") or 0)
                chunks_total = prog.get("chunks_total")
                pct = (files_done / files_total) if files_total else 0.0
                bar.progress(min(1.0, max(0.0, pct)))
                extra = f"Files: {files_done}/{files_total}"
                if chunks_total is not None:
                    extra += f" - Chunks counted: {int(chunks_total)}"
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

            status_line.info("Starting indexing...")
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

                status_line.info(s.get("message") or "Indexing...")
                prog = (s.get("progress") or {})
                batches_total = int(prog.get("batches_total") or 0)
                batches_done = int(prog.get("batches_done") or 0)
                points_upserted = int(prog.get("points_upserted") or 0)
                files_total = int(prog.get("files_total") or 0)
                files_done = int(prog.get("files_done") or 0)

                pct = (batches_done / batches_total) if batches_total else ((files_done / files_total) if files_total else 0.0)
                bar.progress(min(1.0, max(0.0, pct)))

                extra = f"Files: {files_done}/{files_total} - Batches: {batches_done}/{batches_total or '?'} - Points upserted: {points_upserted}"
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
        "Quick question presets",
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
    st.caption("Controls (compliance + transparency)")
    state = st.selectbox("Jurisdiction / State", ["IL", "CA", "NY", "TX", "FL"], index=0)
    require_citations = st.checkbox("Require citations (block answers without sources)", value=True)
    consent = st.checkbox(
        "I confirm I'm using redacted/non-sensitive data.",
        value=False
    )

    with st.expander("Impact & metrics", expanded=False):
        st.caption("Keep these numbers honest. If these are estimates, say so.")

        if "impact_metrics" not in st.session_state:
            st.session_state["impact_metrics"] = {
                "baseline_handle_time_minutes": 12.0,
                "target_handle_time_minutes": 7.0,
                "first_contact_resolution_uplift_pct": 12.0,
                "deflection_rate_pct": 15.0,
                "escalation_reduction_pct": 10.0,
                "notes": "Assumptions: common coverage questions; policy packet indexed; agent used for first-pass guidance with citations.",
            }

        m = dict(st.session_state.get("impact_metrics") or {})

        baseline = st.number_input(
            "Baseline handle time (minutes)",
            min_value=1.0,
            max_value=120.0,
            value=float(m.get("baseline_handle_time_minutes", 12.0)),
            step=1.0,
        )
        target = st.number_input(
            "Target handle time (minutes)",
            min_value=1.0,
            max_value=120.0,
            value=float(m.get("target_handle_time_minutes", 7.0)),
            step=1.0,
        )
        fcr = st.number_input(
            "First-contact resolution uplift (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(m.get("first_contact_resolution_uplift_pct", 12.0)),
            step=1.0,
        )
        deflect = st.number_input(
            "Deflection rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(m.get("deflection_rate_pct", 15.0)),
            step=1.0,
        )
        esc_red = st.number_input(
            "Escalation reduction (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(m.get("escalation_reduction_pct", 10.0)),
            step=1.0,
        )
        notes = st.text_area(
            "Assumptions / notes",
            value=str(m.get("notes") or ""),
            height=90,
        )

        reduction_pct = 0.0
        if baseline > 0:
            reduction_pct = max(0.0, min(100.0, (baseline - target) / baseline * 100.0))

        st.info(
            f"Estimated handle-time reduction: {reduction_pct:.0f}% (from {baseline:.0f} -> {target:.0f} minutes)"
        )

        impact_payload = {
            "baseline_handle_time_minutes": float(baseline),
            "target_handle_time_minutes": float(target),
            "estimated_handle_time_reduction_pct": float(round(reduction_pct, 2)),
            "first_contact_resolution_uplift_pct": float(fcr),
            "deflection_rate_pct": float(deflect),
            "escalation_reduction_pct": float(esc_red),
            "notes": notes,
        }

        st.session_state["impact_metrics"] = impact_payload
        st.download_button(
            "Download impact snapshot (JSON)",
            data=json.dumps(impact_payload, indent=2, ensure_ascii=False),
            file_name="impact_snapshot.json",
            mime="application/json",
        )

    with st.expander("Quote / rating summary", expanded=False):
        st.caption("Paste a quote or rating summary. This normalizes it into key fields.")
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

def _run_graph_once(*, question: str, state: str, require_citations: bool) -> dict:
    graph = build_graph()
    graph_input: GraphState = {"question": question, "state": state, "require_citations": require_citations}
    return graph.invoke(graph_input)


def _render_run(*, out: dict, question: str, state: str, require_citations: bool, status_payload: dict | None) -> None:
    answer = out.get("answer", "")
    sources = out.get("sources", "")
    raw_results = out.get("raw_results", [])
    trace = out.get("trace", [])
    run_id = out.get("run_id")
    blocked = bool(out.get("blocked", False))
    validation = out.get("validation", None)

    st.session_state["last_trace"] = trace
    st.session_state["last_run_id"] = run_id

    strength_label, strength_kind, strength_reasons = _evidence_strength_from_validation(
        validation if isinstance(validation, dict) else None
    )
    _render_evidence_strength(strength_label, strength_kind, strength_reasons)
    _render_run_summary(validation=validation if isinstance(validation, dict) else None, run_seconds=st.session_state.get("last_run_seconds"))

    top_a, top_b, top_c = st.columns([1, 1, 3])
    with top_a:
        if st.button("Retry", key=f"retry_{run_id or 'run'}"):
            st.session_state["retry_requested"] = True
            st.rerun()
    with top_b:
        if st.button("Edit question", key=f"editq_{run_id or 'run'}"):
            st.session_state["preset_choice"] = "(custom)"
            st.session_state["question_text"] = question
            st.rerun()
    with top_c:
        note_key = f"fb_note_{run_id or 'run'}"
        st.text_input("Feedback note (optional)", key=note_key, placeholder="Short note")
        fb1, fb2, fb3 = st.columns([1, 1, 2])
        with fb1:
            up = st.button("Helpful", key=f"fb_up_{run_id or 'run'}")
        with fb2:
            down = st.button("Needs work", key=f"fb_down_{run_id or 'run'}")
        with fb3:
            st.caption("Saved locally to .demo_state/")

        if up or down:
            _append_feedback(
                {
                    "created_unix": int(time.time()),
                    "run_id": run_id,
                    "question_redacted": _redact_display_text(question),
                    "state": state,
                    "require_citations": bool(require_citations),
                    "rating": "up" if up else "down",
                    "note": str(st.session_state.get(note_key) or "").strip(),
                }
            )
            st.success("Feedback saved")

    if blocked:
        st.error(
            "Blocked: The workflow could not retrieve strong enough evidence to answer safely. "
            "Try ingest/indexing more docs or rephrasing the question."
        )
        if validation:
            st.subheader("Why it was blocked")
            st.json(validation)

            next_actions = validation.get("next_actions") if isinstance(validation, dict) else None
            if next_actions:
                st.subheader("What to do next")
                st.markdown("- " + "\n- ".join(next_actions))

        st.subheader("What was retrieved")
        st.code(sources or "(no sources found)")
        _render_next_steps(status_payload=status_payload)
        return

    st.subheader("Answer (with sources)")
    st.write(answer)

    endorsement_signals = out.get("endorsement_signals")
    if isinstance(endorsement_signals, dict) and endorsement_signals.get("present"):
        with st.expander("Endorsement override check", expanded=False):
            st.caption("When endorsements are retrieved, treat them as potential overrides and verify the exact wording.")
            st.json(endorsement_signals)

    st.subheader("Sources used (snippets)")
    show_unredacted = st.checkbox("Show unredacted snippets (local only)", value=False)
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

    st.subheader("Handoff to human agent")
    st.caption("Share a structured summary with citations for a human reviewer.")

    col_h1, col_h2, col_h3 = st.columns([1, 1, 1])
    with col_h1:
        create_ticket = st.button("Create handoff ticket (MCP)")
    with col_h2:
        show_ticket = st.checkbox("Show last ticket", value=True)
    with col_h3:
        list_tickets = st.button("List tickets")

    sanitized_matches = _sanitize_retrieved_matches_for_audit(raw_results)

    if create_ticket:
        try:
            payload = _run(
                mcp_client.create_handoff_ticket(
                    question=question,
                    state=state,
                    answer=answer,
                    sources=sources,
                    run_id=run_id,
                    retrieved_matches=sanitized_matches,
                    notes="Includes citations and redacted snippet previews only.",
                )
            )
            st.session_state["last_handoff_ticket"] = payload
            ticket_id = (payload or {}).get("ticket_id")
            if ticket_id:
                st.success(f"Created ticket {ticket_id}")
            else:
                st.success("Created ticket")
        except Exception as e:
            st.error(f"Ticket creation failed: {e}")

    if list_tickets:
        try:
            st.session_state["handoff_tickets"] = _run(mcp_client.list_handoff_tickets(limit=20))
        except Exception as e:
            st.error(f"List tickets failed: {e}")

    if show_ticket and st.session_state.get("last_handoff_ticket"):
        t = st.session_state.get("last_handoff_ticket")
        st.json(t)
        st.download_button(
            "Download last handoff ticket (JSON)",
            data=json.dumps(t, indent=2, ensure_ascii=False),
            file_name=f"handoff_ticket_{(t or {}).get('ticket_id') or (run_id or 'run')}.json",
            mime="application/json",
        )

    if st.session_state.get("handoff_tickets"):
        with st.expander("Recent tickets", expanded=False):
            st.json(st.session_state.get("handoff_tickets"))

    with st.expander("Audit trail", expanded=True):
        st.caption(
            "This log is for audit and troubleshooting. It does not store full inputs, and it redacts common PII patterns in text previews."
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


run_requested = bool(ask or st.session_state.pop("retry_requested", False))
if run_requested:
    st.session_state["last_inputs"] = {
        "question": question,
        "state": state,
        "require_citations": bool(require_citations),
    }

    run_started = time.time()
    with st.spinner("Retrieving policy clauses (MCP) + generating answer..."):
        st.session_state["last_out"] = _run_graph_once(
            question=question,
            state=state,
            require_citations=bool(require_citations),
        )
    st.session_state["last_run_seconds"] = time.time() - run_started

last_out = st.session_state.get("last_out")
last_inputs = st.session_state.get("last_inputs")
if isinstance(last_out, dict) and isinstance(last_inputs, dict):
    _render_run(
        out=last_out,
        question=str(last_inputs.get("question") or ""),
        state=str(last_inputs.get("state") or ""),
        require_citations=bool(last_inputs.get("require_citations", True)),
        status_payload=status_payload,
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
