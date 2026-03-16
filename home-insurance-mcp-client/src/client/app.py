import streamlit as st
import asyncio
import json
from pathlib import Path
import re
import hashlib
import time

# When Streamlit runs a file directly, relative imports can be unreliable.
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
st.title("Coverage Concierge")
st.caption(
    "Educational use only — not legal advice or a binding coverage decision. "
    "All real coverage and claims decisions require a qualified human reviewer."
)

with st.expander("Privacy notice", expanded=False):
    st.markdown(
        "Do **not** paste: SSN, bank/credit card numbers, full address, DOB, policy number, or claim number. "
        "Only cited document snippets are shown in the UI. Use redacted documents whenever possible."
    )


def _default_docs_dir() -> str:
    # Default docs location. Override in the sidebar.
    return str((Path(__file__).resolve().parents[3] / "docs").resolve())


_DEMO_STATE_DIR = (Path(__file__).resolve().parents[3] / ".demo_state").resolve()
_DOCS_FINGERPRINT_FILE = _DEMO_STATE_DIR / "docs_fingerprint.json"
_FEEDBACK_FILE = _DEMO_STATE_DIR / "feedback.jsonl"

_SUPPORTED_DOC_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg"})
_KB_CHOICE_USE_CURRENT = "Use the current indexed policy documents"
_KB_CHOICE_ADD_UPDATE = "Add or update my policy documents"
_STATUS_BUTTON_LABEL = "Check policy documents status"
_HEALTH_BUTTON_LABEL = "Check service health"
_SCAN_BUTTON_LABEL = "Scan policy documents"
_BUILD_BUTTON_LABEL = "Build searchable policy documents"

if "consent_confirmed" not in st.session_state:
    st.session_state["consent_confirmed"] = False
if "consent_confirmed_input" not in st.session_state:
    st.session_state["consent_confirmed_input"] = bool(st.session_state["consent_confirmed"])
if "wants_reindex" not in st.session_state:
    st.session_state["wants_reindex"] = False
if "wants_reindex_input" not in st.session_state:
    st.session_state["wants_reindex_input"] = bool(st.session_state["wants_reindex"])
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []


def _append_feedback(record: dict) -> None:
    try:
        _DEMO_STATE_DIR.mkdir(parents=True, exist_ok=True)
        with _FEEDBACK_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        return


def _set_setup_notice(kind: str, message: str) -> None:
    st.session_state["setup_notice"] = {"kind": kind, "message": message}


def _render_setup_notice() -> None:
    notice_obj = st.session_state.pop("setup_notice", None)
    notice = notice_obj if isinstance(notice_obj, dict) else {}
    message = str(notice.get("message") or "").strip()
    if not message:
        return

    kind = str(notice.get("kind") or "info").lower()
    if kind == "success":
        st.success(message)
    elif kind == "warning":
        st.warning(message)
    elif kind == "error":
        st.error(message)
    else:
        st.info(message)


def _compute_docs_fingerprint(folder_path: str) -> dict:
    """Build a document fingerprint for PDFs (names + mtimes + sizes).

    We avoid hashing full PDFs here to keep this step fast.
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


def _summarize_local_docs(folder_path: str) -> dict:
    """Inspect the selected local docs folder and summarize supported files."""
    summary = {
        "path": folder_path,
        "exists": False,
        "is_dir": False,
        "supported_files": 0,
        "pdf_files": 0,
        "image_files": 0,
        "sample_files": [],
        "error": None,
    }

    try:
        folder = Path(folder_path).expanduser().resolve()
    except Exception as e:
        summary["error"] = str(e)
        return summary

    summary["path"] = str(folder)
    summary["exists"] = folder.exists()
    summary["is_dir"] = folder.is_dir()
    if not summary["exists"] or not summary["is_dir"]:
        return summary

    files = sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in _SUPPORTED_DOC_EXTENSIONS],
        key=lambda p: str(p).lower(),
    )
    pdf_files = [p for p in files if p.suffix.lower() == ".pdf"]
    image_files = [p for p in files if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    summary["supported_files"] = len(files)
    summary["pdf_files"] = len(pdf_files)
    summary["image_files"] = len(image_files)
    summary["sample_files"] = [str(p.relative_to(folder)).replace("\\", "/") for p in files[:3]]
    return summary


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


def _save_uploaded_policy_documents(folder_path: str, uploaded_files: list) -> dict:
    result = {"saved": 0, "sample_names": [], "error": None}

    try:
        folder = Path(folder_path).expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        result["error"] = f"Could not use the selected docs folder: {e}"
        return result

    for uploaded in uploaded_files or []:
        filename = Path(str(getattr(uploaded, "name", ""))).name
        if not filename or Path(filename).suffix.lower() not in _SUPPORTED_DOC_EXTENSIONS:
            continue

        target_path = folder / filename
        target_path.write_bytes(uploaded.getbuffer())
        result["saved"] += 1
        if len(result["sample_names"]) < 3:
            result["sample_names"].append(filename)

    return result


def _run(coro):
    # Streamlit runs sync code; our MCP helpers are async.
    # Bridge the two with asyncio.run() for predictable execution.
    return asyncio.run(coro)


def _format_error_for_ui(e: BaseException) -> str:
    """Make async/network errors readable in the Streamlit UI.

    Some network failures bubble up as an ExceptionGroup with the top-level message
    "unhandled errors in a TaskGroup". We unwrap to the root exception(s).
    """

    def _flatten(exc: BaseException) -> list[BaseException]:
        inner = getattr(exc, "exceptions", None)
        if isinstance(inner, list) and inner:
            out: list[BaseException] = []
            for sub in inner:
                out.extend(_flatten(sub))
            return out
        return [exc]

    flattened = _flatten(e)
    parts: list[str] = []
    for sub in flattened[:3]:
        msg = str(sub).strip()
        parts.append(f"{type(sub).__name__}: {msg}" if msg else type(sub).__name__)
    if len(flattened) > 3:
        parts.append(f"...and {len(flattened) - 3} more")

    base = "; ".join(parts) if parts else str(e)

    # Add a helpful hint for the most common failure mode.
    if "connection" in base.lower() or "connect" in base.lower() or "refused" in base.lower():
        base += f"\n\nCheck: MCP server is running and MCP_SERVER_URL is correct ({mcp_client.MCP_SERVER_URL})."
    return base


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


import re as _re


def _parse_relevance_rating(answer_text: str) -> str | None:
    """Extract RELEVANCE_RATING from the model output (first line).

    Returns one of 'HIGH', 'MEDIUM', 'LOW', 'NONE', or None if not found.
    """
    m = _re.search(r"RELEVANCE_RATING:\s*(HIGH|MEDIUM|LOW|NONE)", answer_text or "", _re.IGNORECASE)
    return m.group(1).upper() if m else None


def _strip_relevance_line(answer_text: str) -> str:
    """Remove the RELEVANCE_RATING line(s) from answer text before display."""
    # Remove the RELEVANCE_RATING line and the explanatory parenthetical that follows.
    cleaned = _re.sub(
        r"RELEVANCE_RATING:\s*(?:HIGH|MEDIUM|LOW|NONE)\s*\n?(?:\(.*?\)\s*\n?)?",
        "",
        answer_text or "",
        flags=_re.IGNORECASE,
    )
    return cleaned.lstrip("\n")


def _evidence_strength_from_validation(
    validation: dict | None,
    answer_text: str = "",
    relevance_rating_override: str | None = None,
) -> tuple[str, str, list[str]]:
    """Compute a Weak / Medium / Strong label.

    PRIMARY signal: the model's self-assessed RELEVANCE_RATING (HIGH/MEDIUM/LOW/NONE)
    embedded in the answer text. This is the most reliable indicator because the
    model reads the source chunks and assesses whether they answer the question.
    Retrieval scores (cosine similarity) are unreliable because vector search
    always returns top-k results even for unrelated queries.

    FALLBACK: If no RELEVANCE_RATING is found, use retrieval stats as a rough
    heuristic (match count + diversity).
    """
    if not isinstance(validation, dict):
        return ("Unknown", "info", ["No validation stats"])

    stats_obj = validation.get("stats")
    stats: dict = stats_obj if isinstance(stats_obj, dict) else {}
    result_count = stats.get("result_count")
    unique_files = stats.get("unique_files")
    unique_doc_types = stats.get("unique_doc_types")
    max_score_raw = stats.get("max_score")
    avg_score_raw = stats.get("avg_score")

    rc = int(result_count) if isinstance(result_count, (int, float)) else None
    uf = int(unique_files) if isinstance(unique_files, (int, float)) else None
    ud = int(unique_doc_types) if isinstance(unique_doc_types, (int, float)) else None
    score = float(max_score_raw) if isinstance(max_score_raw, (int, float)) else None
    avg = float(avg_score_raw) if isinstance(avg_score_raw, (int, float)) else None

    reasons: list[str] = []
    if score is not None:
        reasons.append(f"Top relevance {score:.2f}")
    if avg is not None:
        reasons.append(f"Avg relevance {avg:.2f}")
    if rc is not None:
        reasons.append(f"Matches {rc}")
    if uf is not None:
        reasons.append(f"Files {uf}")
    if ud is not None:
        reasons.append(f"Doc types {ud}")

    # --- Hard gate: no results at all ---
    if rc == 0 or rc is None:
        return ("Weak", "error", reasons + ["No matches"])

    # === PRIMARY: model relevance rating ===
    rating = _parse_relevance_rating(answer_text)
    # If answer was cleared (blocked), use the graph-stored rating.
    if not rating and relevance_rating_override:
        rating = relevance_rating_override.upper() if isinstance(relevance_rating_override, str) else None
    if rating:
        reasons.append(f"Relevance rating: {rating}")
        if rating == "HIGH":
            return ("Strong", "success", reasons)
        elif rating == "MEDIUM":
            return ("Medium", "warning", reasons)
        else:  # LOW or NONE
            return ("Weak", "error", reasons)

    # === FALLBACK: retrieval heuristic (no relevance rating available) ===
    reasons.append("No relevance rating available; using retrieval heuristic")
    if score is not None and score < 0.10:
        return ("Weak", "error", reasons + ["Very low relevance"])
    if uf is not None and uf >= 2 and ud is not None and ud >= 2:
        return ("Strong", "success", reasons)
    if rc is not None and rc >= 3:
        return ("Medium", "warning", reasons)
    return ("Weak", "error", reasons)


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


def _render_semantic_grounding_panel(*, validation: dict | None, expanded: bool = False) -> None:
    if not isinstance(validation, dict):
        return

    semantic = validation.get("semantic_grounding")
    if not isinstance(semantic, dict):
        return

    stats_obj = semantic.get("stats")
    stats: dict = stats_obj if isinstance(stats_obj, dict) else {}
    claims_obj = semantic.get("claims")
    claims: list[dict] = claims_obj if isinstance(claims_obj, list) else []
    issues_obj = semantic.get("issues")
    issues: list[str] = [str(v) for v in issues_obj] if isinstance(issues_obj, list) else []

    with st.expander("Semantic grounding checks", expanded=expanded):
        passed = bool(semantic.get("passed"))
        if passed:
            st.success("All evaluated claims are supported by cited snippets.")
        else:
            st.error("One or more claims are not semantically supported by cited snippets.")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Claims", stats.get("claims_total") or 0)
        with c2:
            st.metric("Supported", stats.get("supported") or 0)
        with c3:
            st.metric("Partial", stats.get("partial") or 0)
        with c4:
            st.metric("Unsupported", stats.get("unsupported") or 0)
        with c5:
            st.metric("Contradictions", stats.get("contradictions") or 0)

        if issues:
            st.caption("Semantic issues")
            st.markdown("- " + "\n- ".join(issues))

        if claims:
            rows: list[dict[str, object]] = []
            for c in claims:
                claim_text = str(c.get("claim") or "").strip()
                reason_text = str(c.get("reason") or "").strip()
                rows.append(
                    {
                        "section": c.get("section"),
                        "verdict": c.get("verdict"),
                        "citations": c.get("citations_count"),
                        "claim": claim_text if len(claim_text) <= 240 else (claim_text[:240].rstrip() + "..."),
                        "reason": reason_text if len(reason_text) <= 220 else (reason_text[:220].rstrip() + "..."),
                    }
                )

            st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_blocked_validation_summary(*, validation: dict | None) -> None:
    if not isinstance(validation, dict):
        return

    reasons_obj = validation.get("reasons")
    reasons: list[str] = [str(v) for v in reasons_obj] if isinstance(reasons_obj, list) else []
    warnings_obj = validation.get("warnings")
    warnings: list[str] = [str(v) for v in warnings_obj] if isinstance(warnings_obj, list) else []
    next_actions_obj = validation.get("next_actions")
    next_actions: list[str] = [str(v) for v in next_actions_obj] if isinstance(next_actions_obj, list) else []

    citation_obj = validation.get("citation_verification")
    citation: dict = citation_obj if isinstance(citation_obj, dict) else {}
    citation_stats_obj = citation.get("stats")
    citation_stats: dict = citation_stats_obj if isinstance(citation_stats_obj, dict) else {}

    semantic_obj = validation.get("semantic_grounding")
    semantic: dict = semantic_obj if isinstance(semantic_obj, dict) else {}
    semantic_stats_obj = semantic.get("stats")
    semantic_stats: dict = semantic_stats_obj if isinstance(semantic_stats_obj, dict) else {}

    st.subheader("Why it was blocked")
    if reasons:
        st.markdown("- " + "\n- ".join(reasons))

    if warnings:
        with st.expander("Warnings", expanded=False):
            st.markdown("- " + "\n- ".join(warnings))

    if citation_stats or semantic_stats:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Citations found", citation_stats.get("citations_found") or 0)
        with c2:
            st.metric("Bad bullets", citation_stats.get("bad_bullets") or 0)
        with c3:
            st.metric("Unsupported claims", semantic_stats.get("unsupported") or 0)
        with c4:
            st.metric("Contradictions", semantic_stats.get("contradictions") or 0)

    if next_actions:
        st.subheader("What to do next")
        st.markdown("- " + "\n- ".join(next_actions))

    with st.expander("Technical validation payload", expanded=False):
        st.json(validation)


def _render_next_steps(*, status_payload: dict | None) -> None:
    steps: list[str] = []

    if not status_payload:
        steps.append(f"Click '{_STATUS_BUTTON_LABEL}' in the sidebar.")
        steps.append(f"If this is first-time setup, run '{_SCAN_BUTTON_LABEL}' and then '{_BUILD_BUTTON_LABEL}'.")
    else:
        q_ok = status_payload.get("status") == "ok"
        collection_exists = bool(status_payload.get("collection_exists"))
        points_count = status_payload.get("points_count")
        openai_ok = status_payload.get("openai_ok")
        openai_configured = bool(status_payload.get("openai_configured"))
        ai_services_ready = bool(openai_configured and (openai_ok is not False))

        if not q_ok:
            steps.append(f"Start required backend services, then click '{_STATUS_BUTTON_LABEL}' again.")
        elif not ai_services_ready:
            steps.append(f"Configure AI service credentials, then click '{_STATUS_BUTTON_LABEL}' again.")
        elif not collection_exists:
            steps.append(f"Build the indexed policy documents by running '{_SCAN_BUTTON_LABEL}' and then '{_BUILD_BUTTON_LABEL}'.")
        elif points_count == 0:
            steps.append(f"The indexed policy documents are empty. Run '{_SCAN_BUTTON_LABEL}' and then '{_BUILD_BUTTON_LABEL}'.")

    steps.append("If answers still look weak, ask a more specific coverage question (peril + coverage + location).")

    st.info("Next steps:\n- " + "\n- ".join(steps))


def _build_setup_flow(
    *,
    status_payload: dict | None,
    server_health_ok: bool,
    consent_checked: bool,
    kb_update_choice: str,
    kb_use_current_confirmed: bool,
    wants_reindex: bool,
    docs_summary: dict,
) -> dict:
    status = status_payload if isinstance(status_payload, dict) else {}
    supported_files = int(docs_summary.get("supported_files") or 0)
    local_docs_available = supported_files > 0
    update_mode = bool(kb_update_choice == _KB_CHOICE_ADD_UPDATE or wants_reindex)

    status_known = bool(status_payload)
    decision_confirmed = bool(update_mode or kb_use_current_confirmed)

    provider_configured = bool(status.get("openai_configured")) if status_known else False
    provider_valid = bool(status.get("openai_ok") is not False) if status_known else False
    ai_services_ready = bool(provider_configured and provider_valid)

    kb_exists = bool(status.get("collection_exists")) if status_known else False
    points_count = status.get("points_count") if status_known else None
    kb_ready = bool(status_known and kb_exists and (points_count is None or (points_count or 0) > 0))

    kb_update_requested = bool(update_mode and wants_reindex)
    kb_action_required = bool((not kb_ready) or kb_update_requested)
    kb_action_skipped = bool(kb_ready and not kb_action_required)

    if not decision_confirmed:
        current_step = "confirm_choice"
        current_step_hint = "Choose how you want to start in First-time setup."
        expected_result = "'Start option confirmed' is marked Done."
    elif not status_known:
        current_step = "refresh_status"
        current_step_hint = f"Click '{_STATUS_BUTTON_LABEL}'."
        expected_result = "Index status appears in the sidebar."
    elif not server_health_ok:
        current_step = "server_health"
        current_step_hint = f"Click '{_HEALTH_BUTTON_LABEL}'."
        expected_result = "'Backend services available' is marked Done."
    elif not ai_services_ready:
        current_step = "ai_credentials"
        current_step_hint = f"Configure AI credentials, then click '{_STATUS_BUTTON_LABEL}'."
        expected_result = "'AI services ready' is marked Done."
    elif kb_action_required and not local_docs_available:
        current_step = "add_documents"
        current_step_hint = (
            "Add policy PDFs or images to the selected docs folder, "
            f"then run '{_SCAN_BUTTON_LABEL}' and '{_BUILD_BUTTON_LABEL}'."
        )
        expected_result = "'Local policy documents ready' is marked Done."
    elif kb_action_required:
        current_step = "ingest_index"
        current_step_hint = f"Click '{_SCAN_BUTTON_LABEL}', then click '{_BUILD_BUTTON_LABEL}'."
        expected_result = "'Indexed policy documents available' is marked Done and the sidebar confirms they are available."
    elif not consent_checked:
        current_step = "consent"
        current_step_hint = "Check the consent box on the right panel."
        expected_result = "'Consent confirmed' is marked Done and 'Review Coverage' is enabled."
    else:
        current_step = "ask"
        current_step_hint = "Click 'Review Coverage'."
        expected_result = "Answer generation runs with current setup."

    rows = [
        {
            "Requirement": "Start option confirmed",
            "Status": "Done" if decision_confirmed else "Pending",
            "Next action": "-" if decision_confirmed else "Choose how you want to start and confirm the selection",
        },
        {
            "Requirement": "Library status checked",
            "Status": "Done" if status_known else "Pending",
            "Next action": "-" if status_known else f"Sidebar -> Library setup -> click '{_STATUS_BUTTON_LABEL}'",
        },
        {
            "Requirement": "Backend services available",
            "Status": "Done" if server_health_ok else "Pending",
            "Next action": "-" if server_health_ok else f"Click '{_HEALTH_BUTTON_LABEL}' (start services first if it fails)",
        },
        {
            "Requirement": "AI services ready",
            "Status": "Done" if ai_services_ready else "Pending",
            "Next action": "-" if ai_services_ready else f"Configure AI credentials, then click '{_STATUS_BUTTON_LABEL}'",
        },
        {
            "Requirement": "Local policy documents ready",
            "Status": (
                "Skipped (using current indexed policy documents)"
                if not update_mode
                else ("Done" if local_docs_available else "Pending")
            ),
            "Next action": (
                "-"
                if not update_mode or local_docs_available
                else "Add PDF, PNG, or JPG policy documents to the selected docs folder"
            ),
        },
        {
            "Requirement": "Indexed policy documents available",
            "Status": "Skipped (up-to-date)" if kb_action_skipped else ("Done" if kb_ready else "Pending"),
            "Next action": (
                "-"
                if (kb_action_skipped or kb_ready)
                else (
                    f"Click '{_SCAN_BUTTON_LABEL}', then click '{_BUILD_BUTTON_LABEL}'"
                    if local_docs_available
                    else f"Add documents first, then click '{_SCAN_BUTTON_LABEL}' and '{_BUILD_BUTTON_LABEL}'"
                )
            ),
        },
        {
            "Requirement": "Consent confirmed",
            "Status": "Done" if consent_checked else "Pending",
            "Next action": "-" if consent_checked else "Check: I confirm I'm using redacted/non-sensitive data",
        },
    ]

    pending_required = sum(1 for r in rows if str(r.get("Status")) == "Pending")

    return {
        "status_known": status_known,
        "decision_confirmed": decision_confirmed,
        "server_health_ok": server_health_ok,
        "ai_services_ready": ai_services_ready,
        "kb_ready": kb_ready,
        "kb_action_required": kb_action_required,
        "kb_action_skipped": kb_action_skipped,
        "local_docs_available": local_docs_available,
        "local_docs_count": supported_files,
        "update_mode": update_mode,
        "current_step": current_step,
        "current_step_hint": current_step_hint,
        "expected_result": expected_result,
        "pending_required": pending_required,
        "rows": rows,
    }


def _render_first_time_readiness(*, status_payload: dict | None, consent_checked: bool) -> None:
    """Render a clear, vendor-agnostic readiness checklist for first-time users."""
    raw_health_payload = st.session_state.get("last_health")
    health_payload: dict = raw_health_payload if isinstance(raw_health_payload, dict) else {}
    services_ok = bool(health_payload.get("status") == "ok")

    if "kb_update_choice" not in st.session_state:
        st.session_state["kb_update_choice"] = _KB_CHOICE_USE_CURRENT
    if "kb_use_current_confirmed" not in st.session_state:
        st.session_state["kb_use_current_confirmed"] = False

    with st.expander("Setup checklist", expanded=True):

        kb_update_choice = st.radio(
            "How would you like to start?",
            options=[_KB_CHOICE_USE_CURRENT, _KB_CHOICE_ADD_UPDATE],
            key="kb_update_choice",
            horizontal=False,
        )

        if kb_update_choice == _KB_CHOICE_ADD_UPDATE:
            st.session_state["wants_reindex"] = True
            st.session_state["wants_reindex_input"] = True
            st.session_state["kb_use_current_confirmed"] = False

        docs_dir_for_readiness = str(st.session_state.get("docs_dir", _default_docs_dir()))
        docs_summary = _summarize_local_docs(docs_dir_for_readiness)

        flow = _build_setup_flow(
            status_payload=status_payload,
            server_health_ok=services_ok,
            consent_checked=bool(consent_checked),
            kb_update_choice=kb_update_choice,
            kb_use_current_confirmed=bool(st.session_state.get("kb_use_current_confirmed", False)),
            wants_reindex=bool(st.session_state.get("wants_reindex", False)),
            docs_summary=docs_summary,
        )

        if kb_update_choice == _KB_CHOICE_USE_CURRENT:
            col_confirm, col_note = st.columns([1, 2])
            with col_confirm:
                if st.button("Confirm current policy documents", key="confirm_use_current_kb"):
                    st.session_state["kb_use_current_confirmed"] = True
                    st.session_state["wants_reindex"] = False
                    st.session_state["wants_reindex_input"] = False
                    st.rerun()
            with col_note:
                if st.session_state.get("kb_use_current_confirmed"):
                    st.success("Using the current indexed policy documents.")
                else:
                    st.info("Click confirm to continue without loading new documents.")

        ready_items: list[str] = []
        pending_items: list[str] = []
        for row in flow["rows"]:
            req = str(row.get("Requirement") or "")
            status_text = str(row.get("Status") or "")
            action_text = str(row.get("Next action") or "")
            if status_text.startswith("Done"):
                ready_items.append(req)
            elif status_text.startswith("Skipped"):
                ready_items.append(f"{req} (up-to-date)")
            else:
                pending_items.append(f"{req}: {action_text}")

        col_ready, col_pending = st.columns(2)
        with col_ready:
            st.markdown("**Ready**")
            if ready_items:
                st.markdown("- " + "\n- ".join(ready_items))
            else:
                st.caption("No completed setup items yet.")
        with col_pending:
            st.markdown("**Needs attention**")
            if pending_items:
                st.markdown("- " + "\n- ".join(pending_items))
            else:
                st.caption("No pending setup items.")

        if flow["update_mode"] and not flow["local_docs_available"]:
            st.warning("No supported documents found. Add PDFs or images to the docs folder.")
        elif flow["kb_action_skipped"] and not flow["update_mode"]:
            st.success("Indexed documents are up to date.")
        elif flow["kb_ready"] and flow["decision_confirmed"] and not flow["update_mode"]:
            st.success("Documents ready.")

        pending_count = int(flow["pending_required"])
        if pending_count:
            st.caption(f"{pending_count} pending item(s). Complete them to enable the query.")
        else:
            st.success("All checks complete — ready to run.")


def _clear_last_run() -> None:
    for k in [
        "last_out",
        "last_inputs",
        "last_trace",
        "last_run_id",
        "last_run_seconds",
        "retry_requested",
        "last_handoff_ticket",
        "handoff_tickets",
    ]:
        st.session_state.pop(k, None)


def _sync_consent_confirmation() -> None:
    st.session_state["consent_confirmed"] = bool(st.session_state.get("consent_confirmed_input", False))
    _clear_last_run()


def _sync_rebuild_choice() -> None:
    st.session_state["wants_reindex"] = bool(st.session_state.get("wants_reindex_input", False))
    _clear_last_run()


def _apply_preset_question() -> None:
    choice = st.session_state.get("preset_choice")
    if choice and choice != "(custom)":
        st.session_state["question_text"] = DEMO_QUESTION_PRESETS.get(choice, "")
    _clear_last_run()


with st.sidebar:
    # ── Product scope and decision boundary ──────────────────────────────────
    st.markdown("#### Coverage Concierge")
    st.caption("Homeowners policy Q&A — answers grounded in indexed documents only. Does not make coverage or claims decisions.")
    st.divider()
    # ─────────────────────────────────────────────────────────────────────────

    last_health = st.session_state.get("last_health")
    index_status_payload = st.session_state.get("index_status")
    last_ingest = st.session_state.get("last_ingest")
    last_index = st.session_state.get("last_index")
    kb_choice_sidebar = str(st.session_state.get("kb_update_choice", _KB_CHOICE_USE_CURRENT))
    kb_use_current_confirmed_sidebar = bool(st.session_state.get("kb_use_current_confirmed", False))
    wants_reindex_sidebar = bool(st.session_state.get("wants_reindex", False))
    consent_done = bool(st.session_state.get("consent_confirmed", False))
    docs_dir_sidebar = str(st.session_state.get("docs_dir") or _default_docs_dir())
    docs_summary_sidebar = _summarize_local_docs(docs_dir_sidebar)
    server_health_ok_sidebar = bool(last_health and last_health.get("status") == "ok")
    preview_flow = _build_setup_flow(
        status_payload=index_status_payload if isinstance(index_status_payload, dict) else None,
        server_health_ok=server_health_ok_sidebar,
        consent_checked=consent_done,
        kb_update_choice=kb_choice_sidebar,
        kb_use_current_confirmed=kb_use_current_confirmed_sidebar,
        wants_reindex=wants_reindex_sidebar,
        docs_summary=docs_summary_sidebar,
    )
    docs_step_active = bool(preview_flow.get("current_step") in {"add_documents", "ingest_index"})
    docs_files_required = bool(preview_flow.get("current_step") == "add_documents")

    with st.expander("Setup checklist", expanded=True):

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

        system_ready = bool(server_ok and qdrant_ok and index_ready and openai_ready)
        demo_ready = bool(preview_flow.get("current_step") == "ask")
        consent_pending_only = bool(preview_flow.get("current_step") == "consent")

        ready_items_sidebar: list[str] = []
        pending_items_sidebar: list[str] = []

        if server_ok:
            ready_items_sidebar.append("Service health")
        else:
            pending_items_sidebar.append(f"Service health -> click '{_HEALTH_BUTTON_LABEL}'")

        if index_ready:
            ready_items_sidebar.append("Indexed policy documents status")
            ready_items_sidebar.append("Load documents (optional)")
            ready_items_sidebar.append("Update indexed policy documents (optional)")
        else:
            pending_items_sidebar.append(f"Indexed policy documents status -> click '{_STATUS_BUTTON_LABEL}'")
            pending_items_sidebar.append(f"Load documents -> click '{_SCAN_BUTTON_LABEL}'")
            pending_items_sidebar.append(f"Build indexed policy documents -> click '{_BUILD_BUTTON_LABEL}'")

        if openai_ready:
            ready_items_sidebar.append("AI services")
        else:
            pending_items_sidebar.append(f"AI services -> configure credentials and click '{_STATUS_BUTTON_LABEL}'")

        if consent_done:
            ready_items_sidebar.append("Consent confirmation")
        else:
            pending_items_sidebar.append("Consent confirmation -> check consent on right panel")

        if preview_flow.get("current_step") == "add_documents":
            pending_items_sidebar.append("Load documents -> add policy files to the selected folder")
        elif preview_flow.get("current_step") == "ingest_index":
            pending_items_sidebar.append(
                f"Update indexed policy documents -> click '{_SCAN_BUTTON_LABEL}' then '{_BUILD_BUTTON_LABEL}'"
            )

        rs_col, pn_col = st.columns(2)
        with rs_col:
            st.markdown("**Ready**")
            if ready_items_sidebar:
                st.markdown("- " + "\n- ".join(ready_items_sidebar))
            else:
                st.caption("No setup items complete yet.")
        with pn_col:
            st.markdown("**Needs attention**")
            if pending_items_sidebar:
                st.markdown("- " + "\n- ".join(pending_items_sidebar))
            else:
                st.caption("No pending setup items.")

        if demo_ready:
            st.success("Ready to run")
        elif consent_pending_only:
            st.warning("Almost ready - confirm you are using redacted/non-sensitive data.")
        else:
            st.warning("Not ready yet")

    # ── Active policy context ─────────────────────────────────────────────────
    with st.expander("Active policy context", expanded=True):
        _ctx_points = (index_status_payload or {}).get("points_count")
        _ctx_collection = (index_status_payload or {}).get("collection") or "home_insurance_docs"
        _ctx_files = (last_ingest or {}).get("files_total") or (last_index or {}).get("files_processed")
        _ctx_chunks = (last_index or {}).get("chunks_indexed") or _ctx_points

        if index_ready:
            st.success("Documents indexed — ready for retrieval")
            _ctx_lines = [
                f"**Line of business:** Homeowners",
                f"**Folder:** `{docs_dir_sidebar}`",
            ]
            if _ctx_files:
                _ctx_lines.append(f"**Documents loaded:** {_ctx_files}")
            if _ctx_chunks:
                _ctx_lines.append(f"**Indexed chunks:** {_ctx_chunks}")
            _ctx_lines.append(f"**Collection:** `{_ctx_collection}`")
            st.markdown("\n\n".join(_ctx_lines))
        else:
            st.warning("No documents indexed — complete setup to enable queries.")
    # ─────────────────────────────────────────────────────────────────────────

    with st.expander("Policy documents + chunking", expanded=docs_step_active):
        docs_dir = st.text_input(
            "Docs folder (absolute path)",
            value=docs_dir_sidebar,
            key="docs_dir",
            help="Use a local folder containing policy PDFs or images.",
        )
        uploaded_policy_documents = st.file_uploader(
            "Upload policy documents",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="uploaded_policy_documents",
            help="Uploaded files are saved to the selected docs folder on this machine.",
        )
        if st.button(
            "Save uploaded files",
            disabled=not uploaded_policy_documents,
            type="primary" if docs_files_required else "secondary",
        ):
            upload_result = _save_uploaded_policy_documents(docs_dir, uploaded_policy_documents)
            if upload_result.get("error"):
                st.error(str(upload_result["error"]))
            elif int(upload_result.get("saved") or 0) == 0:
                st.warning("No supported files were uploaded. Add PDF, PNG, or JPG policy documents.")
            else:
                st.session_state["kb_update_choice"] = _KB_CHOICE_ADD_UPDATE
                st.session_state["kb_use_current_confirmed"] = False
                st.session_state["wants_reindex"] = True
                st.session_state["wants_reindex_input"] = True
                _clear_last_run()
                _set_setup_notice(
                    "success",
                    f"Saved {upload_result['saved']} file(s). Next: click '{_SCAN_BUTTON_LABEL}'.",
                )
                st.rerun()

        docs_summary = _summarize_local_docs(docs_dir)
        if docs_summary.get("error"):
            st.error(f"Could not read the selected docs folder: {docs_summary['error']}")
        elif not docs_summary.get("exists"):
            if docs_files_required:
                st.warning("Upload files below, or create this folder and place policy documents here to continue.")
        elif not docs_summary.get("is_dir"):
            st.error("The selected path is not a folder. Choose a folder that contains policy documents.")
        elif int(docs_summary.get("supported_files") or 0) == 0:
            if docs_files_required:
                st.warning("Upload policy documents below or copy PDF, PNG, or JPG files into this folder to continue.")
        elif docs_step_active:
            st.success(f"{docs_summary['supported_files']} supported document(s) found and ready for scanning.")
        else:
            st.caption(f"{docs_summary['supported_files']} supported document(s) detected in this folder.")
        if int(docs_summary.get("supported_files") or 0) > 0:
            sample_files = docs_summary.get("sample_files") or []
            if sample_files:
                st.caption("Examples: " + ", ".join(str(v) for v in sample_files))
        max_pages = st.number_input("Max PDF pages (per file)", min_value=1, max_value=200, value=25, step=1)
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1200, step=50)
        overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=150, step=25)
        if int(overlap) >= int(chunk_size):
            st.error(
                f"Chunk overlap ({int(overlap)}) must be less than Chunk size ({int(chunk_size)}). "
                "Reduce the overlap or increase the chunk size before scanning."
            )
            overlap = max(0, int(chunk_size) - 1)

    st.divider()
    st.subheader("Policy documents setup")
    _render_setup_notice()

    kb_choice = str(st.session_state.get("kb_update_choice", _KB_CHOICE_USE_CURRENT))
    kb_use_current_confirmed = bool(st.session_state.get("kb_use_current_confirmed", False))
    consent_checked = bool(st.session_state.get("consent_confirmed", False))
    status_payload = st.session_state.get("index_status")
    raw_last_health = st.session_state.get("last_health")
    last_health_payload: dict = raw_last_health if isinstance(raw_last_health, dict) else {}
    server_health_ok = bool(last_health_payload.get("status") == "ok")
    update_mode = bool(kb_choice == _KB_CHOICE_ADD_UPDATE or st.session_state.get("wants_reindex", False))

    button_flow = _build_setup_flow(
        status_payload=status_payload if isinstance(status_payload, dict) else None,
        server_health_ok=server_health_ok,
        consent_checked=consent_checked,
        kb_update_choice=kb_choice,
        kb_use_current_confirmed=kb_use_current_confirmed,
        wants_reindex=bool(st.session_state.get("wants_reindex", False)),
        docs_summary=docs_summary,
    )

    status_col_a, status_col_b = st.columns([1, 1])
    with status_col_a:
        refresh_status = st.button(
            _STATUS_BUTTON_LABEL,
            disabled=not kb_use_current_confirmed and kb_choice != _KB_CHOICE_ADD_UPDATE,
            type="primary" if button_flow.get("current_step") == "refresh_status" else "secondary",
        )
    with status_col_b:
        show_status = st.checkbox("Show status details", value=False)

    if not kb_use_current_confirmed and not update_mode:
        st.caption("In the setup checklist above, confirm your start option to continue.")

    if refresh_status:
        try:
            refreshed_status = _run(mcp_client.index_status())
            st.session_state["index_status"] = refreshed_status
            q_ok = refreshed_status.get("status") == "ok"
            collection_exists = bool(refreshed_status.get("collection_exists"))
            points_count = refreshed_status.get("points_count")
            if q_ok and collection_exists and (points_count is None or (points_count or 0) > 0):
                if update_mode:
                    _set_setup_notice("info", "Status updated. Next: add files below, then scan and rebuild the indexed policy documents.")
                else:
                    _set_setup_notice("success", "Status updated.")
            elif q_ok and collection_exists:
                _set_setup_notice("warning", "Status updated. Indexed policy documents exist, but they are empty.")
            elif q_ok:
                _set_setup_notice("info", "Status updated. Indexed policy documents are not available yet.")
            else:
                _set_setup_notice("error", "Status updated, but the document services are unreachable.")
            st.rerun()
        except Exception as e:
            st.session_state["index_status"] = {"status": "degraded", "error": str(e)}
            _set_setup_notice("error", f"Status check failed: {e}")
            st.rerun()

    status_payload = st.session_state.get("index_status")
    show_status_summary = bool(
        status_payload
        and not update_mode
        and button_flow.get("current_step") not in {"consent", "ask"}
    )
    if status_payload and show_status_summary:
        q_ok = status_payload.get("status") == "ok"
        collection_exists = bool(status_payload.get("collection_exists"))
        points_count = status_payload.get("points_count")
        local_docs_count = int(docs_summary.get("supported_files") or 0)

        if q_ok and collection_exists and (points_count is None or points_count > 0):
            st.success("Indexed policy documents available")
            st.caption("Indexed policy documents are ready.")
        elif q_ok and collection_exists:
            st.warning("Indexed policy documents exist but are empty")
            if local_docs_count == 0:
                st.info("No local policy documents were found yet. Add files first, then scan and build the indexed policy documents.")
            else:
                st.info(f"{local_docs_count} local document(s) are ready. Scan them and build the indexed policy documents.")
        elif q_ok:
            st.warning("No indexed policy documents are available yet")
            if local_docs_count == 0:
                st.info("No local policy documents were found in the selected folder yet.")
            else:
                st.info(f"{local_docs_count} local document(s) are ready. Scan them and build the indexed policy documents.")
        else:
            st.error("Document services are unreachable")

        if show_status:
            st.json(status_payload)

    # Indexing intent:
    # - If the collection does NOT exist yet: indexing is required.
    # - If the collection already exists: only index again if the user says they added/changed docs.
    status_payload = st.session_state.get("index_status")
    status_known = bool(status_payload)
    q_ok = bool(status_payload and status_payload.get("status") == "ok")
    ai_ready = bool(
        status_payload
        and status_payload.get("openai_configured")
        and (status_payload.get("openai_ok") is not False)
    )
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
            st.warning("The selected documents look different from the last build. Rebuild the indexed policy documents to include those changes.")
            if "wants_reindex" not in st.session_state:
                st.session_state["wants_reindex"] = True
        elif index_has_points:
            st.caption("Indexed policy documents already exist. Rebuild them only if the selected documents changed.")

        st.session_state["wants_reindex_input"] = bool(st.session_state.get("wants_reindex", False))
        wants_reindex = st.checkbox(
            "I changed my documents - rebuild indexed policy documents",
            key="wants_reindex_input",
            on_change=_sync_rebuild_choice,
            help="Use this only when you changed the documents in the selected folder.",
        )
        st.session_state["wants_reindex"] = bool(wants_reindex)

        if update_mode and index_has_points:
            if st.button(
                "Cancel document update and use current indexed policy documents",
                type="secondary",
            ):
                st.session_state["kb_update_choice"] = _KB_CHOICE_USE_CURRENT
                st.session_state["kb_use_current_confirmed"] = True
                st.session_state["wants_reindex"] = False
                _clear_last_run()
                _set_setup_notice(
                    "info",
                    "Document update canceled. The app is using the current indexed policy documents.",
                )
                st.rerun()

    flow = _build_setup_flow(
        status_payload=status_payload,
        server_health_ok=server_health_ok,
        consent_checked=consent_checked,
        kb_update_choice=kb_choice,
        kb_use_current_confirmed=kb_use_current_confirmed,
        wants_reindex=bool(wants_reindex),
        docs_summary=docs_summary,
    )
    st.session_state["setup_current_step"] = str(flow.get("current_step") or "")
    st.session_state["setup_current_step_hint"] = str(flow.get("current_step_hint") or "")
    st.session_state["setup_expected_result"] = str(flow.get("expected_result") or "")

    if flow["current_step"] == "ask":
        st.success("Setup complete.")

    can_run_health = bool(flow["decision_confirmed"] and flow["status_known"] and flow["current_step"] not in {"confirm_choice", "refresh_status"})
    can_ingest = bool(flow["current_step"] == "ingest_index")
    can_index = bool(
        flow["current_step"] == "ingest_index"
        and status_known
        and q_ok
        and server_health_ok
        and ai_ready
        and (not collection_exists or wants_reindex)
    )

    col_a, col_b = st.columns(2)
    with col_a:
        do_ingest = st.button(
            _SCAN_BUTTON_LABEL,
            disabled=not can_ingest,
            type="primary" if flow["current_step"] == "ingest_index" and not can_index else "secondary",
        )
    with col_b:
        do_index = st.button(
            _BUILD_BUTTON_LABEL,
            disabled=not can_index,
            type="primary" if flow["current_step"] == "ingest_index" and can_index else "secondary",
            help=(
                f"First click '{_STATUS_BUTTON_LABEL}'. Build the searchable policy documents when local documents are ready. "
                "If indexed policy documents already exist, enable rebuild only after changing documents."
            ),
        )

    # Basic health check before running.
    if st.button(
        _HEALTH_BUTTON_LABEL,
        disabled=not can_run_health,
        type="primary" if flow["current_step"] == "server_health" else "secondary",
        help="Verify the MCP server and Qdrant are reachable before running a query.",
    ):
        try:
            payload = _run(mcp_client.health())
            st.session_state["last_health"] = payload
            _set_setup_notice("success", "Service health confirmed. Follow the recommended next step below.")
            st.rerun()
        except Exception as e:
            st.error(f"Health check failed: {e}")

    if last_health_payload:
        with st.expander("Health details", expanded=False):
            st.json(last_health_payload)

    with st.expander("Self-check", expanded=False):

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
            st.write(f"3) Indexed policy documents available: {'Yes' if idx_ready else 'No'}")

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
            if isinstance(job, dict) and job.get("status") == "error":
                raise RuntimeError(job.get("error") or "Failed to start ingest")
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

                    try:
                        st.session_state["index_status"] = _run(mcp_client.index_status())
                    except Exception:
                        pass
                    break

                time.sleep(0.35)
        except Exception as e:
            st.error(f"Ingest failed: {_format_error_for_ui(e)}")

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
            if isinstance(job, dict) and job.get("status") == "error":
                raise RuntimeError(job.get("error") or "Failed to start indexing")
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

                    # Refresh the sidebar state immediately after indexing.
                    try:
                        st.session_state["index_status"] = _run(mcp_client.index_status())
                    except Exception:
                        pass

                    break

                time.sleep(0.35)
        except Exception as e:
            st.error(f"Index failed: {_format_error_for_ui(e)}")

# ---------- CONVERSATION HISTORY ----------
_prior_turns = list(st.session_state.get("conversation_history") or [])
if _prior_turns:
    with st.expander(f"Conversation history ({len(_prior_turns)} prior turn{'s' if len(_prior_turns) != 1 else ''})", expanded=False):
        for _i, _turn in enumerate(_prior_turns, 1):
            st.markdown(f"**Q{_i}:** {_turn.get('question', '')}")
            st.markdown(f"**A{_i}:** {_turn.get('answer', '')}")
            if _i < len(_prior_turns):
                st.divider()
        if st.button("Clear conversation history", key="clear_history"):
            st.session_state["conversation_history"] = []
            st.rerun()

# ---------- INPUTS ----------
left, right = st.columns([2, 1])

DEMO_QUESTION_PRESETS = {
    # --- Strong evidence (direct homeowners policy questions) ---
    "Water backup / sump pump": "Does my policy cover water backup or drainage system issues?",
    "Pipe burst + resulting damage": "If a pipe bursts inside the home, is the resulting water damage covered?",
    "Mold after water loss": "Is mold remediation covered after a covered water loss, and what limits apply?",
    "Roof leak from storm": "If wind or hail damages my roof and rain enters, what is covered and what is excluded?",
    "Loss of use": "If my home is uninhabitable after a covered loss, does the policy cover temporary living expenses?",
    # Cross-policy scenario for domain-boundary testing.
    "Cross-policy scenario (auto coverage context)": "Does my policy cover damage to my car parked in the driveway during a hailstorm, and what is the collision deductible?",
    # Out-of-scope scenario for domain-boundary testing.
    "Out-of-scope scenario (non-insurance topic)": "What is the best recipe for sourdough bread and how long should I let the dough rise?",
}


with left:
    if "question_text" not in st.session_state:
        st.session_state["question_text"] = DEMO_QUESTION_PRESETS["Water backup / sump pump"]

    st.selectbox(
        "Question presets",
        options=["(custom)"] + list(DEMO_QUESTION_PRESETS.keys()),
        index=0,
        key="preset_choice",
        on_change=_apply_preset_question,
    )

    question = st.text_area(
        "Ask a homeowners insurance coverage question",
        key="question_text",
        height=90,
        on_change=_clear_last_run,
    )

with right:
    setup_current_step = str(st.session_state.get("setup_current_step") or "")

    if "state" not in st.session_state:
        st.session_state["state"] = "IL"
    if "require_citations" not in st.session_state:
        st.session_state["require_citations"] = True

    state = st.selectbox(
        "Jurisdiction / State",
        ["IL", "CA", "NY", "TX", "FL"],
        index=0,
        key="state",
        on_change=_clear_last_run,
    )
    require_citations = st.checkbox(
        "Require citations (block answers without sources)",
        value=True,
        key="require_citations",
        on_change=_clear_last_run,
    )
    if not require_citations:
        st.warning(
            "Citations are disabled. In this mode the model may draw on general knowledge "
            "in addition to indexed policy documents. Enable citations to enforce strict "
            "document-grounded answers."
        )
    elif setup_current_step == "ask":
        st.success("Ready.")
    st.session_state["consent_confirmed_input"] = bool(st.session_state.get("consent_confirmed", False))
    consent = st.checkbox(
        "I confirm I'm using redacted/non-sensitive data.",
        key="consent_confirmed_input",
        on_change=_sync_consent_confirmation,
    )
    st.session_state["consent_confirmed"] = bool(consent)

    with st.expander("Impact & metrics", expanded=False):

        if "impact_metrics" not in st.session_state:
            st.session_state["impact_metrics"] = {
                "baseline_handle_time_minutes": 12.0,
                "target_handle_time_minutes": 7.0,
                "first_contact_resolution_uplift_pct": 12.0,
                "deflection_rate_pct": 15.0,
                "escalation_reduction_pct": 10.0,
                "notes": "Assumptions: common coverage questions; policy packet indexed; used for initial guidance with citations.",
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
    _render_first_time_readiness(status_payload=status_payload, consent_checked=bool(consent))

ask = st.button("Review Coverage", type="primary", disabled=not (consent and demo_ready))

def _run_graph_once(*, question: str, state: str, require_citations: bool, conversation_history: list | None = None) -> dict:
    graph = build_graph()
    graph_input: GraphState = {
        "question": question,
        "state": state,
        "require_citations": require_citations,
        "conversation_history": list(conversation_history or []),
    }
    return graph.invoke(graph_input)


def _render_run(*, out: dict, question: str, state: str, require_citations: bool, status_payload: dict | None) -> None:
    answer = out.get("answer", "")
    sources = out.get("sources", "")
    raw_results = out.get("raw_results", [])
    trace = out.get("trace", [])
    run_id = out.get("run_id")
    blocked = bool(out.get("blocked", False))
    validation = out.get("validation", None)
    retrieval_plan = out.get("retrieval_plan")
    precedence_check = out.get("precedence_check")

    st.session_state["last_trace"] = trace
    st.session_state["last_run_id"] = run_id

    strength_label, strength_kind, strength_reasons = _evidence_strength_from_validation(
        validation if isinstance(validation, dict) else None,
        answer_text=answer or "",
        relevance_rating_override=out.get("relevance_rating"),
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
            _clear_last_run()
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
        relevance_rating = out.get("relevance_rating")
        if relevance_rating in ("LOW", "NONE"):
            st.error(
                "Blocked: The question appears to be outside the scope of the indexed homeowners insurance documents. "
                "The system only answers questions grounded in your policy documents."
            )
        else:
            st.error(
                "Blocked: Not enough policy evidence was found to answer safely. "
                "Try ingest/indexing more docs or rephrasing the question."
            )
        if validation:
            _render_semantic_grounding_panel(validation=validation if isinstance(validation, dict) else None, expanded=True)
            _render_blocked_validation_summary(validation=validation if isinstance(validation, dict) else None)

        st.subheader("What was retrieved")
        st.code(sources or "(no sources found)")
        _render_next_steps(status_payload=status_payload)
        return

    st.markdown("### Answer")
    st.write(_strip_relevance_line(answer))

    _render_semantic_grounding_panel(validation=validation if isinstance(validation, dict) else None, expanded=False)

    endorsement_signals = out.get("endorsement_signals")
    if isinstance(endorsement_signals, dict) and endorsement_signals.get("present"):
        with st.expander("Endorsement override check", expanded=False):
            st.json(endorsement_signals)

    with st.expander("Sources used", expanded=False):
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
                    snippet = (r.get("snippet") or "").strip()
                    st.write(snippet if show_unredacted else _redact_display_text(snippet) or "(empty)")
                    st.divider()
        else:
            st.code(sources)

    with st.expander("Handoff for human review", expanded=False):
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

    with st.expander("Audit log", expanded=False):

        sanitized_matches = _sanitize_retrieved_matches_for_audit(raw_results)

        doc_type_counts: dict[str, int] = {}
        for r in raw_results or []:
            dt = r.get("doc_type")
            if isinstance(dt, str) and dt:
                doc_type_counts[dt] = doc_type_counts.get(dt, 0) + 1

        retrieval_topic_counts: dict[str, int] = {}
        for r in raw_results or []:
            topics = r.get("retrieval_topics")
            if isinstance(topics, list):
                for t in topics:
                    if isinstance(t, str) and t:
                        retrieval_topic_counts[t] = retrieval_topic_counts.get(t, 0) + 1
        trace_bundle = {
            "run_id": run_id,
            "inputs": {"question": question, "state": state, "require_citations": require_citations},
            "validation": validation,
            "retrieval_plan": retrieval_plan,
            "precedence_check": precedence_check,
            "evidence_summary": {
                "doc_type_counts": doc_type_counts,
                "retrieval_topic_counts": retrieval_topic_counts,
                "retrieved_matches": len(raw_results or []),
            },
            "trace": trace,
            "retrieved_matches": sanitized_matches,
        }

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Steps (trace)**")
            st.json(trace)
        with col2:
            st.markdown("**Inputs + evidence**")
            st.json(
                {
                    "inputs": trace_bundle["inputs"],
                    "validation": validation,
                    "retrieval_plan": retrieval_plan,
                    "precedence_check": precedence_check,
                    "evidence_summary": trace_bundle["evidence_summary"],
                    "retrieved_matches": sanitized_matches,
                }
            )

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
            conversation_history=list(st.session_state.get("conversation_history") or []),
        )
    st.session_state["last_run_seconds"] = time.time() - run_started

    # Append to conversation history after a successful (non-blocked) run.
    _result_out = st.session_state.get("last_out") or {}
    if not _result_out.get("blocked") and _result_out.get("answer"):
        _hist = list(st.session_state.get("conversation_history") or [])
        _hist.append({"question": question, "answer": _strip_relevance_line(_result_out["answer"])})
        st.session_state["conversation_history"] = _hist[-5:]  # keep last 5 turns

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
        st.caption(f"Run '{_SCAN_BUTTON_LABEL}' or '{_BUILD_BUTTON_LABEL}' from the sidebar.")
    if ingest:
        st.markdown("**Last ingest**")
        st.json(ingest)
    if index:
        st.markdown("**Last index**")
        st.json(index)
