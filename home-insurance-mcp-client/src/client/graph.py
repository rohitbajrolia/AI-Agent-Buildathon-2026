import os
import time
import hashlib
import uuid
import re
from datetime import datetime, timezone
from typing import Any, NotRequired, Required, TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END

from client.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from client.mcp_client import retrieve_clauses

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Keep requests bounded so a transient network stall doesn't hang the run.
client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0, max_retries=0)


class GraphState(TypedDict):
    question: Required[str]
    sources: NotRequired[str]
    answer: NotRequired[str]
    raw_results: NotRequired[list[dict[str, Any]]]
    state: NotRequired[str]
    require_citations: NotRequired[bool]
    run_id: NotRequired[str]
    trace: NotRequired[list[dict[str, Any]]]
    blocked: NotRequired[bool]
    validation: NotRequired[dict[str, Any]]
    answer_retry_count: NotRequired[int]
    endorsement_signals: NotRequired[dict[str, Any]]
    retrieval_plan: NotRequired[list[dict[str, Any]]]
    precedence_check: NotRequired[dict[str, Any]]


_ENDORSEMENT_MODIFY_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bthis endorsement\b", flags=re.IGNORECASE),
    re.compile(r"\bchanges? the policy\b", flags=re.IGNORECASE),
    re.compile(r"\bamend(s|ed)?\b", flags=re.IGNORECASE),
    re.compile(r"\bmodif(y|ies|ied)\b", flags=re.IGNORECASE),
    re.compile(r"\breplace(s|d)?\b", flags=re.IGNORECASE),
    re.compile(r"\bdeleted and replaced\b", flags=re.IGNORECASE),
    re.compile(r"\bsupersede(s|d)?\b", flags=re.IGNORECASE),
]


def _detect_endorsement_conflicts(raw_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect simple, defensible 'endorsement may override base policy' signals.

    We do NOT try to fully interpret endorsements (that would be guessing).
    We only surface: (1) endorsements were retrieved, (2) they include language that
    suggests modification, and (3) we should prefer endorsement language when conflicts exist.
    """
    results = raw_results or []

    endorsements: list[dict[str, Any]] = []
    base_docs: list[dict[str, Any]] = []
    for r in results:
        doc_type = (r.get("doc_type") or "").strip().lower()
        if doc_type == "endorsement":
            endorsements.append(r)
        else:
            base_docs.append(r)

    if not endorsements:
        return {"present": False}

    signals: list[dict[str, Any]] = []
    for r in endorsements:
        snippet = (r.get("snippet") or "").strip()
        matched = []
        for p in _ENDORSEMENT_MODIFY_PATTERNS:
            if p.search(snippet):
                matched.append(p.pattern)
        if matched:
            signals.append(
                {
                    "file_name": r.get("file_name"),
                    "page_number": r.get("page_number"),
                    "chunk_index": r.get("chunk_index"),
                    "matched": matched,
                }
            )

    return {
        "present": True,
        "endorsement_matches": len(endorsements),
        "base_matches": len(base_docs),
        "modification_language_found": bool(signals),
        "signals": signals,
        "guidance": (
            "Endorsements can modify or override base policy language. "
            "If endorsement text conflicts with the booklet/declarations, treat the endorsement as controlling "
            "and ask a human to confirm applicability (form, effective date, property/state)."
        ),
    }


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _redact_text(text: str, *, max_preview_chars: int = 180) -> dict[str, Any]:
    """Create a trace-safe summary of user-provided text.

    The audit log should be useful without turning into a PII dump.
    We keep a short preview (lightly redacted), plus length and a hash.
    """
    cleaned = (text or "").strip()

    # Light redaction heuristics: redact long digit runs and emails.
    redacted = []
    i = 0
    while i < len(cleaned):
        ch = cleaned[i]

        if ch.isdigit():
            j = i
            while j < len(cleaned) and cleaned[j].isdigit():
                j += 1
            if (j - i) >= 6:
                redacted.append("[REDACTED_DIGITS]")
            else:
                redacted.append(cleaned[i:j])
            i = j
            continue

        redacted.append(ch)
        i += 1

    redacted_text = "".join(redacted)

    # Simple email masking for audit preview.
    if "@" in redacted_text and "." in redacted_text:
        parts = redacted_text.split()
        masked_parts = []
        for p in parts:
            if "@" in p and "." in p:
                masked_parts.append("[REDACTED_EMAIL]")
            else:
                masked_parts.append(p)
        redacted_text = " ".join(masked_parts)

    preview = redacted_text[:max_preview_chars]
    digest = hashlib.sha256(cleaned.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return {"preview": preview, "length": len(cleaned), "sha256_12": digest}


def _append_trace(state: GraphState, event: dict[str, Any]) -> None:
    trace = state.get("trace")
    if trace is None:
        trace = []
        state["trace"] = trace
    trace.append(event)


def _stable_result_key(r: dict[str, Any]) -> tuple[str, str, int, int | None] | None:
    file_name = r.get("file_name")
    doc_type = r.get("doc_type")
    chunk_index = r.get("chunk_index")
    page_number = r.get("page_number")

    if not isinstance(file_name, str) or not file_name.strip():
        return None
    if not isinstance(doc_type, str) or not doc_type.strip():
        return None
    if not isinstance(chunk_index, int):
        return None
    if page_number is not None and not isinstance(page_number, int):
        page_number = None

    return (file_name.strip(), doc_type.strip(), int(chunk_index), page_number)


def _build_retrieval_plan(*, question: str, state_code: str) -> list[dict[str, str]]:
    q = (question or "").strip()
    if not q:
        return []

    plan: list[dict[str, str]] = []
    plan.append({"topic": "Coverage grant", "query": q})
    plan.append({"topic": "Definitions", "query": f"Definitions that matter for: {q}"})
    plan.append({"topic": "Exclusions / limits", "query": f"Exclusions, limitations, and not covered language for: {q}"})
    plan.append(
        {
            "topic": "Endorsements / changes",
            "query": "Endorsement language that changes, replaces, modifies, or supersedes base policy terms for: " + q,
        }
    )

    q_lower = q.lower()
    if any(w in q_lower for w in ["deductible", "conditions", "duties", "notice", "proof of loss"]):
        plan.append({"topic": "Conditions / duties", "query": f"Conditions, duties after loss, and notice requirements related to: {q}"})

    if any(w in q_lower for w in ["state", "jurisdiction"]):
        plan.append({"topic": "State-specific", "query": f"State-specific clauses for {state_code}: {q}"})

    return plan[:5]


def plan_retrievals_node(state: GraphState) -> GraphState:
    if not state.get("run_id"):
        state["run_id"] = uuid.uuid4().hex

    question = state["question"]
    state_code = state.get("state", "IL")

    plan_raw = _build_retrieval_plan(question=question, state_code=state_code)
    state["retrieval_plan"] = [
        {"topic": p.get("topic", ""), "query_redacted": _redact_text(p.get("query", "")).get("preview", "")}
        for p in plan_raw
    ]

    _append_trace(
        state,
        {
            "ts": _now_iso_utc(),
            "step": "plan",
            "question": _redact_text(question),
            "jurisdiction": state_code,
            "planned": state.get("retrieval_plan", []),
        },
    )
    return state


def multi_retrieve_node(state: GraphState) -> GraphState:
    question = state["question"]
    state_code = state.get("state", "IL")
    plan_raw = _build_retrieval_plan(question=question, state_code=state_code)

    if not plan_raw:
        state["sources"] = "(no sources found)"
        state["raw_results"] = []
        return state

    import asyncio

    t0 = time.perf_counter()
    errors: list[str] = []
    combined: list[dict[str, Any]] = []

    for p in plan_raw:
        topic = p.get("topic") or "(untitled)"
        query = p.get("query") or question
        try:
            data = asyncio.run(retrieve_clauses(query, top_k=4))
            results = data.get("results", []) if isinstance(data, dict) else []
        except Exception as e:
            results = []
            errors.append(f"{topic}: {e}")

        for r in results or []:
            if not isinstance(r, dict):
                continue
            r2 = dict(r)
            r2["retrieval_topic"] = topic
            combined.append(r2)

    deduped: dict[tuple[str, str, int, int | None], dict[str, Any]] = {}
    for r in combined:
        key = _stable_result_key(r)
        if key is None:
            continue

        existing = deduped.get(key)
        if existing is None:
            r3 = dict(r)
            r3["retrieval_topics"] = [r.get("retrieval_topic")] if r.get("retrieval_topic") else []
            deduped[key] = r3
            continue

        topics = list(existing.get("retrieval_topics") or [])
        t = r.get("retrieval_topic")
        if isinstance(t, str) and t and t not in topics:
            topics.append(t)
            existing["retrieval_topics"] = topics

        old_score = existing.get("score")
        new_score = r.get("score")
        if isinstance(new_score, (int, float)) and (not isinstance(old_score, (int, float)) or float(new_score) > float(old_score)):
            for k in ["score", "snippet", "text"]:
                if k in r:
                    existing[k] = r.get(k)

    results_final = list(deduped.values())
    results_final.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    results_final = results_final[:12]

    dt_ms = (time.perf_counter() - t0) * 1000.0

    lines: list[str] = []
    for r in results_final:
        page_number = r.get("page_number")
        if page_number is not None:
            cite = f'[{r.get("file_name")} | {r.get("doc_type")} | p. {page_number} | chunk {r.get("chunk_index")}]'
        else:
            cite = f'[{r.get("file_name")} | {r.get("doc_type")} | chunk {r.get("chunk_index")}]'
        snippet = (r.get("snippet") or "").replace("\n", " ").strip()
        if len(snippet) > 360:
            snippet = snippet[:360].rstrip() + "..."
        lines.append(f"- {cite} {snippet}")

    state["sources"] = "\n".join(lines) if lines else "(no sources found)"
    state["raw_results"] = results_final

    by_topic: dict[str, int] = {}
    for r in results_final:
        topics = r.get("retrieval_topics")
        if not isinstance(topics, list):
            continue
        for t in topics:
            if isinstance(t, str) and t:
                by_topic[t] = by_topic.get(t, 0) + 1

    top_scores: list[float] = []
    for r in results_final[:3]:
        s = r.get("score")
        if isinstance(s, (int, float)):
            top_scores.append(float(s))

    _append_trace(
        state,
        {
            "ts": _now_iso_utc(),
            "step": "retrieve",
            "duration_ms": round(dt_ms, 1),
            "question": _redact_text(question),
            "planned_topics": [p.get("topic") for p in plan_raw],
            "per_topic_matches": by_topic,
            "result_count": len(results_final),
            "top_scores": top_scores,
            "errors": errors[:5],
        },
    )

    if errors and not results_final:
        state["blocked"] = True
        state["validation"] = {
            "passed": False,
            "reasons": ["Retrieval failed; cannot proceed safely."] + errors[:3],
            "warnings": [],
            "next_actions": [
                "Confirm the tool server and Qdrant are running, then try again.",
                "Use Refresh Index Status in the sidebar to confirm readiness.",
            ],
            "stats": {"result_count": 0, "unique_files": 0, "unique_doc_types": 0, "max_score": None},
        }

    return state


def precedence_check_node(state: GraphState) -> GraphState:
    results = state.get("raw_results", []) or []
    endorsement_signals = _detect_endorsement_conflicts(results)
    state["endorsement_signals"] = endorsement_signals

    override_risk = bool(
        endorsement_signals.get("present")
        and endorsement_signals.get("modification_language_found")
        and (endorsement_signals.get("endorsement_matches") or 0) > 0
    )

    state["precedence_check"] = {
        "endorsements_retrieved": bool(endorsement_signals.get("present")),
        "override_risk": override_risk,
        "note": (
            "Endorsements can change base policy wording. If there is a conflict, treat the endorsement as controlling and verify form, effective date, and state."
            if endorsement_signals.get("present")
            else "No endorsements were retrieved in the top matches."
        ),
    }

    _append_trace(
        state,
        {
            "ts": _now_iso_utc(),
            "step": "precedence_check",
            "endorsements_retrieved": bool(endorsement_signals.get("present")),
            "override_risk": override_risk,
        },
    )
    return state


def retrieve_node(state: GraphState) -> GraphState:
    """
    Backward-compatible wrapper for retrieval.

    Runs:
    - plan_retrievals_node
    - multi_retrieve_node
    - precedence_check_node
    """
    state = plan_retrievals_node(state)
    state = multi_retrieve_node(state)
    state = precedence_check_node(state)
    return state


def answer_node(state: GraphState) -> GraphState:
    """
    Node 2: Generate an answer using ONLY the retrieved sources.
    """
    question = state["question"]
    state_code = state.get("state", "IL")  # State code for jurisdiction.
    sources = state.get("sources") or "(no sources found)"
    require_citations = state.get("require_citations", True)

    if not OPENAI_API_KEY:
        state["blocked"] = True
        state["answer"] = ""
        state["validation"] = {
            "passed": False,
            "reasons": ["OPENAI_API_KEY is not configured; cannot generate an answer."],
            "warnings": [],
            "next_actions": ["Set OPENAI_API_KEY in the client .env, then restart the app."],
            "stats": state.get("validation", {}).get("stats") if isinstance(state.get("validation"), dict) else {},
        }
        _append_trace(
            state,
            {
                "ts": _now_iso_utc(),
                "step": "answer",
                "engine": "text_generation",
                "duration_ms": 0.0,
                "error": "missing_openai_key",
            },
        )
        return state

    prompt = USER_TEMPLATE.format(
        question=question,
        state_code=state_code,
        sources=sources,
        require_citations=require_citations,
    )

    raw_results = state.get("raw_results", []) or []
    endorsement_signals = state.get("endorsement_signals")
    if not isinstance(endorsement_signals, dict):
        endorsement_signals = _detect_endorsement_conflicts(raw_results)
        state["endorsement_signals"] = endorsement_signals

    precedence_obj = state.get("precedence_check")
    precedence: dict[str, Any] = precedence_obj if isinstance(precedence_obj, dict) else {}

    if endorsement_signals.get("present"):
        prompt += (
            "\n\nEndorsement precedence note:\n"
            "- Endorsements can change base policy language.\n"
            "- If endorsement text conflicts with booklet or declarations, call out the conflict and treat the endorsement as controlling.\n"
            "- If applicability is unclear (form, effective date, state), list it under 'What to verify'.\n"
        )
        if precedence.get("override_risk"):
            prompt += "- Override risk is flagged based on the retrieved endorsement language.\n"

    # Provide an explicit allow-list of citation tags.
    # This reduces made-up chunk numbers and makes the verification step more reliable.
    allowed_tags: list[str] = []
    for r in raw_results:
        file_name = r.get("file_name")
        doc_type = r.get("doc_type")
        chunk_index = r.get("chunk_index")
        page_number = r.get("page_number")

        if not isinstance(file_name, str) or not file_name.strip():
            continue
        if not isinstance(doc_type, str) or not doc_type.strip():
            continue
        if not isinstance(chunk_index, int):
            continue

        if isinstance(page_number, int):
            allowed_tags.append(f"[{file_name.strip()} | {doc_type.strip()} | p. {page_number} | chunk {chunk_index}]")
        else:
            allowed_tags.append(f"[{file_name.strip()} | {doc_type.strip()} | chunk {chunk_index}]")

    if require_citations and allowed_tags:
        allowed_block = "\n".join(f"- {t}" for t in sorted(set(allowed_tags)))
        prompt += (
            "\n\nCitation rule (safety):\n"
            "- Use ONLY citation tags from this list (copy exactly).\n"
            "- Do NOT invent new chunk numbers or new file names.\n"
            "- Every bullet in the required sections must end with one or more citations.\n\n"
            "Allowed citation tags:\n"
            f"{allowed_block}\n"
        )

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=500,
            timeout=30.0,
        )
        error = None
    except Exception as e:
        resp = None
        error = str(e)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if resp is None:
        state["blocked"] = True
        state["answer"] = ""
        state["validation"] = {
            "passed": False,
            "reasons": ["Answer generation failed; not showing a partial answer.", error],
            "warnings": [],
            "next_actions": ["Confirm OPENAI_API_KEY and network access, then try again."],
            "stats": state.get("validation", {}).get("stats") if isinstance(state.get("validation"), dict) else {},
        }
        _append_trace(
            state,
            {
                "ts": _now_iso_utc(),
                "step": "answer",
                "engine": "text_generation",
                "temperature": 0.0,
                "duration_ms": round(dt_ms, 1),
                "error": error,
            },
        )
        return state

    state["answer"] = resp.choices[0].message.content or ""

    usage: dict[str, Any] = {}
    if getattr(resp, "usage", None) is not None:
        # Keep it JSON-friendly; OpenAI objects aren't always plain dicts.
        usage = {
            "input_tokens": getattr(resp.usage, "prompt_tokens", None),
            "output_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        }

    _append_trace(
        state,
        {
            "ts": _now_iso_utc(),
            "step": "answer",
            "engine": "text_generation",
            "temperature": 0.0,
            "duration_ms": round(dt_ms, 1),
            "jurisdiction": state_code,
            "require_citations": bool(require_citations),
            "input_chars": len(prompt),
            "sources_chars": len(sources),
            "usage": usage,
            "error": None,
        },
    )
    return state


def _normalize_required_bullet_citations(answer: str) -> str:
    """Move any existing citations in required bullets to the end of the bullet line.

    This does not add or remove citations; it only normalizes placement so the
    verification rule ("bullets end with citations") isn't tripped by harmless formatting.
    """
    current_required: str | None = None
    out_lines: list[str] = []

    for raw_line in (answer or "").splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.endswith(":"):
            header = stripped[:-1].strip()
            current_required = _required_section_key(header)
            out_lines.append(line)
            continue

        if current_required is not None and stripped.startswith("-"):
            cites = [m.group(0) for m in _CITATION_RE.finditer(line)]
            if cites:
                # Remove citations from their original positions.
                without = _CITATION_RE.sub("", line).rstrip()
                # Collapse excessive spaces created by removal.
                without = re.sub(r"\s{2,}", " ", without).rstrip()
                normalized = (without.rstrip(" .") + " " + " ".join(cites)).rstrip()
                out_lines.append(normalized)
                continue

        out_lines.append(line)

    return "\n".join(out_lines)


def _collapse_wrapped_required_bullets(answer: str) -> str:
    """Collapse wrapped bullet lines in required sections into single logical bullets.

    Some model outputs hard-wrap long bullets across multiple lines. The citation
    verifier expects each required bullet to be a single line that ends with one
    or more citations. This normalizes formatting without changing meaning.
    """
    current_required: str | None = None
    out_lines: list[str] = []

    bullet_acc: str | None = None

    def _flush_bullet() -> None:
        nonlocal bullet_acc
        if bullet_acc is not None:
            out_lines.append(bullet_acc.rstrip())
            bullet_acc = None

    for raw_line in (answer or "").splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        # Section header
        if stripped.endswith(":"):
            _flush_bullet()
            header = stripped[:-1].strip()
            current_required = _required_section_key(header)
            out_lines.append(line)
            continue

        # Some outputs insert blank lines mid-bullet. Ignore those while accumulating.
        if bullet_acc is not None and not stripped:
            continue

        if current_required is not None and stripped.startswith("-"):
            # Sometimes a single bullet gets hard-wrapped and the next line
            # incorrectly starts with '-' again. If the prior bullet is just a
            # short fragment with no citations, treat this as a continuation.
            if bullet_acc is not None:
                prior_has_cite = bool(_CITATION_RE.search(bullet_acc))
                prior_short = len(bullet_acc) <= 48
                continuation_text = stripped[1:].lstrip()
                continuation_starts_lower = bool(continuation_text) and continuation_text[:1].islower()
                if (not prior_has_cite) and prior_short and continuation_starts_lower:
                    bullet_acc = (bullet_acc.rstrip() + " " + continuation_text).strip()
                    continue

            _flush_bullet()
            bullet_acc = stripped
            continue

        # Continuation of a wrapped bullet inside required sections
        if current_required is not None and bullet_acc is not None and stripped and not stripped.startswith("-"):
            bullet_acc = (bullet_acc.rstrip() + " " + stripped).strip()
            continue

        _flush_bullet()
        out_lines.append(line)

    _flush_bullet()
    return "\n".join(out_lines)


_CITATION_RE = re.compile(
    r"\[(?P<file>[^\]|]+?)\s*\|\s*(?P<doc_type>[^\]|]+?)\s*\|\s*(?:p\.\s*(?P<page>\d+)\s*\|\s*)?chunk\s*(?P<chunk>\d+)\s*\]",
    re.IGNORECASE,
)


def _expected_citation_keys(raw_results: list[dict[str, Any]]) -> set[tuple[str, str, int, int | None]]:
    keys: set[tuple[str, str, int, int | None]] = set()
    for r in raw_results or []:
        file_name = r.get("file_name")
        doc_type = r.get("doc_type")
        chunk_index = r.get("chunk_index")
        page_number = r.get("page_number")

        if not isinstance(file_name, str) or not file_name.strip():
            continue
        if not isinstance(doc_type, str) or not doc_type.strip():
            continue
        if not isinstance(chunk_index, int):
            continue
        if page_number is not None and not isinstance(page_number, int):
            page_number = None

        keys.add((file_name.strip(), doc_type.strip(), int(chunk_index), page_number))
    return keys


def _citation_matches(
    cite: tuple[str, str, int, int | None],
    expected: set[tuple[str, str, int, int | None]],
) -> bool:
    file_name, doc_type, chunk_index, page_number = cite
    if (file_name, doc_type, chunk_index, page_number) in expected:
        return True
    # If a page number is missing, we accept any page for that chunk.
    if page_number is None:
        for f, d, c, _p in expected:
            if f == file_name and d == doc_type and c == chunk_index:
                return True
    return False


def _bullets_requiring_citations(answer: str) -> list[str]:
    current_required: str | None = None
    bullets: list[str] = []

    for raw_line in (answer or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.endswith(":"):
            header = line[:-1].strip()
            current_required = _required_section_key(header)
            continue

        if line.startswith("-") and current_required is not None:
            bullets.append(line)

    return bullets


def _required_section_key(header: str) -> str | None:
    """Return a stable key for required sections, tolerating minor formatting drift."""
    h = (header or "").strip().lower()
    # Normalize punctuation/slashes to spaces, collapse whitespace.
    h = re.sub(r"[^a-z0-9]+", " ", h)
    h = re.sub(r"\s+", " ", h).strip()

    if h == "coverage answer":
        return "coverage"
    # Accept minor variations like "key conditions exclusions to watch".
    if h.startswith("key conditions"):
        return "conditions"
    return None


def _verify_answer_citations(answer: str, raw_results: list[dict[str, Any]]) -> dict[str, Any]:
    expected = _expected_citation_keys(raw_results)
    citations: list[tuple[str, str, int, int | None]] = []

    for m in _CITATION_RE.finditer(answer or ""):
        file_name = (m.group("file") or "").strip()
        doc_type = (m.group("doc_type") or "").strip()
        chunk_index = int(m.group("chunk"))
        page_raw = m.group("page")
        page_number = int(page_raw) if page_raw is not None else None
        citations.append((file_name, doc_type, chunk_index, page_number))

    issues: list[str] = []
    if not citations:
        issues.append("No citations found in the answer.")

    unknown: list[str] = []
    for file_name, doc_type, chunk_index, page_number in citations:
        if not _citation_matches((file_name, doc_type, chunk_index, page_number), expected):
            if page_number is None:
                unknown.append(f"[{file_name} | {doc_type} | chunk {chunk_index}]")
            else:
                unknown.append(f"[{file_name} | {doc_type} | p. {page_number} | chunk {chunk_index}]")
    if unknown:
        issues.append("Citations not found in retrieved matches: " + ", ".join(unknown[:8]) + (" ..." if len(unknown) > 8 else ""))

    bad_bullets: list[str] = []
    for bullet in _bullets_requiring_citations(answer):
        # Bullet must end with a citation block (or multiple), not trailing prose.
        if not re.search(r"\]\s*$", bullet):
            bad_bullets.append(bullet)
    if bad_bullets:
        issues.append("Some required bullets do not end with citations.")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "bad_bullets_preview": [b[:240] for b in bad_bullets[:3]],
        "stats": {
            "retrieved_matches": len(raw_results or []),
            "expected_citation_keys": len(expected),
            "citations_found": len(citations),
            "bad_bullets": len(bad_bullets),
        },
    }


def verify_citations_node(state: GraphState) -> GraphState:
    require_citations = bool(state.get("require_citations", True))
    if not require_citations:
        return state

    answer = state.get("answer") or ""
    raw_results = state.get("raw_results", []) or []

    # First, collapse wrapped bullets in required sections so we don't mis-read
    # line breaks as missing end-of-bullet citations.
    collapsed = _collapse_wrapped_required_bullets(answer)
    if collapsed != answer:
        answer = collapsed
        state["answer"] = collapsed

    # Then, normalize harmless formatting issues (citations mid-bullet).
    normalized = _normalize_required_bullet_citations(answer)
    if normalized != answer:
        answer = normalized
        state["answer"] = normalized

    started = time.perf_counter()
    check = _verify_answer_citations(answer, raw_results)
    dt_ms = (time.perf_counter() - started) * 1000.0

    # One retry for citation-formatting drift. More than that just burns time.
    retry_count = int(state.get("answer_retry_count") or 0)
    if not check.get("passed") and retry_count < 1:
        sources = state.get("sources") or "(no sources found)"
        question = state.get("question")
        state_code = state.get("state", "IL")

        raw_results_for_tags = raw_results
        allowed_tags: list[str] = []
        for r in raw_results_for_tags:
            file_name = r.get("file_name")
            doc_type = r.get("doc_type")
            chunk_index = r.get("chunk_index")
            page_number = r.get("page_number")
            if not isinstance(file_name, str) or not file_name.strip():
                continue
            if not isinstance(doc_type, str) or not doc_type.strip():
                continue
            if not isinstance(chunk_index, int):
                continue
            if isinstance(page_number, int):
                allowed_tags.append(f"[{file_name.strip()} | {doc_type.strip()} | p. {page_number} | chunk {chunk_index}]")
            else:
                allowed_tags.append(f"[{file_name.strip()} | {doc_type.strip()} | chunk {chunk_index}]")

        if allowed_tags:
            allowed_block = "\n".join(f"- {t}" for t in sorted(set(allowed_tags)))
            retry_prompt = USER_TEMPLATE.format(
                question=question,
                state_code=state_code,
                sources=sources,
                require_citations=True,
            )
            retry_prompt += (
                "\n\nYour previous draft failed citation checks. Rewrite from scratch.\n"
                "Rules (must follow):\n"
                "- Keep it short: use exactly 2 bullets in 'Coverage answer' and exactly 2 bullets in 'Key conditions / exclusions to watch'.\n"
                "- Keep each bullet concise (aim for one sentence).\n"
                "- Use ONLY citation tags from the Allowed list (copy exactly).\n"
                "- Every bullet in 'Coverage answer' and 'Key conditions / exclusions to watch' must end with citations.\n"
                "- Do not add any extra prose after the last citation on a bullet.\n\n"
                "Allowed citation tags:\n"
                f"{allowed_block}\n"
            )

            try:
                resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": retry_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=500,
                    timeout=30.0,
                )
                rewritten = (resp.choices[0].message.content or "").strip()
                rewritten = _collapse_wrapped_required_bullets(rewritten)
                rewritten = _normalize_required_bullet_citations(rewritten)
                state["answer"] = rewritten
                state["answer_retry_count"] = retry_count + 1
                answer = rewritten

                started = time.perf_counter()
                check = _verify_answer_citations(answer, raw_results)
                dt_ms = (time.perf_counter() - started) * 1000.0
            except Exception:
                # If the retry fails, we fall through to the normal block behavior.
                state["answer_retry_count"] = retry_count + 1

    if not check.get("passed"):
        state["blocked"] = True

        existing_validation = state.get("validation")
        validation: dict[str, Any] = existing_validation if isinstance(existing_validation, dict) else {}
        reasons = list(validation.get("reasons") or [])
        next_actions = list(validation.get("next_actions") or [])

        reasons.append("Citation verification failed; refusing to show an unverified answer.")
        for issue in check.get("issues") or []:
            reasons.append(str(issue))

        if not next_actions:
            next_actions.append("Try asking again; citation formatting can be finicky.")
            next_actions.append("If it keeps failing, re-index and ask a narrower question.")

        validation["passed"] = False
        validation["reasons"] = reasons
        validation["next_actions"] = next_actions
        validation["citation_verification"] = check
        state["validation"] = validation

        # Do not leak an answer that cannot be defended.
        state["answer"] = ""

    _append_trace(
        state,
        {
            "ts": _now_iso_utc(),
            "step": "citation_verify",
            "passed": bool(check.get("passed")),
            "duration_ms": round(dt_ms, 1),
            "stats": check.get("stats"),
        },
    )

    return state


def validate_node(state: GraphState) -> GraphState:
    """Gate the answer step when retrieval is clearly insufficient.

    Blocks when matches are missing/weak, and otherwise allows the answer step to run
    while recording the decision in the audit trace.
    """
    # If an upstream node already decided we must stop (e.g., retrieval tool failed),
    # don't overwrite its validation details.
    if state.get("blocked"):
        _append_trace(
            state,
            {
                "ts": _now_iso_utc(),
                "step": "validate",
                "passed": False,
                "note": "skipped_validate_due_to_prior_block",
            },
        )
        return state

    require_citations = bool(state.get("require_citations", True))
    results = state.get("raw_results", []) or []

    unique_files = sorted({v for r in results for v in [r.get("file_name")] if isinstance(v, str) and v})
    unique_doc_types = sorted({v for r in results for v in [r.get("doc_type")] if isinstance(v, str) and v})

    scores: list[float] = []
    for r in results:
        s = r.get("score")
        if isinstance(s, (int, float)):
            scores.append(float(s))

    max_score = max(scores) if scores else None

    reasons: list[str] = []
    warnings: list[str] = []
    next_actions: list[str] = []
    blocked = False

    if require_citations and len(results) == 0:
        blocked = True
        reasons.append("No retrieved matches; cannot provide a grounded answer.")
        next_actions.append("Confirm you indexed the correct docs folder, then re-run indexing.")
        next_actions.append("Try a more specific question (peril + coverage + where it happened).")

    if require_citations and max_score is not None and max_score < 0.10:
        blocked = True
        reasons.append(f"Top relevance score is very low ({max_score:.3f}); likely not grounded.")
        next_actions.append("Re-index the docs folder (or add more policy/endorsement pages), then try again.")
    elif require_citations and max_score is not None and max_score < 0.20:
        warnings.append(f"Top relevance score is low ({max_score:.3f}); answer may be weak.")

    if require_citations and len(unique_files) == 1 and len(results) >= 1:
        warnings.append("All matches are from a single file; consider indexing more document types.")

    # Guardrail: if evidence is both weak and narrow, stop.
    if require_citations and len(results) > 0:
        low_conf = (max_score is None) or (max_score < 0.25)
        low_diversity = (len(unique_files) < 2) and (len(unique_doc_types) < 2)
        if low_conf and low_diversity:
            blocked = True
            reasons.append(
                "Evidence is too thin (low diversity + weak relevance). Safer to stop and ask for better evidence."
            )
            next_actions.append("Index more of the policy packet (declarations + endorsements + policy booklet).")
            next_actions.append("Rephrase the question using policy terms (e.g., 'water backup', 'sewer', 'sump overflow').")
        elif low_diversity:
            warnings.append("Evidence comes from a narrow slice of the packet; treat the answer as incomplete.")

    state["blocked"] = blocked
    state["validation"] = {
        "passed": not blocked,
        "reasons": reasons,
        "warnings": warnings,
        "next_actions": next_actions,
        "stats": {
            "result_count": len(results),
            "unique_files": len(unique_files),
            "unique_doc_types": len(unique_doc_types),
            "max_score": max_score,
        },
    }

    _append_trace(
        state,
        {
            "ts": _now_iso_utc(),
            "step": "validate",
            "passed": not blocked,
            "reasons": reasons,
            "warnings": warnings,
            "stats": state["validation"]["stats"],
        },
    )
    return state


def build_graph():
    """
    Creates: plan -> retrieve -> precedence_check -> validate -> answer -> END
    """
    g = StateGraph(GraphState)
    g.add_node("plan", plan_retrievals_node)
    g.add_node("retrieve", multi_retrieve_node)
    g.add_node("precedence_check", precedence_check_node)
    g.add_node("validate", validate_node)
    g.add_node("answer", answer_node)
    g.add_node("citation_verify", verify_citations_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "precedence_check")
    g.add_edge("precedence_check", "validate")

    def _route_after_validate(state: GraphState) -> str:
        return "end" if state.get("blocked") else "answer"

    g.add_conditional_edges(
        "validate",
        _route_after_validate,
        {
            "answer": "answer",
            "end": END,
        },
    )

    # We verify citations after the answer is generated, and can still block the run.
    g.add_edge("answer", "citation_verify")
    g.add_edge("citation_verify", END)

    return g.compile()
