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


def retrieve_node(state: GraphState) -> GraphState:
    """
    Node 1: Fetch relevant clauses from Qdrant via MCP.
    Input:  state["question"]
    Output: state["sources"] (a formatted string)
    """
    if not state.get("run_id"):
        state["run_id"] = uuid.uuid4().hex

    question = state["question"]

    # LangGraph node is sync, MCP client is async. This is the simplest bridge.
    import asyncio
    t0 = time.perf_counter()
    try:
        data = asyncio.run(retrieve_clauses(question, top_k=5))
        err = None
    except Exception as e:
        data = {"results": []}
        err = str(e)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # Convert results list into a readable SOURCES block for the answer step.
    lines = []
    for r in data.get("results", []):
        page_number = r.get("page_number")
        if page_number is not None:
            cite = f'[{r.get("file_name")} | {r.get("doc_type")} | p. {page_number} | chunk {r.get("chunk_index")}]'
        else:
            cite = f'[{r.get("file_name")} | {r.get("doc_type")} | chunk {r.get("chunk_index")}]'
        snippet = (r.get("snippet") or "").replace("\n", " ").strip()
        # Keep this short; long snippets add latency without helping citations.
        if len(snippet) > 360:
            snippet = snippet[:360].rstrip() + "..."
        lines.append(f"- {cite} {snippet}")

    state["sources"] = "\n".join(lines) if lines else "(no sources found)"
    state["raw_results"] = data.get("results", [])

    results = state.get("raw_results", [])
    top_scores: list[float] = []
    for r in results[:3]:
        s = r.get("score")
        if isinstance(s, (int, float)):
            top_scores.append(float(s))

    _append_trace(
        state,
        {
            "ts": _now_iso_utc(),
            "step": "retrieve",
            "tool": "retrieve_clauses",
            "duration_ms": round(dt_ms, 1),
            "question": _redact_text(question),
            "args": {"top_k": 5},
            "result_count": len(results),
            "top_scores": top_scores,
            "error": err,
        },
    )

    if err:
        state["blocked"] = True
        state["validation"] = {
            "passed": False,
            "reasons": ["Retrieval failed; cannot proceed safely.", err],
            "warnings": [],
            "next_actions": [
                "Check MCP server, Qdrant, and OPENAI_API_KEY, then try again.",
                "Refresh Index Status in the sidebar to confirm readiness.",
            ],
            "stats": {"result_count": 0, "unique_files": 0, "unique_doc_types": 0, "max_score": None},
        }
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
                "model": "gpt-4.1-mini",
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

    # Endorsement check: if endorsements show up, flag possible overrides.
    # This part was a pain to get right; keep it simple and explicit.
    raw_results = state.get("raw_results", []) or []
    endorsement_signals = _detect_endorsement_conflicts(raw_results)
    state["endorsement_signals"] = endorsement_signals

    if endorsement_signals.get("present"):
        prompt += (
            "\n\nEndorsement conflict check (important):\n"
            "- Endorsements may modify or override the base policy language.\n"
            "- If you see a conflict between endorsement and booklet/declarations, say so explicitly and prefer the endorsement text.\n"
            "- If applicability is unclear (effective date / form / state), call it out under 'What to verify'.\n"
        )

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
            max_tokens=350,
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
                "model": "gpt-4.1-mini",
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
            "model": "gpt-4.1-mini",
            "temperature": 0.0,
            "duration_ms": round(dt_ms, 1),
            "jurisdiction": state_code,
            "require_citations": bool(require_citations),
            "prompt_chars": len(prompt),
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
    required_sections = {"Coverage answer", "Key conditions / exclusions to watch"}
    current_section: str | None = None
    out_lines: list[str] = []

    for raw_line in (answer or "").splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.endswith(":"):
            current_section = stripped[:-1].strip()
            out_lines.append(line)
            continue

        if current_section in required_sections and stripped.startswith("-"):
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
    required_sections = {
        "Coverage answer": True,
        "Key conditions / exclusions to watch": True,
    }
    current_section: str | None = None
    bullets: list[str] = []

    for raw_line in (answer or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.endswith(":"):
            header = line[:-1].strip()
            current_section = header
            continue

        if line.startswith("-") and current_section in required_sections:
            bullets.append(line)

    return bullets


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

    # First, normalize harmless formatting issues (citations mid-bullet).
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
                    max_tokens=350,
                    timeout=30.0,
                )
                rewritten = (resp.choices[0].message.content or "").strip()
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
    Creates: retrieve -> answer -> END
    """
    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("validate", validate_node)
    g.add_node("answer", answer_node)
    g.add_node("citation_verify", verify_citations_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "validate")

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
