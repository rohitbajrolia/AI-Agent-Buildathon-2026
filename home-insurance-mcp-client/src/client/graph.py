import os
import time
import json
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
    retrieval_round: NotRequired[int]
    retrieval_gaps: NotRequired[list[dict[str, Any]]]
    pending_retrieval_queries: NotRequired[list[dict[str, str]]]
    relevance_rating: NotRequired[str | None]
    domain_plausible: NotRequired[bool]
    conversation_history: NotRequired[list[dict[str, str]]]


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


# ---- Domain pre-check (keyword-based heuristic) ----
_INSURANCE_KEYWORDS = frozenset({
    "insurance", "policy", "coverage", "claim", "deductible", "premium",
    "endorsement", "exclusion", "peril", "liability", "dwelling", "homeowner",
    "covered", "loss", "damage", "flood", "fire", "theft", "wind", "hail",
    "water", "mold", "sewer", "backup", "sump", "pipe", "burst", "tree",
    "collapse", "earthquake", "tornado", "hurricane", "lightning",
    "property", "structure", "roof", "foundation", "basement",
    "personal property", "other structures", "additional living",
    "declarations", "conditions", "duties", "notice", "proof of loss",
    "subrogation", "replacement cost", "actual cash value", "limit",
    "sublimit", "rider", "schedule", "insured", "policyholder",
})

_HOMEOWNERS_KEYWORDS = frozenset({
    "homeowners", "homeowner", "dwelling", "other structures", "personal property",
    "additional living expenses", "section i", "residence premises", "policy booklet",
    "home policy", "property coverage", "coverage a", "coverage b", "coverage c", "coverage d",
})

# Broader property/peril terms that suggest a homeowners angle even when auto terms are also present.
# Used to distinguish mixed-LOB questions from pure auto questions.
_HOMEOWNERS_ADJACENT_KEYWORDS = frozenset({
    "home", "house", "garage", "driveway", "yard", "lawn", "attic", "deck",
    "patio", "pool", "fence", "shed", "backyard", "roof",
    "hail", "hailstorm", "windstorm", "fallen tree", "falling object",
    "landlord", "tenant", "renter", "condo",
})

_AUTO_ONLY_KEYWORDS = frozenset({
    "collision deductible", "collision coverage", "comprehensive coverage", "auto insurance",
    "car insurance", "vehicle deductible", "auto deductible", "motor vehicle liability",
    "uninsured motorist", "underinsured motorist", "vin", "license plate",
})


def _is_plausibly_insurance(question: str) -> bool:
    """Return True if the question contains at least one insurance-domain keyword."""
    q_lower = (question or "").lower()
    return any(kw in q_lower for kw in _INSURANCE_KEYWORDS)


def _scope_gate(question: str) -> tuple[bool, str | None]:
    """Deterministic scope gate for the homeowners-insurance workflow."""
    q_lower = (question or "").lower()

    if not _is_plausibly_insurance(question):
        return (False, "non_insurance_topic")

    auto_hits = any(k in q_lower for k in _AUTO_ONLY_KEYWORDS)
    if not auto_hits:
        return (True, None)

    # Question contains auto-specific terms. If it also has a homeowners or
    # property/peril signal, treat it as a mixed-LOB question and let it through.
    # The model will answer only what the loaded docs support and defer the rest.
    homeowners_hits = any(k in q_lower for k in _HOMEOWNERS_KEYWORDS)
    homeowners_adjacent = any(k in q_lower for k in _HOMEOWNERS_ADJACENT_KEYWORDS)

    if homeowners_hits or homeowners_adjacent:
        return (True, None)

    if _STRICT_LINE_OF_BUSINESS_GATE:
        return (False, "auto_line_of_business")

    return (True, None)


def plan_retrievals_node(state: GraphState) -> GraphState:
    if not state.get("run_id"):
        state["run_id"] = uuid.uuid4().hex

    question = state["question"]
    state_code = state.get("state", "IL")

    state["domain_plausible"] = _is_plausibly_insurance(question)
    in_scope, scope_reason = _scope_gate(question)

    if not in_scope:
        state["blocked"] = True
        state["relevance_rating"] = "NONE"
        state["retrieval_plan"] = []
        state["raw_results"] = []
        state["sources"] = "(no sources found)"

        reason_text = (
            "This question is about auto coverage. The loaded documents are homeowners policy documents. "
            "Load auto policy documents or switch to the auto workflow to answer this question."
            if scope_reason == "auto_line_of_business"
            else "Question appears outside insurance-policy coverage scope for this workflow."
        )

        state["validation"] = {
            "passed": False,
            "reasons": [reason_text],
            "warnings": [],
            "next_actions": [
                "Ask a homeowners-insurance question grounded in indexed policy documents.",
                "For auto-policy questions, use an auto-insurance workflow/document set.",
            ],
            "stats": {
                "result_count": 0,
                "unique_files": 0,
                "unique_doc_types": 0,
                "max_score": None,
                "avg_score": None,
            },
        }

        _append_trace(
            state,
            {
                "ts": _now_iso_utc(),
                "step": "plan",
                "question": _redact_text(question),
                "jurisdiction": state_code,
                "domain_plausible": state["domain_plausible"],
                "scope_gate": scope_reason,
                "blocked": True,
            },
        )
        return state

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
            "domain_plausible": state["domain_plausible"],
            "planned": state.get("retrieval_plan", []),
        },
    )
    return state


def multi_retrieve_node(state: GraphState) -> GraphState:
    if state.get("blocked"):
        return state

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
            data = asyncio.run(
                retrieve_clauses(
                    query,
                    top_k=_RETRIEVE_TOP_K,
                    score_threshold=_RETRIEVE_SCORE_THRESHOLD,
                )
            )
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
    results_final = results_final[:_RESULT_CAP]

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


# ---------------------------------------------------------------------------
# Adaptive re-retrieval loop
# ---------------------------------------------------------------------------
# After the initial multi-retrieve, the workflow evaluates retrieved results:
#   - Are there coverage gaps (planned topics with zero or weak results)?
#   - Is doc-type diversity too low (e.g., no endorsements at all)?
#   - Are relevance scores too weak across the board?
#
# If gaps are found and re-retrieval rounds remain, the workflow:
#   1. Uses the model to reformulate search queries targeting the gaps.
#   2. Executes the reformulated queries via MCP retrieve_clauses.
#   3. Merges new results with existing ones (dedup, keep best scores).
#   4. Re-evaluates. Loops up to _MAX_RE_RETRIEVAL_ROUNDS times.
#
# This loop improves retrieval quality by re-querying when evidence is weak.
# ---------------------------------------------------------------------------

_MAX_RE_RETRIEVAL_ROUNDS = 2
_RETRIEVE_TOP_K = 4
_RESULT_CAP = 12
_RETRIEVE_SCORE_THRESHOLD = float(os.getenv("RETRIEVE_SCORE_THRESHOLD", "0.55") or "0.55")
_STRICT_LINE_OF_BUSINESS_GATE = os.getenv("STRICT_LINE_OF_BUSINESS_GATE", "1").strip().lower() in {"1", "true", "yes", "y"}


def _evaluate_retrieval_quality(
    raw_results: list[dict[str, Any]],
    plan_topics: list[str],
) -> dict[str, Any]:
    """Evaluate retrieval quality and identify evidence gaps.

    Returns a dict with:
    - sufficient (bool): True if evidence is good enough to proceed.
    - gaps (list): identified weaknesses that re-retrieval should target.
    - stats: summary metrics.
    """
    if not raw_results:
        return {
            "sufficient": False,
            "gaps": [{"type": "no_results", "detail": "Zero results across all retrieval topics."}],
            "stats": {
                "total": 0, "max_score": 0.0, "avg_score": 0.0,
                "doc_types": [], "unique_files": 0,
                "covered_topics": [], "missing_topics": list(plan_topics),
            },
        }

    scores = [
        float(r.get("score", 0))
        for r in raw_results
        if isinstance(r.get("score"), (int, float))
    ]
    max_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0

    doc_types = sorted({
        (r.get("doc_type") or "").strip().lower()
        for r in raw_results if r.get("doc_type")
    })
    unique_files = len({
        r.get("file_name") for r in raw_results if r.get("file_name")
    })

    # Determine which plan topics were covered by at least one result.
    covered_topics: set[str] = set()
    for r in raw_results:
        topics = r.get("retrieval_topics", [])
        if isinstance(topics, list):
            for t in topics:
                # Strip " (re-retrieve round N)" suffix for matching.
                base = re.sub(r"\s*\(re-retrieve round \d+\)$", "", str(t)).strip()
                covered_topics.add(base)

    missing_topics = [t for t in plan_topics if t not in covered_topics]

    gaps: list[dict[str, Any]] = []

    # Gap: entire plan topics returned nothing.
    for mt in missing_topics:
        gaps.append({
            "type": "missing_topic",
            "topic": mt,
            "detail": f"Topic '{mt}' returned zero results.",
        })

    # Gap: no endorsement docs at all (critical for insurance override detection).
    if "endorsement" not in doc_types and len(raw_results) >= 2:
        gaps.append({
            "type": "missing_doc_type",
            "doc_type": "endorsement",
            "detail": "No endorsement documents retrieved; endorsements can override base policy.",
        })

    # Gap: all results clustered in one doc_type.
    if len(doc_types) < 2 and len(raw_results) >= 3:
        gaps.append({
            "type": "low_doc_diversity",
            "detail": f"All {len(raw_results)} results from doc_type(s) {doc_types}.",
        })

    # Gap: top score is very weak.
    if 0 < max_score < 0.45:
        gaps.append({
            "type": "low_relevance",
            "detail": f"Best score is {max_score:.3f}; results may be off-topic.",
        })

    # Gap: average score is low (noise floor for embedding model).
    if avg_score > 0 and avg_score < 0.35:
        gaps.append({
            "type": "low_avg_relevance",
            "detail": f"Average score is {avg_score:.3f}; most results lack strong semantic match.",
        })

    return {
        "sufficient": len(gaps) == 0,
        "gaps": gaps,
        "stats": {
            "total": len(raw_results),
            "max_score": round(max_score, 4),
            "avg_score": round(avg_score, 4),
            "doc_types": doc_types,
            "unique_files": unique_files,
            "covered_topics": sorted(covered_topics),
            "missing_topics": missing_topics,
        },
    }


_REFORMULATION_SYSTEM = (
    "You are a search query reformulation expert for insurance policy document retrieval. "
    "Given a user question and retrieval gaps, generate alternative search queries to fill those gaps. "
    "Use insurance-specific terminology (section headers, coverage names, endorsement forms). "
    "Output ONLY a valid JSON array of objects, each with 'topic' (string) and 'query' (string) fields. "
    "Return only the JSON array."
)

_REFORMULATION_USER = (
    'Original question: "{question}"\n\n'
    "Retrieval gaps:\n{gap_summary}\n\n"
    "Generate 1-2 alternative search queries per gap. Use different phrasing and insurance "
    "terminology (e.g., 'Section I Perils Insured Against', 'Loss Settlement', policy form headers).\n"
    "Output ONLY a JSON array.\n"
    'Example: [{{"topic": "Exclusions", "query": "Section I Exclusions water damage not covered"}}]'
)


def _reformulate_gap_queries(
    question: str,
    gaps: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Use the model to generate better search queries for identified retrieval gaps.

    Falls back to simple keyword reformulation if the model call fails.
    """
    if not gaps:
        return []

    gap_lines = []
    for g in gaps:
        g_type = g.get("type", "")
        topic = g.get("topic", g.get("doc_type", ""))
        detail = g.get("detail", "")
        gap_lines.append(f"- [{g_type}] {topic}: {detail}")
    gap_summary = "\n".join(gap_lines)

    # Attempt model-based reformulation.
    if OPENAI_API_KEY:
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": _REFORMULATION_SYSTEM},
                    {"role": "user", "content": _REFORMULATION_USER.format(
                        question=question, gap_summary=gap_summary)},
                ],
                temperature=0.0,
                max_tokens=300,
                timeout=15.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            # Strip markdown fences if the model wraps output.
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.lower().startswith("json"):
                    raw = raw[4:]
            queries = json.loads(raw)
            if isinstance(queries, list):
                parsed = [
                    {"topic": str(q.get("topic", "")), "query": str(q.get("query", ""))}
                    for q in queries
                    if isinstance(q, dict) and q.get("query")
                ]
                if parsed:
                    return parsed[:6]  # cap at 6 reformulated queries
        except Exception:
            pass  # continue with the heuristic fallback

    # Heuristic fallback: simple keyword rephrasing.
    fallback: list[dict[str, str]] = []
    for g in gaps:
        g_type = g.get("type", "")
        topic = g.get("topic", g.get("doc_type", ""))
        if ("exclusion" in (topic or "").lower()) or (
            g_type == "missing_topic" and "exclusion" in (g.get("detail") or "").lower()
        ):
            fallback.append({"topic": topic, "query": f"What is not covered excluded or limited: {question}"})
        elif ("endorsement" in (topic or "").lower()) or g_type == "missing_doc_type":
            fallback.append({"topic": topic, "query": f"Endorsement form amendment modification: {question}"})
        elif "definition" in (topic or "").lower():
            fallback.append({"topic": topic, "query": f"Definitions meaning of terms: {question}"})
        elif "condition" in (topic or "").lower():
            fallback.append({"topic": topic, "query": f"Conditions duties after loss notice proof: {question}"})
        elif g_type == "low_relevance":
            fallback.append({"topic": "expanded search", "query": f"Home insurance policy coverage for: {question}"})
        else:
            fallback.append({"topic": topic or "general", "query": f"Insurance policy provisions: {question}"})
    return fallback[:6]


def evaluate_retrieval_node(state: GraphState) -> GraphState:
    """Evaluate retrieval results and decide whether to re-retrieve.

    The workflow:
    1. Observes its own retrieval results.
    2. Reasons about what's missing (topic gaps, doc-type gaps, weak scores).
    3. If gaps exist and re-retrieval rounds remain, uses the model to reformulate
       search queries targeting the gaps.
    4. Routes to adaptive_re_retrieve (loop) or proceeds to precedence_check.
    """
    if state.get("blocked"):
        state["pending_retrieval_queries"] = []
        _append_trace(
            state,
            {
                "ts": _now_iso_utc(),
                "step": "evaluate_retrieval",
                "skipped": True,
                "note": "already_blocked_upstream",
            },
        )
        return state

    retrieval_round = int(state.get("retrieval_round") or 0)
    question = state["question"]
    state_code = state.get("state", "IL")
    raw_results = state.get("raw_results", []) or []

    plan_raw = _build_retrieval_plan(question=question, state_code=state_code)
    plan_topics = [p.get("topic", "") for p in plan_raw]

    evaluation = _evaluate_retrieval_quality(raw_results, plan_topics)
    gaps = evaluation.get("gaps", [])
    sufficient = evaluation.get("sufficient", True)

    will_re_retrieve = False
    if not sufficient and retrieval_round < _MAX_RE_RETRIEVAL_ROUNDS:
        reformulated = _reformulate_gap_queries(question=question, gaps=gaps)
        if reformulated:
            state["pending_retrieval_queries"] = reformulated
            state["retrieval_gaps"] = gaps
            will_re_retrieve = True

    if not will_re_retrieve:
        state["pending_retrieval_queries"] = []
        state["retrieval_gaps"] = gaps  # record for audit even if not re-retrieving

    _append_trace(
        state,
        {
            "ts": _now_iso_utc(),
            "step": "evaluate_retrieval",
            "round": retrieval_round,
            "sufficient": sufficient,
            "gap_count": len(gaps),
            "gaps": [
                {"type": g.get("type"), "topic": g.get("topic", g.get("doc_type", "")),
                 "detail": g.get("detail")}
                for g in gaps
            ],
            "will_re_retrieve": will_re_retrieve,
            "reformulated_query_count": len(state.get("pending_retrieval_queries") or []),
            "stats": evaluation.get("stats", {}),
        },
    )
    return state


def adaptive_re_retrieve_node(state: GraphState) -> GraphState:
    """Execute reformulated queries and merge results with existing evidence.

    Called only when evaluate_retrieval_node identified gaps and decided to loop.
    After this node runs, the graph routes back to evaluate_retrieval_node for
    another round of self-evaluation.
    """
    if state.get("blocked"):
        return state

    import asyncio

    retrieval_round = int(state.get("retrieval_round") or 0)
    state["retrieval_round"] = retrieval_round + 1
    round_label = state["retrieval_round"]

    reformulated = state.get("pending_retrieval_queries") or []
    if not reformulated:
        return state

    question = state["question"]
    existing_results = list(state.get("raw_results", []) or [])

    t0 = time.perf_counter()
    errors: list[str] = []
    new_results: list[dict[str, Any]] = []

    for q in reformulated:
        topic = q.get("topic", "(reformulated)")
        query = q.get("query", question)
        try:
            data = asyncio.run(
                retrieve_clauses(
                    query,
                    top_k=_RETRIEVE_TOP_K,
                    score_threshold=_RETRIEVE_SCORE_THRESHOLD,
                )
            )
            results = data.get("results", []) if isinstance(data, dict) else []
        except Exception as e:
            results = []
            errors.append(f"{topic}: {e}")

        for r in results or []:
            if not isinstance(r, dict):
                continue
            r2 = dict(r)
            r2["retrieval_topic"] = f"{topic} (re-retrieve round {round_label})"
            new_results.append(r2)

    # Merge new results with existing (dedup by stable key, keep best score).
    deduped: dict[tuple[str, str, int, int | None], dict[str, Any]] = {}
    for r in existing_results:
        key = _stable_result_key(r)
        if key is None:
            continue
        r3 = dict(r)
        if not isinstance(r3.get("retrieval_topics"), list):
            r3["retrieval_topics"] = (
                [r3.get("retrieval_topic")] if r3.get("retrieval_topic") else []
            )
        deduped[key] = r3

    for r in new_results:
        key = _stable_result_key(r)
        if key is None:
            continue

        existing = deduped.get(key)
        if existing is None:
            r3 = dict(r)
            r3["retrieval_topics"] = (
                [r.get("retrieval_topic")] if r.get("retrieval_topic") else []
            )
            deduped[key] = r3
            continue

        # Merge retrieval_topics.
        topics = list(existing.get("retrieval_topics") or [])
        t = r.get("retrieval_topic")
        if isinstance(t, str) and t and t not in topics:
            topics.append(t)
            existing["retrieval_topics"] = topics

        # Keep the higher score.
        old_score = existing.get("score")
        new_score = r.get("score")
        if isinstance(new_score, (int, float)) and (
            not isinstance(old_score, (int, float))
            or float(new_score) > float(old_score)
        ):
            for k in ["score", "snippet", "text"]:
                if k in r:
                    existing[k] = r.get(k)

    results_final = sorted(
        deduped.values(),
        key=lambda x: float(x.get("score") or 0.0),
        reverse=True,
    )[:_RESULT_CAP]

    # Rebuild SOURCES block.
    lines: list[str] = []
    for r in results_final:
        page_number = r.get("page_number")
        if page_number is not None:
            cite = (
                f'[{r.get("file_name")} | {r.get("doc_type")} | '
                f'p. {page_number} | chunk {r.get("chunk_index")}]'
            )
        else:
            cite = f'[{r.get("file_name")} | {r.get("doc_type")} | chunk {r.get("chunk_index")}]'
        snippet = (r.get("snippet") or "").replace("\n", " ").strip()
        if len(snippet) > 360:
            snippet = snippet[:360].rstrip() + "..."
        lines.append(f"- {cite} {snippet}")

    state["sources"] = "\n".join(lines) if lines else "(no sources found)"
    state["raw_results"] = results_final
    state["pending_retrieval_queries"] = []  # consumed

    dt_ms = (time.perf_counter() - t0) * 1000.0

    _append_trace(
        state,
        {
            "ts": _now_iso_utc(),
            "step": "adaptive_re_retrieve",
            "round": round_label,
            "reformulated_queries": [q.get("query", "")[:120] for q in reformulated],
            "new_results_found": len(new_results),
            "total_after_merge": len(results_final),
            "duration_ms": round(dt_ms, 1),
            "errors": errors[:3],
        },
    )
    return state


def _route_after_evaluation(state: GraphState) -> str:
    """Routing: re-retrieve if the agent prepared reformulated queries, else proceed."""
    if state.get("blocked"):
        return "proceed"
    pending = state.get("pending_retrieval_queries") or []
    return "re_retrieve" if pending else "proceed"


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

    history: list[dict[str, str]] = state.get("conversation_history") or []
    history_block = ""
    if history:
        lines = ["\n\nPrior conversation context (for follow-up reference only — do NOT use as policy evidence):"]
        for i, h in enumerate(history[-3:], 1):
            q_text = str(h.get("question") or "").strip()
            a_text = str(h.get("answer") or "").strip()
            if q_text:
                lines.append(f"Q{i}: {q_text}")
            if a_text:
                lines.append(f"A{i}: {a_text}")
        history_block = "\n".join(lines) + "\n"

    prompt = USER_TEMPLATE.format(
        question=question,
        state_code=state_code,
        sources=sources,
        require_citations=require_citations,
    ) + history_block

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
            max_tokens=900,
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

    # ------------------------------------------------------------------
    # Graph-level hard gate: parse RELEVANCE_RATING from the model output.
    # If the model itself says the sources are LOW or NONE relevance, block
    # the answer regardless of retrieval stats or citation formatting.
    # ------------------------------------------------------------------
    _relevance_match = re.search(
        r"RELEVANCE_RATING:\s*(HIGH|MEDIUM|LOW|NONE)",
        state["answer"],
        re.IGNORECASE,
    )
    relevance_rating = _relevance_match.group(1).upper() if _relevance_match else None
    state["relevance_rating"] = relevance_rating  # store for downstream nodes & UI

    if relevance_rating in ("LOW", "NONE"):
        state["blocked"] = True
        existing_validation = state.get("validation")
        existing_val: dict[str, Any] = existing_validation if isinstance(existing_validation, dict) else {}
        state["validation"] = {
            "passed": False,
            "reasons": [
                f"Model assessed source relevance as {relevance_rating}; "
                "the retrieved documents do not meaningfully address the question."
            ],
            "warnings": existing_val.get("warnings", []),
            "next_actions": [
                "This question may be outside the scope of your indexed homeowners insurance documents.",
                "If you expected coverage information, try rephrasing with specific policy terms.",
            ],
            "stats": existing_val.get("stats", {}),
        }
        state["answer"] = ""  # clear the answer — do not show off-topic content

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
    verification rule ("bullets end with citations") is not tripped by formatting-only issues.
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


def _extract_cited_claims(answer: str) -> list[dict[str, Any]]:
    """Extract required-section bullet claims and their inline citations."""
    claims: list[dict[str, Any]] = []
    current_required: str | None = None

    for raw_line in (answer or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.endswith(":"):
            current_required = _required_section_key(line[:-1].strip())
            continue

        if current_required is None or not line.startswith("-"):
            continue

        cites: list[tuple[str, str, int, int | None]] = []
        for m in _CITATION_RE.finditer(line):
            file_name = (m.group("file") or "").strip()
            doc_type = (m.group("doc_type") or "").strip()
            chunk_index = int(m.group("chunk"))
            page_raw = m.group("page")
            page_number = int(page_raw) if page_raw is not None else None
            cites.append((file_name, doc_type, chunk_index, page_number))

        claim_text = _CITATION_RE.sub("", line)
        claim_text = re.sub(r"^-\s*", "", claim_text).strip()
        claim_text = re.sub(r"\s{2,}", " ", claim_text)

        if claim_text:
            claims.append(
                {
                    "section": current_required,
                    "claim": claim_text,
                    "citations": cites,
                    "raw_bullet": line,
                }
            )

    return claims


def _snippet_for_citation(
    cite: tuple[str, str, int, int | None],
    raw_results: list[dict[str, Any]],
) -> str:
    """Return snippet text matching citation key; tolerant of missing page number."""
    file_name, doc_type, chunk_index, page_number = cite

    for r in raw_results or []:
        rf = r.get("file_name")
        rd = r.get("doc_type")
        rc = r.get("chunk_index")
        rp = r.get("page_number")

        if not (isinstance(rf, str) and isinstance(rd, str) and isinstance(rc, int)):
            continue
        if rf.strip() != file_name or rd.strip() != doc_type or int(rc) != int(chunk_index):
            continue

        if page_number is None or (isinstance(rp, int) and int(rp) == int(page_number)):
            text = r.get("snippet") or r.get("text") or ""
            return str(text).strip()

    return ""


_GROUNDING_SYSTEM = (
    "You are a policy-claim grounding reviewer. "
    "Given claims and cited snippets, decide whether each claim is supported by the cited text. "
    "Be conservative: if text is missing or ambiguous, mark NOT_SUPPORTED. "
    "Output ONLY valid JSON with this shape: "
    "{\"results\":[{\"idx\":0,\"verdict\":\"SUPPORTS|PARTIAL|NOT_SUPPORTED|CONTRADICTS\",\"reason\":\"...\"}]}"
)


def _semantic_grounding_check(answer: str, raw_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Semantic grounding check for required-section claims.

    Blocks when claims are cited structurally but not actually supported by cited snippets.
    """
    claims = _extract_cited_claims(answer)
    if not claims:
        return {
            "passed": False,
            "issues": ["No required claims found for semantic grounding check."],
            "stats": {"claims_total": 0, "supported": 0, "partial": 0, "unsupported": 0, "contradictions": 0},
            "claims": [],
        }

    # Build compact claim bundle with cited snippets.
    bundle: list[dict[str, Any]] = []
    for idx, c in enumerate(claims):
        snippets: list[str] = []
        for cite in c.get("citations") or []:
            snip = _snippet_for_citation(cite, raw_results)
            if snip:
                snippets.append(snip[:800])
        bundle.append(
            {
                "idx": idx,
                "section": c.get("section"),
                "claim": c.get("claim", ""),
                "snippets": snippets,
            }
        )

    verdicts: dict[int, dict[str, str]] = {}

    # Preferred path: model-based semantic assessment (deterministic temperature).
    if OPENAI_API_KEY:
        try:
            payload = json.dumps({"claims": bundle}, ensure_ascii=False)
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": _GROUNDING_SYSTEM},
                    {"role": "user", "content": payload},
                ],
                temperature=0.0,
                max_tokens=500,
                timeout=20.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.lower().startswith("json"):
                    raw = raw[4:]
            obj = json.loads(raw)
            rows = obj.get("results") if isinstance(obj, dict) else None
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    idx = row.get("idx")
                    verdict = str(row.get("verdict") or "").upper()
                    reason = str(row.get("reason") or "").strip()
                    if isinstance(idx, int) and verdict in {"SUPPORTS", "PARTIAL", "NOT_SUPPORTED", "CONTRADICTS"}:
                        verdicts[idx] = {"verdict": verdict, "reason": reason}
        except Exception:
            verdicts = {}

    # Fallback path: conservative lexical overlap heuristic.
    if not verdicts:
        stop = {
            "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "with", "is", "are", "was", "were",
            "be", "by", "this", "that", "it", "as", "at", "from", "if", "not", "only", "may", "can",
        }
        for row in bundle:
            idx = int(row["idx"])
            claim = str(row.get("claim") or "")
            snippets_text = " ".join(row.get("snippets") or []).lower()
            claim_tokens = [
                t for t in re.findall(r"[a-zA-Z]{3,}", claim.lower())
                if t not in stop
            ]
            if not claim_tokens or not snippets_text:
                verdicts[idx] = {"verdict": "NOT_SUPPORTED", "reason": "Missing claim tokens or cited snippet text."}
                continue
            overlap = sum(1 for t in set(claim_tokens) if t in snippets_text)
            ratio = overlap / max(1, len(set(claim_tokens)))
            if ratio >= 0.45:
                verdicts[idx] = {"verdict": "SUPPORTS", "reason": f"Token overlap ratio {ratio:.2f}."}
            elif ratio >= 0.22:
                verdicts[idx] = {"verdict": "PARTIAL", "reason": f"Token overlap ratio {ratio:.2f}."}
            else:
                verdicts[idx] = {"verdict": "NOT_SUPPORTED", "reason": f"Low token overlap ratio {ratio:.2f}."}

    merged_claims: list[dict[str, Any]] = []
    unsupported = 0
    contradictions = 0
    partial = 0
    supports = 0
    issues: list[str] = []

    for idx, c in enumerate(claims):
        v = verdicts.get(idx, {"verdict": "NOT_SUPPORTED", "reason": "No verdict returned."})
        verdict = v.get("verdict", "NOT_SUPPORTED")
        reason = v.get("reason", "")

        if verdict == "SUPPORTS":
            supports += 1
        elif verdict == "PARTIAL":
            partial += 1
        elif verdict == "CONTRADICTS":
            contradictions += 1
        else:
            unsupported += 1

        merged_claims.append(
            {
                "section": c.get("section"),
                "claim": c.get("claim"),
                "verdict": verdict,
                "reason": reason,
                "citations_count": len(c.get("citations") or []),
            }
        )

    if contradictions > 0:
        issues.append(f"{contradictions} claim(s) contradict cited snippets.")
    if unsupported > 0:
        issues.append(f"{unsupported} claim(s) are not supported by cited snippets.")

    passed = (contradictions == 0 and unsupported == 0)

    return {
        "passed": passed,
        "issues": issues,
        "stats": {
            "claims_total": len(claims),
            "supported": supports,
            "partial": partial,
            "unsupported": unsupported,
            "contradictions": contradictions,
        },
        "claims": merged_claims[:12],
    }


def verify_citations_node(state: GraphState) -> GraphState:
    require_citations = bool(state.get("require_citations", True))
    if not require_citations:
        return state

    # If answer_node already blocked on relevance, skip this step.
    if state.get("blocked"):
        _append_trace(state, {
            "ts": _now_iso_utc(),
            "step": "citation_verify",
            "skipped": True,
            "note": "already_blocked_by_relevance_rating",
        })
        return state

    answer = state.get("answer") or ""
    raw_results = state.get("raw_results", []) or []

    # First, collapse wrapped bullets in required sections so we don't mis-read
    # line breaks as missing end-of-bullet citations.
    collapsed = _collapse_wrapped_required_bullets(answer)
    if collapsed != answer:
        answer = collapsed
        state["answer"] = collapsed

    # Then, normalize formatting-only issues (citations mid-bullet).
    normalized = _normalize_required_bullet_citations(answer)
    if normalized != answer:
        answer = normalized
        state["answer"] = normalized

    started = time.perf_counter()
    check = _verify_answer_citations(answer, raw_results)
    dt_ms = (time.perf_counter() - started) * 1000.0

    # One retry for citation-formatting drift.
    # Additional retries increase latency with limited benefit.
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
                    max_tokens=700,
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
                # If the retry fails, continue with normal block behavior.
                state["answer_retry_count"] = retry_count + 1

    semantic: dict[str, Any] | None = None
    if check.get("passed"):
        semantic = _semantic_grounding_check(answer, raw_results)

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
            next_actions.append("Try asking again; citation formatting may fail on some outputs.")
            next_actions.append("If it keeps failing, re-index and ask a narrower question.")

        validation["passed"] = False
        validation["reasons"] = reasons
        validation["next_actions"] = next_actions
        validation["citation_verification"] = check
        state["validation"] = validation

        # Do not leak an answer that cannot be defended.
        state["answer"] = ""

    elif isinstance(semantic, dict) and not semantic.get("passed"):
        state["blocked"] = True

        existing_validation = state.get("validation")
        validation: dict[str, Any] = existing_validation if isinstance(existing_validation, dict) else {}
        reasons = list(validation.get("reasons") or [])
        next_actions = list(validation.get("next_actions") or [])

        reasons.append("Semantic grounding verification failed; cited snippets do not support one or more claims.")
        for issue in semantic.get("issues") or []:
            reasons.append(str(issue))

        if not next_actions:
            next_actions.append("Ask a narrower question and avoid combining multiple unrelated coverage asks.")
            next_actions.append("Ensure indexed documents contain explicit language for each requested claim.")

        validation["passed"] = False
        validation["reasons"] = reasons
        validation["next_actions"] = next_actions
        validation["citation_verification"] = check
        validation["semantic_grounding"] = semantic
        state["validation"] = validation

        # Do not leak an answer that is not semantically grounded.
        state["answer"] = ""

    elif isinstance(semantic, dict):
        existing_validation = state.get("validation")
        validation: dict[str, Any] = existing_validation if isinstance(existing_validation, dict) else {}
        validation["semantic_grounding"] = semantic
        state["validation"] = validation

    _append_trace(
        state,
        {
            "ts": _now_iso_utc(),
            "step": "citation_verify",
            "passed": bool(check.get("passed")) and bool((semantic or {}).get("passed", True)),
            "duration_ms": round(dt_ms, 1),
            "stats": check.get("stats"),
            "semantic": semantic.get("stats") if isinstance(semantic, dict) else None,
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
    avg_score = (sum(scores) / len(scores)) if scores else None

    reasons: list[str] = []
    warnings: list[str] = []
    next_actions: list[str] = []
    blocked = False

    # ---------- Evidence quality warnings (always computed, regardless of require_citations) ----------
    # These warnings feed the UI's evidence strength indicator.
    if len(results) == 0:
        warnings.append("No retrieved matches; evidence is absent.")
    elif max_score is not None and max_score < 0.10:
        warnings.append(f"Top relevance score is very low ({max_score:.3f}); likely not grounded.")
    elif max_score is not None and max_score < 0.30:
        warnings.append(f"Top relevance score is low ({max_score:.3f}); answer may be weak.")
    elif max_score is not None and max_score < 0.50:
        warnings.append(f"Top relevance score is moderate ({max_score:.3f}); evidence could be stronger.")
    elif max_score is not None and max_score < 0.70:
        warnings.append(f"Top relevance score is fair ({max_score:.3f}); matches may be tangential.")

    if avg_score is not None and avg_score < 0.25:
        warnings.append(f"Average relevance across matches is low ({avg_score:.3f}).")
    elif avg_score is not None and avg_score < 0.45:
        warnings.append(f"Average relevance across matches is modest ({avg_score:.3f}); many results may be noise.")

    if len(unique_files) == 1 and len(results) >= 1:
        warnings.append("All matches are from a single file; consider indexing more document types.")

    if len(unique_doc_types) < 2 and len(results) >= 2:
        warnings.append("Evidence comes from a narrow doc-type slice; treat the answer as potentially incomplete.")

    if len(results) >= 1 and len(results) <= 2:
        warnings.append(f"Only {len(results)} match(es) found; limited corroboration.")

    # ---------- Blocking gates (only enforced when citations are required) ----------
    if require_citations and len(results) == 0:
        blocked = True
        reasons.append("No retrieved matches; cannot provide a grounded answer.")
        next_actions.append("Confirm you indexed the correct docs folder, then re-run indexing.")
        next_actions.append("Try a more specific question (peril + coverage + where it happened).")

    if require_citations and max_score is not None and max_score < 0.10:
        blocked = True
        reasons.append(f"Top relevance score is very low ({max_score:.3f}); likely not grounded.")
        next_actions.append("Re-index the docs folder (or add more policy/endorsement pages), then try again.")

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
            "avg_score": avg_score,
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
    Retrieval graph with adaptive re-retrieval loop.

    Flow:
        plan -> retrieve -> evaluate_retrieval <-> adaptive_re_retrieve
                                    |  (sufficient)
                            precedence_check -> validate -> answer -> citation_verify -> END
                                                   |  (blocked)
                                                  END
    """
    g = StateGraph(GraphState)
    g.add_node("plan", plan_retrievals_node)
    g.add_node("retrieve", multi_retrieve_node)
    g.add_node("evaluate_retrieval", evaluate_retrieval_node)
    g.add_node("adaptive_re_retrieve", adaptive_re_retrieve_node)
    g.add_node("precedence_check", precedence_check_node)
    g.add_node("validate", validate_node)
    g.add_node("answer", answer_node)
    g.add_node("citation_verify", verify_citations_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "evaluate_retrieval")

    # Adaptive loop: evaluate -> re-retrieve -> evaluate (up to _MAX_RE_RETRIEVAL_ROUNDS).
    g.add_conditional_edges(
        "evaluate_retrieval",
        _route_after_evaluation,
        {
            "re_retrieve": "adaptive_re_retrieve",
            "proceed": "precedence_check",
        },
    )
    g.add_edge("adaptive_re_retrieve", "evaluate_retrieval")

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
