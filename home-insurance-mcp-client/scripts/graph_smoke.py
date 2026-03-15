"""Smoke test: run the full production graph (build_graph) end-to-end.

Usage:
    python scripts/graph_smoke.py                              # default insurance question
    python scripts/graph_smoke.py --question "sourdough bread"  # should block as off-topic
    python scripts/graph_smoke.py --regression                  # run built-in regression suite
"""
import argparse
import json
import time
import sys
from pathlib import Path

_SRC_ROOT = (Path(__file__).resolve().parents[1] / "src").resolve()
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


# ---- Regression test cases ----
_REGRESSION_CASES = [
    {
        "name": "on_topic_water_backup",
        "question": "Does the policy cover water backup or sewer/drain overflow? If yes, what are the key exclusions?",
        "expect_blocked": False,
        "expect_relevance": ["HIGH"],
    },
    {
        "name": "off_topic_sourdough",
        "question": "What is the best recipe for sourdough bread and how long should I let the dough rise?",
        "expect_blocked": True,
        "expect_relevance": ["NONE", "LOW"],
    },
    {
        "name": "borderline_car_collision",
        "question": "Does my policy cover damage to my car parked in the driveway during a hailstorm, and what is the collision deductible?",
        "expect_blocked": True,
        "expect_relevance": ["LOW", "NONE"],
    },
]


def _run_single(question: str, state_code: str) -> dict:
    """Run build_graph() end-to-end and return the final state."""
    from client.graph import build_graph, GraphState

    graph = build_graph()
    graph_input: GraphState = {
        "question": question,
        "state": state_code,
        "require_citations": True,
    }
    return graph.invoke(graph_input)


def run_once(question: str, state_code: str) -> int:
    """Run a single question through the full graph. Returns exit code."""
    t0 = time.perf_counter()
    print(f"question={question!r}", flush=True)
    print(f"state={state_code}", flush=True)

    out = _run_single(question, state_code)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    blocked = bool(out.get("blocked", False))
    relevance = out.get("relevance_rating")
    answer = (out.get("answer") or "").strip()
    validation = out.get("validation") if isinstance(out.get("validation"), dict) else {}

    print(f"duration_ms={dt_ms:.1f}")
    print(f"blocked={blocked}")
    print(f"relevance_rating={relevance}")
    print(f"answer_chars={len(answer)}")
    print("validation=\n" + json.dumps(validation, indent=2))
    if answer:
        print("answer_preview=\n" + answer[:1200])

    if blocked:
        return 2
    if not answer:
        return 3
    return 0


def run_regression(state_code: str) -> int:
    """Run all regression cases and report pass/fail."""
    results = []
    for case in _REGRESSION_CASES:
        name = case["name"]
        question = case["question"]
        expect_blocked = case["expect_blocked"]
        expect_relevance = case["expect_relevance"]

        print(f"\n{'='*60}")
        print(f"REGRESSION: {name}")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        out = _run_single(question, state_code)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        blocked = bool(out.get("blocked", False))
        relevance = out.get("relevance_rating")
        answer = (out.get("answer") or "").strip()

        passed = True
        notes = []

        # Check blocked expectation
        if expect_blocked is not None and blocked != expect_blocked:
            passed = False
            notes.append(f"expected blocked={expect_blocked}, got blocked={blocked}")

        # Check relevance rating
        if relevance and relevance not in expect_relevance:
            passed = False
            notes.append(f"expected relevance in {expect_relevance}, got {relevance}")

        status = "PASS" if passed else "FAIL"
        results.append({"name": name, "passed": passed, "notes": notes})

        print(f"  blocked={blocked}, relevance={relevance}, answer_chars={len(answer)}, {dt_ms:.0f}ms")
        print(f"  => {status}" + (f" ({'; '.join(notes)})" if notes else ""))

    # Summary
    print(f"\n{'='*60}")
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    print(f"REGRESSION SUMMARY: {passed}/{total} passed, {failed} failed")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['name']}" + (f" — {'; '.join(r['notes'])}" if r['notes'] else ""))
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full production LangGraph pipeline and report pass/fail."
    )
    parser.add_argument(
        "--question",
        default="Does the policy cover water backup or sewer/drain overflow? If yes, what are the key exclusions?",
        help="Question to ask (single mode)",
    )
    parser.add_argument("--state", default="IL", help="Jurisdiction state code")
    parser.add_argument(
        "--regression", action="store_true",
        help="Run built-in regression suite (sourdough, car, on-topic)"
    )
    args = parser.parse_args()

    if args.regression:
        return run_regression(args.state)
    else:
        return run_once(args.question, args.state)


if __name__ == "__main__":
    raise SystemExit(main())
