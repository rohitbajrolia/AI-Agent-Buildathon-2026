import argparse
import json
import time

import sys
from pathlib import Path


_SRC_ROOT = (Path(__file__).resolve().parents[1] / "src").resolve()
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the client LangGraph pipeline once and report pass/fail.")
    parser.add_argument(
        "--question",
        default="Does the policy cover water backup or sewer/drain overflow? If yes, what are the key exclusions?",
        help="Question to ask",
    )
    parser.add_argument("--state", default="IL", help="Jurisdiction state code")
    args = parser.parse_args()

    from client.graph import GraphState, retrieve_node, validate_node, answer_node, verify_citations_node

    state: GraphState = {
        "question": args.question,
        "state": args.state,
        "require_citations": True,
    }

    t0 = time.perf_counter()

    print("step=retrieve", flush=True)
    state = retrieve_node(state)
    print(f"retrieved={len(state.get('raw_results') or [])}", flush=True)

    print("step=validate", flush=True)
    state = validate_node(state)
    print(f"blocked_after_validate={bool(state.get('blocked'))}", flush=True)

    if state.get("blocked"):
        dt_ms = (time.perf_counter() - t0) * 1000.0
        validation = state.get("validation") if isinstance(state.get("validation"), dict) else {}
        print(f"duration_ms={dt_ms:.1f}")
        print("validation=\n" + json.dumps(validation, indent=2))
        return 2

    print("step=answer", flush=True)
    state = answer_node(state)
    answer = (state.get("answer") or "").strip()
    print(f"answer_chars={len(answer)}", flush=True)

    print("step=citation_verify", flush=True)
    state = verify_citations_node(state)
    blocked = bool(state.get("blocked"))
    validation = state.get("validation") if isinstance(state.get("validation"), dict) else {}

    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"duration_ms={dt_ms:.1f}")
    print(f"blocked={blocked}")
    print("validation=\n" + json.dumps(validation, indent=2))
    print("answer_preview=\n" + answer[:1200])

    if blocked:
        return 2
    if not answer:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
