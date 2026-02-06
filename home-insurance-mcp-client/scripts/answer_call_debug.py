import asyncio
import os
import time

import sys
from pathlib import Path

from openai import OpenAI

_SRC_ROOT = (Path(__file__).resolve().parents[1] / "src").resolve()
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from client.mcp_client import retrieve_clauses
from client.prompts import SYSTEM_PROMPT, USER_TEMPLATE


def main() -> int:
    q = "Does the policy cover water backup or sewer/drain overflow? If yes, what are the key exclusions?"

    print("retrieving...", flush=True)
    data = asyncio.run(retrieve_clauses(q, top_k=5))
    results = data.get("results", [])
    print(f"got_results={len(results)}", flush=True)

    lines: list[str] = []
    tags: list[str] = []

    for r in results:
        page_number = r.get("page_number")
        if page_number is not None:
            cite = f"[{r.get('file_name')} | {r.get('doc_type')} | p. {page_number} | chunk {r.get('chunk_index')}]"
        else:
            cite = f"[{r.get('file_name')} | {r.get('doc_type')} | chunk {r.get('chunk_index')}]"
        tags.append(cite)
        snippet = (r.get("snippet") or "").replace("\n", " ").strip()
        lines.append(f"- {cite} {snippet}")

    sources = "\n".join(lines) if lines else "(no sources found)"

    prompt = USER_TEMPLATE.format(
        question=q,
        state_code="IL",
        sources=sources,
        require_citations=True,
    )
    allowed_block = "\n".join(f"- {t}" for t in sorted(set(tags)))
    prompt += "\n\nAllowed citation tags:\n" + allowed_block + "\n"

    print(f"prompt_chars={len(prompt)}", flush=True)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""), timeout=20.0, max_retries=0)

    print("calling_chat...", flush=True)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        timeout=20.0,
    )
    dt = time.perf_counter() - t0
    text = resp.choices[0].message.content or ""

    print(f"chat_seconds={dt:.2f}", flush=True)
    print(f"out_chars={len(text)}", flush=True)
    print(text[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
