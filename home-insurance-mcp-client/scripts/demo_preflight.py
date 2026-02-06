import asyncio
import json
import sys
from pathlib import Path


_SRC_ROOT = (Path(__file__).resolve().parents[1] / "src").resolve()
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from client import mcp_client


def _print_json(title: str, payload: dict) -> None:
    print(f"\n== {title} ==")
    print(json.dumps(payload, indent=2))


def main() -> int:
    required_failures: list[str] = []
    warnings: list[str] = []

    health = asyncio.run(mcp_client.health())
    _print_json("health", health)
    if health.get("status") != "ok":
        required_failures.append("MCP server health is not ok")

    status = asyncio.run(mcp_client.index_status())
    _print_json("index_status", status)

    if status.get("status") != "ok":
        required_failures.append("Qdrant/index_status is not ok")

    if not status.get("collection_exists"):
        required_failures.append("Qdrant collection does not exist (need indexing)")

    points_count = status.get("points_count")
    if isinstance(points_count, int) and points_count <= 0:
        required_failures.append("Qdrant collection has 0 points (need indexing)")

    if not status.get("openai_configured"):
        required_failures.append("OPENAI_API_KEY is not configured on the server")

    if status.get("openai_ok") is False:
        required_failures.append("OPENAI key validation failed (openai_ok=false)")

    if status.get("docs_root_exists") is False:
        required_failures.append("MCP_DOCS_ROOT does not exist on the server")

    if status.get("openai_key_matches_env_file") is False:
        warnings.append("Server is not using the same OPENAI_API_KEY as its .env file (env precedence mismatch)")

    if status.get("ocr_fallback_enabled"):
        if status.get("pymupdf_available") is False:
            warnings.append("OCR fallback enabled but PyMuPDF (fitz) is not available")
        if status.get("tesseract_available") is False:
            warnings.append("OCR fallback enabled but Tesseract is not available on PATH")

    print("\n== Summary ==")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")

    if required_failures:
        print("\nFAIL (not demo-ready):")
        for f in required_failures:
            print(f"- {f}")
        return 2

    print("PASS (demo-ready)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
