import asyncio
import logging
import json
import os

# ClientSession wraps the protocol once we have transport streams.
from mcp import ClientSession

import mcp.types as types
from pathlib import Path

# Streamable HTTP transport for MCP.
from mcp.client.streamable_http import streamable_http_client


logger = logging.getLogger(__name__)

# Default to the repo-level docs folder (override with DOCS_DIR if you want).
docs_dir = Path(os.getenv("DOCS_DIR", Path(__file__).resolve().parents[2] / "docs"))
docs_dir = docs_dir.resolve()

# Index marker: after the first successful index, later smoke-test runs can skip re-indexing.
# Delete this marker (or set REINDEX=1) to force a rebuild.
index_marker = Path(os.getenv("INDEX_MARKER", Path(__file__).resolve().parents[1] / ".qdrant_indexed"))

# Supported extensions (server supports these)
supported_ext = {".pdf", ".png", ".jpg", ".jpeg"}


async def main() -> int:
    # MCP endpoint (server mounts MCP at /mcp).
    server_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:4200/mcp/")

    preflight_error: str | None = None

    # Transport yields read/write streams (plus optional metadata).
    async with streamable_http_client(server_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            # 1) MCP handshake
            await session.initialize()

            # 2) Optional: list tools
            tools_resp = await session.list_tools()
            logger.info("Connected to MCP server. Tools exposed:")
            for tool in tools_resp.tools:
                logger.info("- %s: %s", tool.name, tool.description)

            # 3) Health check first; abort if it fails.
            health_result = await session.call_tool("health", {})
            logger.info("health() result content:")
            for part in health_result.content:
                if isinstance(part, types.TextContent):
                    logger.info("%s", part.text)
                else:
                    logger.info("[%s content]", part.type)

            # 3.5) Preflight: require Qdrant/index and a usable OpenAI key.
            status_result = await session.call_tool("index_status", {})
            status_payload = None
            for part in status_result.content:
                if isinstance(part, types.TextContent):
                    logger.info("index_status() result:\n%s", part.text)
                    try:
                        status_payload = json.loads(part.text)
                    except Exception:
                        status_payload = None

            if isinstance(status_payload, dict):
                if status_payload.get("status") != "ok":
                    preflight_error = "Qdrant is not reachable (index_status != ok)."
                elif not status_payload.get("openai_configured"):
                    preflight_error = "OPENAI_API_KEY is not configured on the server."
                elif status_payload.get("openai_ok") is False:
                    preflight_error = "OPENAI_API_KEY is present but invalid (openai_ok=false)."

            if preflight_error:
                logger.error("Preflight failed: %s", preflight_error)
                return 2
            
            # 4) Index docs into Qdrant (RAG-ready)
            # Default is "once": index on the first run, then skip on later runs.
            # If you want to reindex, set REINDEX=1.
            force_reindex = os.getenv("REINDEX", "0") == "1"
            index_once_enabled = os.getenv("INDEX_ONCE", "1") == "1"
            do_index = force_reindex or (index_once_enabled and not index_marker.exists())

            if do_index:
                logger.info("Indexing folder into Qdrant: %s", str(docs_dir))
                idx = await session.call_tool("index_folder_qdrant", {"folder_path": str(docs_dir)})
                for part in idx.content:
                    if isinstance(part, types.TextContent):
                        logger.info("index_folder_qdrant result:\n%s", part.text)

                # Write a marker so the next run can skip indexing.
                if not force_reindex and index_once_enabled:
                    try:
                        index_marker.write_text(
                            json.dumps(
                                {
                                    "docs_dir": str(docs_dir),
                                    "server_url": server_url,
                                    "note": "Index completed. Delete this file or set REINDEX=1 to reindex.",
                                },
                                indent=2,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                    except Exception as e:
                        logger.warning("Failed to write index marker %s: %s", str(index_marker), e)
            else:
                if index_marker.exists() and index_once_enabled and not force_reindex:
                    logger.info(
                        "Skipping indexing (already indexed once: %s). Set REINDEX=1 to reindex.",
                        str(index_marker),
                    )
                else:
                    logger.info("Skipping indexing (set REINDEX=1 to reindex)")
            # 5) Retrieval test (grounding)
            logger.info("Retrieving clauses...")
            res = await session.call_tool(
                "retrieve_clauses",
                {"query": "Does the policy cover water backup or drainage system issues?", "top_k": 5}
            )
            for part in res.content:
                if isinstance(part, types.TextContent):
                    logger.info("retrieve_clauses result:\n%s", part.text)
                else:
                    logger.info("[%s content]", part.type)

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # Run the async entry point.
    raise SystemExit(asyncio.run(main()))
