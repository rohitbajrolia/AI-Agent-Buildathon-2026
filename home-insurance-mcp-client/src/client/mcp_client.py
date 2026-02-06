import os
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv(override=True)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:4200/mcp/")


def _ensure_trailing_slash(url: str) -> str:
    return url if url.endswith("/") else url + "/"


MCP_SERVER_URL = _ensure_trailing_slash(MCP_SERVER_URL)


@asynccontextmanager
async def mcp_session():
    """
    Creates an MCP client session.
    - streamable_http_client() opens a read/write stream to the MCP server URL.
    - ClientSession wraps those streams and gives us list_tools/call_tool.
    """
    async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


def _extract_text_payload(result) -> str | None:
    for part in result.content:
        if isinstance(part, types.TextContent):
            return part.text
    return None


async def health() -> dict:
    """Calls the MCP tool health and returns parsed JSON."""
    async with mcp_session() as session:
        result = await session.call_tool("health", {})
        text_payload = _extract_text_payload(result)
        if not text_payload:
            return {}
        import json
        return json.loads(text_payload)


async def retrieve_clauses(query: str, top_k: int = 5, doc_type: str | None = None, file_name: str | None = None) -> dict:
    """
    Calls the MCP tool retrieve_clauses and returns parsed JSON.
    """
    async with mcp_session() as session:
        args = {"query": query, "top_k": top_k}
        if doc_type:
            args["doc_type"] = doc_type
        if file_name:
            args["file_name"] = file_name

        result = await session.call_tool("retrieve_clauses", args)

        text_payload = _extract_text_payload(result)

        if not text_payload:
            return {"results": []}

        import json
        return json.loads(text_payload)


async def ingest_folder(
    folder_path: str,
    max_pages: int = 25,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> dict:
    """Calls the MCP tool ingest_folder and returns parsed JSON."""
    async with mcp_session() as session:
        args = {
            "folder_path": folder_path,
            "max_pages": max_pages,
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
        result = await session.call_tool("ingest_folder", args)
        text_payload = _extract_text_payload(result)
        if not text_payload:
            return {}
        import json
        return json.loads(text_payload)


async def index_folder_qdrant(
    folder_path: str,
    max_pages: int = 25,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> dict:
    """Calls the MCP tool index_folder_qdrant and returns parsed JSON."""
    async with mcp_session() as session:
        args = {
            "folder_path": folder_path,
            "max_pages": max_pages,
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
        result = await session.call_tool("index_folder_qdrant", args)
        text_payload = _extract_text_payload(result)
        if not text_payload:
            return {}
        import json
        return json.loads(text_payload)


async def index_status() -> dict:
    """Calls the MCP tool index_status and returns parsed JSON."""
    async with mcp_session() as session:
        result = await session.call_tool("index_status", {})
        text_payload = _extract_text_payload(result)
        if not text_payload:
            return {}
        import json
        return json.loads(text_payload)


async def normalize_quote_snapshot(
    *,
    raw_text: str = "",
    carrier: str | None = None,
    annual_premium: float | None = None,
    dwelling_limit: float | None = None,
    deductible: float | None = None,
    liability_limit: float | None = None,
) -> dict:
    """Calls the MCP tool normalize_quote_snapshot and returns parsed JSON."""
    async with mcp_session() as session:
        args: dict = {"raw_text": raw_text}
        if carrier is not None:
            args["carrier"] = carrier
        if annual_premium is not None:
            args["annual_premium"] = annual_premium
        if dwelling_limit is not None:
            args["dwelling_limit"] = dwelling_limit
        if deductible is not None:
            args["deductible"] = deductible
        if liability_limit is not None:
            args["liability_limit"] = liability_limit

        result = await session.call_tool("normalize_quote_snapshot", args)
        text_payload = _extract_text_payload(result)
        if not text_payload:
            return {}
        import json
        return json.loads(text_payload)
