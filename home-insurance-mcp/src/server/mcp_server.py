from __future__ import annotations

import os
import time
import json
import contextlib
import uuid
import re
import asyncio
import threading
import math
from typing import AsyncIterator

from dotenv import load_dotenv, dotenv_values
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send
import uvicorn

from pathlib import Path
from pypdf import PdfReader
from PIL import Image
import pytesseract

# -----------------------------
# MCP tools live here
# -----------------------------

app = Server("home-insurance-mcp")

ENV_PATH = (Path(__file__).resolve().parents[2] / ".env").resolve()

# Load env from a local .env file.
# Security first: keep real keys in `.env` and out of git.
load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "home_insurance_docs")

if not OPENAI_API_KEY:
    # Server can still boot for UI wiring, but anything that needs embeddings will fail.
    pass

openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0, max_retries=1)
qdrant = QdrantClient(url=QDRANT_URL, check_compatibility=False)


# -----------------------------
# Simple in-memory job tracker
# (start job -> poll status)
# -----------------------------

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()

_TICKETS: dict[str, dict] = {}
_TICKETS_LOCK = threading.Lock()

_DEMO_STATE_DIR = (Path(__file__).resolve().parents[3] / ".demo_state").resolve()
_TICKETS_STORE_PATH = Path(os.getenv("MCP_TICKETS_STORE", str(_DEMO_STATE_DIR / "handoff_tickets.jsonl"))).resolve()


def _tickets_store_append(record: dict) -> None:
    try:
        _TICKETS_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _TICKETS_STORE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Persistence is best-effort; if disk write fails, we still keep going.
        return


def _tickets_store_load() -> None:
    if not _TICKETS_STORE_PATH.exists():
        return
    try:
        with _TICKETS_STORE_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                raw = (line or "").strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                tid = rec.get("ticket_id")
                if not tid:
                    continue
                with _TICKETS_LOCK:
                    _TICKETS[str(tid)] = rec
    except Exception:
        return


def _ticket_create(*, payload: dict) -> dict:
    ticket_id = str(uuid.uuid4())
    record = {
        "ticket_id": ticket_id,
        "created_unix": int(time.time()),
        "payload": payload,
    }
    with _TICKETS_LOCK:
        _TICKETS[ticket_id] = record
    _tickets_store_append(record)
    return record


def _ticket_list(*, limit: int = 20) -> list[dict]:
    with _TICKETS_LOCK:
        items = list(_TICKETS.values())
    items.sort(key=lambda r: int(r.get("created_unix") or 0), reverse=True)
    return items[: max(1, min(int(limit), 100))]


def _job_create(*, kind: str, params: dict) -> str:
    job_id = str(uuid.uuid4())
    record = {
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "message": "queued",
        "params": params,
        "created_unix": int(time.time()),
        "started_unix": None,
        "finished_unix": None,
        "progress": {},
        "result": None,
        "error": None,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = record
    return job_id


def _job_update(job_id: str, **fields) -> None:
    with _JOBS_LOCK:
        rec = _JOBS.get(job_id)
        if not rec:
            return
        for k, v in fields.items():
            if k == "progress" and isinstance(v, dict):
                rec["progress"].update(v)
            else:
                rec[k] = v


def _job_get(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        rec = _JOBS.get(job_id)
        return dict(rec) if rec else None


def _list_supported_files(folder_path: Path) -> list[Path]:
    supported_ext = {".pdf", ".png", ".jpg", ".jpeg"}
    return [p for p in folder_path.rglob("*") if p.is_file() and p.suffix.lower() in supported_ext]


def _estimate_total_chunks(*, files: list[Path], folder_path: Path, max_pages: int, chunk_size: int, overlap: int) -> int:
    total = 0
    for p in files:
        try:
            ext = p.suffix.lower()
            if ext == ".pdf":
                pages = read_pdf_pages(p, max_pages=max_pages)
                for page_text in pages:
                    total += len(chunk_text(page_text, chunk_size=chunk_size, overlap=overlap))
            else:
                text = read_image_text(p)
                total += len(chunk_text(text, chunk_size=chunk_size, overlap=overlap))
        except Exception:
            continue
    return total


def _run_ingest_job(*, job_id: str, folder_path: Path, max_pages: int, chunk_size: int, overlap: int) -> None:
    _job_update(job_id, status="running", message="ingesting", started_unix=int(time.time()))

    files = _list_supported_files(folder_path)
    summaries = []
    errors = []
    chunks_total = 0

    _job_update(job_id, progress={"files_total": len(files), "files_done": 0})

    for i, p in enumerate(files, start=1):
        try:
            ext = p.suffix.lower()
            if ext == ".pdf":
                text = read_pdf_text(p, max_pages=max_pages)
            else:
                text = read_image_text(p)

            doc_type = classify_doc(text)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            chunks_total += len(chunks)

            rel = str(p.relative_to(folder_path)).replace("\\", "/")
            summaries.append(
                {
                    "file_name": rel,
                    "doc_type": doc_type,
                    "text_chars": len(text),
                    "chunks_count": len(chunks),
                }
            )
        except Exception as e:
            errors.append(
                {
                    "file_name": str(p.relative_to(folder_path)).replace("\\", "/"),
                    "error": _redact_error_text(e),
                }
            )

        _job_update(
            job_id,
            message=f"ingesting ({i}/{len(files)})",
            progress={"files_done": i, "chunks_total": chunks_total},
        )

    payload = {
        "folder_path": str(folder_path),
        "docs_root": str(DOCS_ROOT),
        "files_total": len(files),
        "summaries": summaries,
        "errors": errors,
        "chunks_total": chunks_total,
    }

    _job_update(job_id, status="completed", message="done", result=payload, finished_unix=int(time.time()))


def _run_index_job(*, job_id: str, folder_path: Path, max_pages: int, chunk_size: int, overlap: int, batch_size: int) -> None:
    _job_update(job_id, status="running", message="estimating chunks", started_unix=int(time.time()))

    files = _list_supported_files(folder_path)
    _job_update(job_id, progress={"files_total": len(files), "files_done": 0})

    total_chunks_est = _estimate_total_chunks(
        files=files,
        folder_path=folder_path,
        max_pages=max_pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    batches_total = int(math.ceil(total_chunks_est / batch_size)) if total_chunks_est > 0 else 0

    _job_update(
        job_id,
        message="indexing",
        progress={
            "chunks_total_est": total_chunks_est,
            "batch_size": batch_size,
            "batches_total": batches_total,
            "batches_done": 0,
            "points_upserted": 0,
        },
    )

    points_buffer: list[PointStruct] = []
    total_chunks_indexed = 0
    errors = []
    files_ok = 0
    batches_done = 0

    def _flush() -> None:
        nonlocal batches_done, total_chunks_indexed
        if not points_buffer:
            return
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=list(points_buffer))
        total_chunks_indexed += len(points_buffer)
        batches_done += 1
        points_buffer.clear()
        _job_update(
            job_id,
            message=f"upserting batches ({batches_done}/{batches_total or '?'})",
            progress={"batches_done": batches_done, "points_upserted": total_chunks_indexed},
        )

    for file_i, p in enumerate(files, start=1):
        try:
            ext = p.suffix.lower()
            rel = str(p.relative_to(folder_path)).replace("\\", "/")

            if ext == ".pdf":
                pages = read_pdf_pages(p, max_pages=max_pages)
                full_text = "\n".join(pages)
                doc_type = classify_doc(full_text)

                running_chunk_index = 0
                for page_i, page_text in enumerate(pages, start=1):
                    page_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
                    vectors = embed_texts(page_chunks)

                    if vectors:
                        ensure_collection(QDRANT_COLLECTION, vector_size=len(vectors[0]))

                    for chunk_in_page, (chunk, vec) in enumerate(zip(page_chunks, vectors)):
                        stable = f"{rel}::p{page_i}::c{chunk_in_page}"
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable))
                        payload = {
                            "file_name": rel,
                            "doc_type": doc_type,
                            "page_number": page_i,
                            "chunk_index": running_chunk_index,
                            "text": chunk,
                        }
                        points_buffer.append(PointStruct(id=point_id, vector=vec, payload=payload))
                        running_chunk_index += 1

                        if len(points_buffer) >= batch_size:
                            _flush()

            else:
                text = read_image_text(p)
                doc_type = classify_doc(text)
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                vectors = embed_texts(chunks)

                if vectors:
                    ensure_collection(QDRANT_COLLECTION, vector_size=len(vectors[0]))

                for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                    stable = f"{rel}::{i}"
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable))
                    payload = {
                        "file_name": rel,
                        "doc_type": doc_type,
                        "chunk_index": i,
                        "text": chunk,
                    }
                    points_buffer.append(PointStruct(id=point_id, vector=vec, payload=payload))
                    if len(points_buffer) >= batch_size:
                        _flush()

            files_ok += 1
        except Exception as e:
            errors.append({"file_name": str(p.relative_to(folder_path)).replace("\\", "/"), "error": _redact_error_text(e)})

        _job_update(job_id, message=f"indexing files ({file_i}/{len(files)})", progress={"files_done": file_i})

    # Final flush
    try:
        _flush()
    except Exception as e:
        errors.append({"file_name": "(flush)", "error": _redact_error_text(e)})

    payload = {
        "collection": QDRANT_COLLECTION,
        "docs_root": str(DOCS_ROOT),
        "files_indexed": files_ok,
        "files_total": len(files),
        "chunks_indexed": total_chunks_indexed,
        "errors": errors,
        "progress": {
            "chunks_total_est": total_chunks_est,
            "batch_size": batch_size,
            "batches_total": batches_total,
            "batches_done": batches_done,
            "points_upserted": total_chunks_indexed,
        },
    }

    _job_update(job_id, status="completed", message="done", result=payload, finished_unix=int(time.time()))


_OPENAI_STATUS_CACHE_TS: float = 0.0
_OPENAI_STATUS_CACHE_OK: bool | None = None
_OPENAI_STATUS_CACHE_ERROR: str | None = None


DOCS_ROOT = Path(
    os.getenv("MCP_DOCS_ROOT", str((Path(__file__).resolve().parents[3] / "docs").resolve()))
).resolve()


def _resolve_docs_folder(folder_path: str) -> Path:
    p = Path(folder_path).expanduser()
    try:
        resolved = p.resolve()
    except Exception:
        resolved = p.absolute()

    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"Folder not found or not a directory: {resolved}")

    try:
        resolved.relative_to(DOCS_ROOT)
    except Exception:
        if resolved != DOCS_ROOT:
            raise ValueError(
                f"Folder must be under docs root: {DOCS_ROOT}. Got: {resolved}"
            )

    return resolved


def _parse_amount(value: str) -> float | None:
    v = (value or "").strip()
    if not v:
        return None
    v = v.replace(",", "")
    v = v.replace("$", "")
    try:
        return float(v)
    except Exception:
        return None


def _extract_amount(raw_text: str, keywords: list[str]) -> float | None:
    t = raw_text or ""
    for kw in keywords:
        m = re.search(rf"{re.escape(kw)}[^\n\r$0-9]*\$?\s*([0-9][0-9,]*(?:\.[0-9]{{1,2}})?)", t, flags=re.IGNORECASE)
        if m:
            amt = _parse_amount(m.group(1))
            if amt is not None:
                return amt
    return None


def _redact_error_text(message: object) -> str:
    m = str(message or "")
    # Redact OpenAI-style secret keys if they ever appear in exception text.
    m = re.sub(r"sk-[A-Za-z0-9_-]{10,}", "sk-***", m)
    return m


def _key_last4(value: str) -> str | None:
    v = (value or "").strip()
    if len(v) < 4:
        return None
    return v[-4:]


def read_pdf_text(file_path: Path, max_pages: int = 25) -> str:
    """
    Read text from a PDF using pypdf.
    max_pages keeps runtime bounded for large PDFs.
    """
    return "\n".join(read_pdf_pages(file_path, max_pages=max_pages))


def read_pdf_pages(file_path: Path, max_pages: int = 25) -> list[str]:
    """Read text from a PDF, page by page.

    We keep page numbers so citations can reference an exact page (e.g., "p. 12").
    """
    reader = PdfReader(str(file_path))

    enable_ocr = os.getenv("ENABLE_PDF_OCR_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "y"}
    fitz = None
    if enable_ocr:
        try:
            import fitz  # type: ignore
        except Exception:
            fitz = None

    def _ocr_page(page_index: int) -> str:
        if not fitz:
            return ""
        try:
            doc = fitz.open(str(file_path))
            try:
                page = doc.load_page(page_index)
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                return pytesseract.image_to_string(img) or ""
            finally:
                doc.close()
        except Exception:
            return ""

    pages: list[str] = []
    for page_i, page in enumerate(reader.pages[:max_pages]):
        extracted = ""
        try:
            extracted = page.extract_text() or ""
        except Exception:
            extracted = ""

        if extracted.strip():
            pages.append(extracted)
            continue

        # Likely scanned/image-only page.
        ocr_text = _ocr_page(page_i) if enable_ocr else ""
        pages.append(ocr_text)

    return pages

def read_image_text(file_path: Path) -> str:
    """
    OCR an image using pytesseract.

    If Tesseract isn't installed, this will raise an error.
    """
    try:
        img = Image.open(file_path)
    except Exception as e:
        raise ValueError(f"Failed to open image: {file_path}. Error: {e}")
    
    try:
        return pytesseract.image_to_string(img)
    except Exception as e:
        raise ValueError(
            "OCR failed. Tesseract engine may not be installed or not in PATH. "
            f"Original error: {e}"
        )


def classify_doc(text: str) -> str:
    """
    Heuristic classifier for insurance docs.
    Priority matters: we check strong signals first.
    """
    t = (text or "").lower()

    # Strong "policy booklet" signals
    booklet_signals = [
        "section i", "section ii", "definitions", "conditions", "exclusions",
        "duties after loss", "loss settlement", "property coverages",
        "liability coverages", "policy jacket"
    ]
    if sum(1 for s in booklet_signals if s in t) >= 3:
        return "policy_booklet"

    # Declarations signals
    decl_signals = [
        "declarations", "policy period", "named insured", "mailing address",
        "effective", "expiration", "deductible", "coverage a", "coverage b",
        "coverage c", "coverage d", "premium"
    ]
    if sum(1 for s in decl_signals if s in t) >= 3:
        return "declarations"

    # Endorsement signals
    # (endorsements often include "this endorsement changes the policy")
    endorse_signals = [
        "endorsement", "this endorsement", "changes the policy", "forms and endorsements",
        "amends", "modifies"
    ]
    if sum(1 for s in endorse_signals if s in t) >= 2:
        return "endorsement"

    return "unknown"



def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """
    Simple chunker for later RAG indexing.
    chunk_size and overlap are a trade-off:
    - bigger chunks = fewer embeddings but less precise citations
    - overlap helps preserve context across boundaries
    """
    text = text or ""
    if not text.strip():
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap  # move back a bit to overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Turns text chunks into vectors using OpenAI embeddings.
    """
    # Filter empty strings to avoid wasted calls
    cleaned = [t for t in texts if t and t.strip()]
    if not cleaned:
        return []

    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=cleaned,
    )
    return [item.embedding for item in resp.data]

def ensure_collection(collection_name: str, vector_size: int):
    """
    Create collection once (or do nothing if it exists).
    """
    existing = [c.name for c in qdrant.get_collections().collections]
    if collection_name in existing:
        return

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )




@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """
    MCP tool dispatcher.
    This is the switchboard: tool calls come in by `name` with JSON `arguments`.
    """
    if name == "health":
        payload = {
            "status": "ok",
            "server": "home-insurance-mcp",
            "unix_time": int(time.time()),
            "note": "MCP server running (streamable HTTP, stateless)",
        }
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]
    
    if name == "normalize_quote_snapshot":
        raw = (arguments.get("raw_text") or "").strip()

        carrier = arguments.get("carrier")
        annual_premium = arguments.get("annual_premium")
        dwelling_limit = arguments.get("dwelling_limit")
        deductible = arguments.get("deductible")
        liability_limit = arguments.get("liability_limit")

        if annual_premium is None:
            annual_premium = _extract_amount(raw, ["annual premium", "total premium", "premium"])
        if dwelling_limit is None:
            dwelling_limit = _extract_amount(raw, ["coverage a", "dwelling", "dwelling limit"])
        if deductible is None:
            deductible = _extract_amount(raw, ["deductible", "all other perils deductible"])
        if liability_limit is None:
            liability_limit = _extract_amount(raw, ["liability", "personal liability", "coverage e"])

        payload = {
            "kind": "quote_snapshot",
            "extracted": {
                "carrier": carrier,
                "annual_premium": annual_premium,
                "dwelling_limit": dwelling_limit,
                "deductible": deductible,
                "liability_limit": liability_limit,
            },
            "raw_hint": raw[:500],
        }
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

    if name == "create_handoff_ticket":
        question = (arguments.get("question") or "").strip()
        state = (arguments.get("state") or "").strip()
        answer = (arguments.get("answer") or "").strip()
        sources = (arguments.get("sources") or "").strip()
        run_id = (arguments.get("run_id") or "").strip() or None
        retrieved_matches = arguments.get("retrieved_matches")
        notes = (arguments.get("notes") or "").strip() or None

        if not question:
            raise ValueError("question is required")
        if not state:
            raise ValueError("state is required")
        if not answer:
            raise ValueError("answer is required")

        def _truncate(s: str, n: int) -> str:
            return s if len(s) <= n else (s[:n] + "...")

        record = _ticket_create(
            payload={
                "kind": "handoff_ticket",
                "run_id": run_id,
                "question": _truncate(question, 2000),
                "state": state,
                "answer": _truncate(answer, 8000),
                "sources": _truncate(sources, 12000),
                "retrieved_matches": retrieved_matches if isinstance(retrieved_matches, list) else None,
                "notes": notes,
            }
        )
        return [types.TextContent(type="text", text=json.dumps(record, indent=2))]

    if name == "list_handoff_tickets":
        limit = int(arguments.get("limit") or 20)
        tickets = _ticket_list(limit=limit)
        payload = {
            "count": len(tickets),
            "tickets": tickets,
        }
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

    if name == "start_ingest_job":
        folder_path = _resolve_docs_folder(arguments["folder_path"])
        max_pages = int(arguments.get("max_pages", 25))
        chunk_size = int(arguments.get("chunk_size", 1200))
        overlap = int(arguments.get("overlap", 150))

        job_id = _job_create(
            kind="ingest",
            params={
                "folder_path": str(folder_path),
                "max_pages": max_pages,
                "chunk_size": chunk_size,
                "overlap": overlap,
            },
        )

        async def _runner():
            try:
                await asyncio.to_thread(
                    _run_ingest_job,
                    job_id=job_id,
                    folder_path=folder_path,
                    max_pages=max_pages,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            except Exception as e:
                _job_update(job_id, status="failed", message="failed", error=_redact_error_text(e), finished_unix=int(time.time()))

        asyncio.create_task(_runner())
        return [types.TextContent(type="text", text=json.dumps({"job_id": job_id}, indent=2))]

    if name == "start_index_job":
        folder_path = _resolve_docs_folder(arguments["folder_path"])
        max_pages = int(arguments.get("max_pages", 25))
        chunk_size = int(arguments.get("chunk_size", 1200))
        overlap = int(arguments.get("overlap", 150))
        batch_size = int(arguments.get("batch_size", 64))

        job_id = _job_create(
            kind="index",
            params={
                "folder_path": str(folder_path),
                "max_pages": max_pages,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "batch_size": batch_size,
            },
        )

        async def _runner():
            try:
                await asyncio.to_thread(
                    _run_index_job,
                    job_id=job_id,
                    folder_path=folder_path,
                    max_pages=max_pages,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    batch_size=batch_size,
                )
            except Exception as e:
                _job_update(job_id, status="failed", message="failed", error=_redact_error_text(e), finished_unix=int(time.time()))

        asyncio.create_task(_runner())
        return [types.TextContent(type="text", text=json.dumps({"job_id": job_id}, indent=2))]

    if name == "job_status":
        job_id = (arguments.get("job_id") or "").strip()
        if not job_id:
            return [types.TextContent(type="text", text=json.dumps({"status": "error", "error": "job_id is required"}, indent=2))]
        rec = _job_get(job_id)
        if not rec:
            return [types.TextContent(type="text", text=json.dumps({"status": "error", "error": "job not found", "job_id": job_id}, indent=2))]
        return [types.TextContent(type="text", text=json.dumps(rec, indent=2))]
    
    if name == "ingest_folder":
        folder_path = _resolve_docs_folder(arguments["folder_path"])
        max_pages = int(arguments.get("max_pages", 25))
        chunk_size = int(arguments.get("chunk_size", 1200))
        overlap = int(arguments.get("overlap", 150))

        supported_ext = {".pdf", ".png", ".jpg", ".jpeg"}
        files = [p for p in folder_path.rglob("*") if p.is_file() and p.suffix.lower() in supported_ext]

        summaries = []
        errors = []

        for p in files:
            try:
                ext = p.suffix.lower()
                if ext == ".pdf":
                    text = read_pdf_text(p, max_pages=max_pages)
                else:
                    text = read_image_text(p)

                doc_type = classify_doc(text)
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

                rel = str(p.relative_to(folder_path)).replace("\\", "/")

                summaries.append({
                    "file_name": rel,
                    "doc_type": doc_type,
                    "text_chars": len(text),
                    "chunks_count": len(chunks),
                })
            except Exception as e:
                errors.append({"file_name": str(p.relative_to(folder_path)).replace("\\", "/"), "error": _redact_error_text(e)})

        payload = {
            "folder_path": str(folder_path),
            "docs_root": str(DOCS_ROOT),
            "files_total": len(files),
            "summaries": summaries,
            "errors": errors,
        }

        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]
    
    if name == "index_folder_qdrant":
        folder_path = _resolve_docs_folder(arguments["folder_path"])
        max_pages = int(arguments.get("max_pages", 25))
        chunk_size = int(arguments.get("chunk_size", 1200))
        overlap = int(arguments.get("overlap", 150))

        supported_ext = {".pdf", ".png", ".jpg", ".jpeg"}
        files = [p for p in folder_path.rglob("*") if p.is_file() and p.suffix.lower() in supported_ext]

        all_points: list[PointStruct] = []
        total_chunks = 0
        errors = []

        for p in files:
            try:
                ext = p.suffix.lower()
                rel = str(p.relative_to(folder_path)).replace("\\", "/")

                if ext == ".pdf":
                    pages = read_pdf_pages(p, max_pages=max_pages)
                    full_text = "\n".join(pages)
                    doc_type = classify_doc(full_text)

                    running_chunk_index = 0
                    for page_i, page_text in enumerate(pages, start=1):
                        page_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
                        vectors = embed_texts(page_chunks)

                        if vectors:
                            ensure_collection(QDRANT_COLLECTION, vector_size=len(vectors[0]))

                        for chunk_in_page, (chunk, vec) in enumerate(zip(page_chunks, vectors)):
                            stable = f"{rel}::p{page_i}::c{chunk_in_page}"
                            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable))
                            payload = {
                                "file_name": rel,
                                "doc_type": doc_type,
                                "page_number": page_i,
                                "chunk_index": running_chunk_index,
                                "text": chunk,
                            }
                            all_points.append(PointStruct(id=point_id, vector=vec, payload=payload))
                            running_chunk_index += 1

                        total_chunks += len(vectors)

                else:
                    text = read_image_text(p)
                    doc_type = classify_doc(text)
                    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                    vectors = embed_texts(chunks)

                    if vectors:
                        ensure_collection(QDRANT_COLLECTION, vector_size=len(vectors[0]))

                    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                        stable = f"{rel}::{i}"
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable))
                        payload = {
                            "file_name": rel,
                            "doc_type": doc_type,
                            "chunk_index": i,
                            "text": chunk,
                        }
                        all_points.append(PointStruct(id=point_id, vector=vec, payload=payload))

                    total_chunks += len(vectors)

            except Exception as e:
                errors.append({"file_name": str(p.relative_to(folder_path)).replace("\\", "/"), "error": _redact_error_text(e)})

        # Upsert in batches (fast + avoids huge request)
        batch_size = 64
        for start in range(0, len(all_points), batch_size):
            qdrant.upsert(
                collection_name=QDRANT_COLLECTION,
                points=all_points[start:start + batch_size],
            )
        payload = {
            "collection": QDRANT_COLLECTION,
            "docs_root": str(DOCS_ROOT),
            "files_indexed": len(files) - len(errors),
            "files_total": len(files),
            "chunks_indexed": total_chunks,
            "errors": errors,
        }
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]
    
    if name == "retrieve_clauses":
        query = arguments["query"]
        top_k = int(arguments.get("top_k", 5))
        doc_type = arguments.get("doc_type")
        file_name = arguments.get("file_name")

        qvec = embed_texts([query])
        if not qvec:
            raise ValueError("Query embedding failed (empty query?)")

        # Optional metadata filters
        must_conditions = []
        if doc_type:
            must_conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
        if file_name:
            must_conditions.append(FieldCondition(key="file_name", match=MatchValue(value=file_name)))
        query_filter = Filter(must=must_conditions) if must_conditions else None

        query_resp = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=qvec[0],
            limit=top_k,
            query_filter=query_filter,
        )
        hits = query_resp.points
        results = []
        for h in hits:
            pl = h.payload or {}
            results.append({
                "score": float(h.score),
                "file_name": pl.get("file_name"),
                "doc_type": pl.get("doc_type"),
                "page_number": pl.get("page_number"),
                "chunk_index": pl.get("chunk_index"),
                "snippet": (pl.get("text") or "")[:800],  # cap output
            })

        return [types.TextContent(type="text", text=json.dumps({"results": results}, indent=2))]

    if name == "index_status":
        started = time.perf_counter()

        qdrant_ok = False
        collection_exists = False
        points_count = None
        error = None

        openai_ok: bool | None = None
        openai_error: str | None = None

        # OCR readiness (scanned PDF fallback)
        ocr_fallback_enabled = os.getenv("ENABLE_PDF_OCR_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "y"}
        pymupdf_available: bool | None = None
        tesseract_available: bool | None = None
        tesseract_error: str | None = None

        if ocr_fallback_enabled:
            try:
                import fitz  # type: ignore
                pymupdf_available = True
            except Exception:
                pymupdf_available = False

            try:
                _ = pytesseract.get_tesseract_version()
                tesseract_available = True
            except Exception as e:
                tesseract_available = False
                tesseract_error = _redact_error_text(e)

        openai_key_last4 = _key_last4(OPENAI_API_KEY)
        env_file_exists = ENV_PATH.exists()
        env_file_openai_last4: str | None = None
        openai_key_matches_env_file: bool | None = None

        if env_file_exists:
            try:
                env_values = dotenv_values(ENV_PATH)
                env_file_key = str(env_values.get("OPENAI_API_KEY") or "").strip()
                if env_file_key:
                    env_file_openai_last4 = _key_last4(env_file_key)
                    if openai_key_last4 is not None and env_file_openai_last4 is not None:
                        openai_key_matches_env_file = (openai_key_last4 == env_file_openai_last4)
            except Exception:
                # Never fail status because a dotenv parse was weird.
                pass

        try:
            collections = qdrant.get_collections().collections
            names = {c.name for c in collections}
            qdrant_ok = True
            collection_exists = QDRANT_COLLECTION in names

            if collection_exists:
                try:
                    cnt = qdrant.count(collection_name=QDRANT_COLLECTION, exact=False)
                    points_count = int(getattr(cnt, "count", 0))
                except Exception:
                    # Fallback for older client/server combos.
                    info = qdrant.get_collection(QDRANT_COLLECTION)
                    points_count = getattr(info, "points_count", None)
        except Exception as e:
            error = _redact_error_text(e)

        # Optional: validate OpenAI credentials with a tiny call.
        # Helps catch invalid keys early.
        check_openai = os.getenv("CHECK_OPENAI_ON_INDEX_STATUS", "1").strip().lower() in {"1", "true", "yes", "y"}
        cache_seconds = float(os.getenv("OPENAI_INDEX_STATUS_CACHE_SECONDS", "60") or "60")
        if check_openai and OPENAI_API_KEY:
            global _OPENAI_STATUS_CACHE_TS, _OPENAI_STATUS_CACHE_OK, _OPENAI_STATUS_CACHE_ERROR
            now = time.time()
            cached_ts = _OPENAI_STATUS_CACHE_TS
            cached_ok = _OPENAI_STATUS_CACHE_OK
            cached_error = _OPENAI_STATUS_CACHE_ERROR

            if cache_seconds > 0 and (now - cached_ts) < cache_seconds and cached_ok is not None:
                openai_ok = bool(cached_ok)
                openai_error = cached_error
            else:
                try:
                    _ = openai_client.embeddings.create(model="text-embedding-3-small", input=["ping"])
                    openai_ok = True
                    openai_error = None
                except Exception as e:
                    openai_ok = False
                    openai_error = _redact_error_text(e)

                _OPENAI_STATUS_CACHE_TS = now
                _OPENAI_STATUS_CACHE_OK = openai_ok
                _OPENAI_STATUS_CACHE_ERROR = openai_error

        payload = {
            "status": "ok" if qdrant_ok else "degraded",
            "qdrant_url": QDRANT_URL,
            "collection": QDRANT_COLLECTION,
            "collection_exists": collection_exists,
            "points_count": points_count,
            "openai_configured": bool(OPENAI_API_KEY),
            "openai_key_last4": openai_key_last4,
            "env_file": str(ENV_PATH),
            "env_file_exists": env_file_exists,
            "openai_key_last4_in_env_file": env_file_openai_last4,
            "openai_key_matches_env_file": openai_key_matches_env_file,
            "openai_ok": openai_ok,
            "openai_error": openai_error,
            "docs_root": str(DOCS_ROOT),
            "docs_root_exists": DOCS_ROOT.exists(),
            "ocr_fallback_enabled": ocr_fallback_enabled,
            "pymupdf_available": pymupdf_available,
            "tesseract_available": tesseract_available,
            "tesseract_error": tesseract_error,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "error": error,
        }
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]


    raise ValueError(f"Unknown tool: {name}")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """
    Tool catalog with input schemas.
    Enables MCP clients to discover available tools and their input schemas.
    """
    return [
            
        types.Tool(
            name="health",
            description="Healthcheck for MCP server.",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="normalize_quote_snapshot",
            description="Normalize key values from a quote/rating snapshot (for deductible what-if scenarios).",
            inputSchema={
                "type": "object",
                "properties": {
                    "carrier": {"type": "string", "description": "Provider name (e.g., Progressive, State Farm)."},
                    "annual_premium": {"type": "number", "description": "Annual premium in USD."},
                    "dwelling_limit": {"type": "number", "description": "Coverage A / dwelling limit in USD."},
                    "deductible": {"type": "number", "description": "Deductible in USD (or main deductible)."},
                    "liability_limit": {"type": "number", "description": "Liability limit in USD."},
                    "raw_text": {"type": "string", "description": "Optional raw text (OCR dump or copied text)."},
                },
                "required": [],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="ingest_folder",
            description="Ingest all supported documents in a folder and return compact summaries per file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder_path": {"type": "string", "description": "Absolute path under the configured docs root."},
                    "max_pages": {"type": "integer", "default": 25},
                    "chunk_size": {"type": "integer", "default": 1200},
                    "overlap": {"type": "integer", "default": 150},
                },
                "required": ["folder_path"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="index_folder_qdrant",
            description="Extract + chunk + embed all docs in a folder, store vectors in Qdrant with metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder_path": {"type": "string", "description": "Absolute path under the configured docs root."},
                    "max_pages": {"type": "integer", "default": 25},
                    "chunk_size": {"type": "integer", "default": 1200},
                    "overlap": {"type": "integer", "default": 150},
                },
                "required": ["folder_path"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="retrieve_clauses",
            description="Semantic search over indexed policy docs. Returns top snippets with citations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "doc_type": {"type": "string", "description": "Optional filter: declarations/endorsement/policy_booklet/unknown"},
                    "file_name": {"type": "string", "description": "Optional filter by file name"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="index_status",
            description="Report Qdrant connectivity and whether the document index collection exists (plus point count when available).",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),

        types.Tool(
            name="create_handoff_ticket",
            description="Create a lightweight handoff ticket (in-memory) containing question, answer, and citations for human review.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "state": {"type": "string", "description": "State / jurisdiction context."},
                    "answer": {"type": "string", "description": "Final grounded answer shown to the user (with citations)."},
                    "sources": {"type": "string", "description": "SOURCES block used to generate the answer."},
                    "run_id": {"type": "string", "description": "Optional client run id for trace correlation."},
                    "retrieved_matches": {"type": "array", "description": "Optional redacted retrieved match summaries."},
                    "notes": {"type": "string", "description": "Optional notes (e.g., safety constraints)."},
                },
                "required": ["question", "state", "answer", "sources"],
                "additionalProperties": False,
            },
        ),

        types.Tool(
            name="list_handoff_tickets",
            description="List recent handoff tickets created in this server session (session-only).",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20, "description": "Max tickets to return."},
                },
                "required": [],
                "additionalProperties": False,
            },
        ),
    ]


# -----------------------------
# Transport wiring (Streamable HTTP, stateless)
# -----------------------------
session_manager = StreamableHTTPSessionManager(
    app=app,
    event_store=None,
    json_response=True,
    stateless=True,
)


async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
    await session_manager.handle_request(scope, receive, send)


@contextlib.asynccontextmanager
async def lifespan(starlette_app: Starlette) -> AsyncIterator[None]:
    async with session_manager.run():
        print("MCP StreamableHTTP session manager started")
        _tickets_store_load()
        try:
            yield
        finally:
            print("MCP server shutting down")


starlette_app = Starlette(
    debug=True,
    routes=[Mount("/mcp", app=handle_streamable_http)],
    lifespan=lifespan,
)


def main() -> None:
    port = int(os.environ.get("MCP_PORT", "4200"))
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    uvicorn.run(starlette_app, host=host, port=port)


if __name__ == "__main__":
    main()
