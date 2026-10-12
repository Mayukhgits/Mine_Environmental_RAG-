import asyncio
import logging
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from hashlib import sha256
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import BASE_DIR, ensure_dirs, settings
from graph.llm import LLMError
from models.schemas import ChatRequest
from services import get_reranker, get_vector_store

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("api")

ensure_dirs()

from graph.workflow import app_graph
from ingestion.pdf_processor import PDFProcessor

pdf_processor = PDFProcessor()
_ingest_sem = asyncio.Semaphore(settings.MAX_CONCURRENT_INGESTIONS)
_chat_sem = asyncio.Semaphore(settings.MAX_CONCURRENT_CHATS)
_hash_locks: Dict[str, asyncio.Lock] = {}
MAX_UPLOAD_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.WARMUP_ON_STARTUP:
        async def warm():
            try:
                await asyncio.to_thread(get_vector_store)
                await asyncio.to_thread(get_reranker)
                log.info("Warm-up complete")
            except Exception:
                log.exception("Warm-up failed (will retry on first request)")

        asyncio.create_task(warm())
    yield


app = FastAPI(title="RAG Pipeline API", lifespan=lifespan)

_wildcard = "*" in settings.CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=not _wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.rid = rid
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("[%s] unhandled error on %s %s", rid, request.method, request.url.path)
        return JSONResponse({"detail": "Internal server error", "request_id": rid}, status_code=500,
                            headers={"X-Request-ID": rid})
    response.headers["X-Request-ID"] = rid
    log.info("[%s] %s %s -> %s (%.0f ms)", rid, request.method, request.url.path,
             response.status_code, (time.perf_counter() - start) * 1000)
    return response


def _service_unavailable(what: str) -> HTTPException:
    return HTTPException(status_code=503, detail=f"{what} is temporarily unavailable. Please retry shortly.")


def _is_backend_down(exc: Exception) -> bool:
    return type(exc).__module__.startswith(("qdrant_client", "httpx", "httpcore")) or isinstance(
        exc, (ConnectionError, TimeoutError)
    )


@app.get("/api/health")
async def health():
    qdrant_ok = True
    try:
        store = await asyncio.to_thread(get_vector_store)
        await asyncio.to_thread(store.client.get_collections)
    except Exception:
        qdrant_ok = False
    body = {"status": "ok" if qdrant_ok else "degraded", "qdrant": qdrant_ok,
            "llm_configured": bool(settings.GEMINI_API_KEY)}
    return JSONResponse(body, status_code=200 if qdrant_ok else 503)


@app.get("/api/documents")
async def get_documents():
    try:
        store = await asyncio.to_thread(get_vector_store)
        return {"documents": await asyncio.to_thread(store.get_all_indexed_documents)}
    except Exception as exc:
        log.exception("listing documents failed")
        raise _service_unavailable("The document index") if _is_backend_down(exc) else HTTPException(500, "Could not list documents")


@app.delete("/api/documents/{doc_hash}")
async def delete_document(doc_hash: str):
    if not re.fullmatch(r"[0-9a-f]{64}", doc_hash):
        raise HTTPException(400, "Invalid document hash")
    store = await asyncio.to_thread(get_vector_store)
    removed = await asyncio.to_thread(store.delete_document, doc_hash)
    if removed == 0:
        raise HTTPException(404, "Document not found")
    return {"status": "deleted", "doc_hash": doc_hash, "chunks_removed": removed}


def _safe_filename(name: str | None) -> str:
    name = os.path.basename((name or "").replace("\\", "/")).strip()
    name = re.sub(r"[^\w.\- ()\[\]]+", "_", name)
    return name[:200] or "document.pdf"


async def _save_upload(file: UploadFile, dest: str) -> str:
    hasher, size, first = sha256(), 0, True
    with open(dest, "wb") as out:
        while chunk := await file.read(1 << 20):
            if first:
                if b"%PDF-" not in chunk[:1024]:
                    raise HTTPException(400, "File is not a valid PDF")
                first = False
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"File too large (limit {settings.MAX_UPLOAD_MB} MB)")
            hasher.update(chunk)
            out.write(chunk)
    if size == 0:
        raise HTTPException(400, "Uploaded file is empty")
    return hasher.hexdigest()


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = _safe_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported")

    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        doc_hash = await _save_upload(file, tmp_path)
        store = await asyncio.to_thread(get_vector_store)

        lock = _hash_locks.setdefault(doc_hash, asyncio.Lock())
        async with lock:
            if await asyncio.to_thread(store.is_document_indexed, doc_hash):
                status, chunk_count = "already_exists", None
            else:
                async with _ingest_sem:
                    chunks = await asyncio.to_thread(pdf_processor.process, tmp_path, filename, doc_hash)
                    chunk_count = await asyncio.to_thread(store.index_chunks, chunks, doc_hash)
                status = "indexed"
        _hash_locks.pop(doc_hash, None)

        return {
            "message": "File processed successfully",
            "filename": filename,
            "doc_hash": doc_hash,
            "status": status,
            "chunks": chunk_count,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        log.exception("upload failed for %s", filename)
        raise _service_unavailable("The document index") if _is_backend_down(exc) else HTTPException(500, "Failed to process the PDF")
    finally:
        await file.close()
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not settings.GEMINI_API_KEY:
        raise HTTPException(503, "The language model is not configured (missing GEMINI_API_KEY)")

    message = request.message.strip()
    if not message:
        raise HTTPException(400, "Message must not be empty")

    state = {
        "original_query": message[: settings.MAX_QUERY_CHARS],
        "rewritten_query": message[: settings.MAX_QUERY_CHARS],
        "conversation_history": request.history,
        "document_ids": request.documents,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "generated_answer": None,
        "validation_passed": False,
        "retry_count": 0,
        "error": None,
    }
    start = time.perf_counter()
    try:
        async with _chat_sem:
            result = await asyncio.to_thread(app_graph.invoke, state)
    except LLMError as exc:
        log.error("LLM failure: %s", exc)
        raise _service_unavailable("The language model") from exc
    except Exception as exc:
        log.exception("chat pipeline failed")
        raise _service_unavailable("The document index") if _is_backend_down(exc) else HTTPException(500, "Failed to generate an answer")

    answer = result.get("generated_answer")
    sources = result.get("reranked_chunks") or result.get("retrieved_chunks") or []
    return {
        "response": answer.model_dump() if answer else None,
        "sources": sources,
        "meta": {
            "rewritten_query": result.get("rewritten_query"),
            "attempts": result.get("retry_count", 0) + (1 if result.get("validation_passed") else 0),
            "grounded": bool(result.get("validation_passed")),
            "latency_ms": round((time.perf_counter() - start) * 1000),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)