# What changed and why

## Bugs that would have broken the old code
| Where | Problem | Fix |
|---|---|---|
| `qdrant_manager.py` | `client.search()` no longer exists in current `qdrant-client` -> every query crashes | `query_points` with dense + BM25 prefetch and **server-side RRF fusion** (1 round trip instead of 2 + Python merge) |
| `qdrant_manager.py` | BM25 sparse vectors created without the **IDF modifier** -> keyword scores meaningless | `modifier=IDF` on the sparse vector config |
| `schemas.py` | `figures: List[Dict[str, Any]]` is rejected by the Gemini Developer API (`additionalProperties`) | Typed `Figure` model; separate small `LLMAnswer` schema for the model |
| `chunker.py` | `from backend.config import ...` (ImportError) and the chunker was never called | Rewritten and wired into the PDF pipeline |
| `config.py` | `CHUNK_SIZE=800` words >> 512-token window of bge-small, so embeddings were truncated | 250 words / 40 overlap (configurable) |
| `main.py` | Temp file named after the upload -> collisions + path traversal; stored name = temp name | `mkstemp`, sanitized original name stored separately |
| `main.py` | Sync pipeline inside `async def` blocked the whole server | `asyncio.to_thread`, bounded by semaphores |
| `nodes.py` | Retry re-ran the *identical* query (pointless); `retry_count` only moved on failure | Each retry changes query + widens `top_k`; clean "not found" fallback |
| `nodes.py` | Any Gemini/JSON error -> HTTP 500 | Retries with backoff, errors become a friendly answer |
| everything | Models loaded at import time (slow start, impossible to test) | Lazy singletons in `services.py` + optional warm-up |

## Quality / accuracy
* **Tables** are extracted as Markdown (your test query asks for exactly this) with their caption ("Statement 7.1"), split by rows with the header repeated.
* **Sections** detected from font size/boldness -> real `section` metadata and better citations.
* **Figures**: icons/logos filtered, duplicates removed, captions captured (so figures are searchable), images served at `/static/extracted_images/...` (it wasn't mounted before).
* **Citations are built server-side** from the chunks the model was given; the model only returns source ids, so it cannot invent pages/documents/image paths.
* Prompt-injection hardening (PDF text is delimited and declared untrusted), dedup before reranking, bounded context size, `query_embed` for query-side embeddings.
* Optional OCR for scanned PDFs (`ENABLE_OCR=true`, needs Tesseract).

## Reliability / ops
* Upload: streamed, type + `%PDF-` magic check, size limit, page limit, per-hash lock (same file twice is indexed once), **rollback on failure** (no half-indexed documents reported as "already_exists"), correct 400/413/422/503 status codes.
* Deterministic chunk ids -> re-indexing is idempotent. Paginated document listing (old code stopped at 10 000 points).
* New endpoints: `GET /api/health`, `DELETE /api/documents/{doc_hash}`. Existing endpoints keep their response shape (`/api/chat` also returns `meta`).
* Request ids + timing logs, no internal error text leaked to clients, CORS configurable (wildcard + credentials was invalid anyway).
* Reranker is optional and degrades gracefully; default `bge-reranker-base` (override with `RERANKER_MODEL`).
* `requirements.txt`: removed unused `docling`, explicit `torch`, and the undeclared `rich`.

## Migrating
1. Keep your existing `.env` (compatible). **Your API key was inside the zip - rotate it.**
2. The default collection is now `academic_pdf_rag_v2` (old data used a different layout). Re-upload your PDFs.
3. `pip install -r requirements.txt` then `python main.py` (or `uvicorn main:app`).
4. Tests: `pip install -r requirements-dev.txt && pytest` (no model downloads or API key needed).
5. Real end-to-end: `python scripts/smoke_test.py your.pdf "your question"`.
