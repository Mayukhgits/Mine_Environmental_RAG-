"""Lazily-created, process-wide singletons (heavy models are loaded once, on first use
or during application start-up, never at import time)."""
import logging
from functools import lru_cache

from config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_vector_store():
    from vectorstore.qdrant_manager import VectorStoreManager

    return VectorStoreManager()


@lru_cache(maxsize=1)
def get_reranker():
    """Returns a CrossEncoder, or None when disabled / unavailable (graceful degradation)."""
    if not settings.ENABLE_RERANKER:
        return None
    try:
        from sentence_transformers import CrossEncoder

        log.info("Loading reranker %s", settings.RERANKER_MODEL)
        return CrossEncoder(settings.RERANKER_MODEL, max_length=512)
    except Exception:  # noqa: BLE001 - missing dep, no network, OOM ...
        log.exception("Reranker unavailable - falling back to hybrid-search order")
        return None


@lru_cache(maxsize=1)
def get_llm_client():
    from google import genai
    from google.genai import types

    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=settings.LLM_TIMEOUT_SECONDS * 1000),
    )
