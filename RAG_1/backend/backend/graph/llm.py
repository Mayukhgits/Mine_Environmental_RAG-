"""Thin wrapper around Gemini: retries transient failures with exponential backoff."""
import logging
import random
import time
from typing import Any, Optional

from config import settings
from services import get_llm_client

log = logging.getLogger(__name__)

_TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when the model cannot produce a usable response."""


def _is_transient(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _TRANSIENT_CODES:
        return True
    name = type(exc).__name__.lower()
    return any(t in name for t in ("timeout", "connect", "unavailable", "readerror", "remoteprotocol"))


def call_llm(contents: Any, config: Optional[Any] = None) -> str:
    client = get_llm_client()  
    delay = 1.0
    last: Optional[Exception] = None
    for attempt in range(1, settings.LLM_MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(model=settings.GEMINI_MODEL, contents=contents, config=config)
            text = (resp.text or "").strip()
            if not text:
                raise LLMError("Model returned an empty response (possibly blocked by safety filters)")
            return text
        except LLMError:
            raise
        except Exception as exc:  
            last = exc
            if attempt == settings.LLM_MAX_RETRIES or not _is_transient(exc):
                break
            sleep_for = delay + random.uniform(0, 0.5)
            log.warning("LLM call failed (%s); retry %d in %.1fs", exc, attempt, sleep_for)
            time.sleep(sleep_for)
            delay *= 2
    raise LLMError(f"LLM request failed: {last}") from last