"""Central configuration. Every value can be overridden through environment / .env."""
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "Multimodal Academic PDF RAG"
    LOG_LEVEL: str = "INFO"

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    LLM_TIMEOUT_SECONDS: int = 60
    LLM_MAX_RETRIES: int = 3

    USE_LOCAL_QDRANT: bool = False
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_LOCAL_PATH: str = str(BASE_DIR / "qdrant_db")
    QDRANT_COLLECTION: str = "academic_pdf_rag_v2"

    TEXT_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    SPARSE_EMBEDDING_MODEL: str = "Qdrant/bm25"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    ENABLE_RERANKER: bool = True
    EMBED_BATCH_SIZE: int = 64
    UPSERT_BATCH_SIZE: int = 128
    WARMUP_ON_STARTUP: bool = True

    CHUNK_SIZE: int = 250
    CHUNK_OVERLAP: int = 40
    TABLE_MAX_ROWS_PER_CHUNK: int = 25

    IMAGE_STORE_DIR: str = str(BASE_DIR / "static" / "extracted_images")
    STATIC_URL_PREFIX: str = "/static/extracted_images"
    MIN_IMAGE_SIDE_PX: int = 120
    MIN_IMAGE_BYTES: int = 4_000
    ENABLE_OCR: bool = False
    MAX_UPLOAD_MB: int = 100
    MAX_PDF_PAGES: int = 2000
    MAX_CONCURRENT_INGESTIONS: int = 1

    RETRIEVAL_TOP_K: int = 25
    RERANK_TOP_N: int = 8
    MAX_RAG_RETRIES: int = 2
    MAX_CHUNK_CHARS: int = 3000
    MAX_CONTEXT_CHARS: int = 24000
    MAX_HISTORY_TURNS: int = 6
    MAX_HISTORY_CHARS: int = 4000
    MAX_QUERY_CHARS: int = 4000
    MAX_CONCURRENT_CHATS: int = 4

    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                return v
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("QDRANT_LOCAL_PATH", "IMAGE_STORE_DIR")
    @classmethod
    def _absolute(cls, v: str) -> str:
        p = Path(v)
        return str(p if p.is_absolute() else (BASE_DIR / p).resolve())

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def _overlap_lt_size(cls, v, info):
        size = info.data.get("CHUNK_SIZE", 250)
        if v >= size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return v


settings = Settings()


def ensure_dirs() -> None:
    Path(settings.IMAGE_STORE_DIR).mkdir(parents=True, exist_ok=True)
    if settings.USE_LOCAL_QDRANT:
        Path(settings.QDRANT_LOCAL_PATH).mkdir(parents=True, exist_ok=True)