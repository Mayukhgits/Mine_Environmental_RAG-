import hashlib
import logging
import os
import threading
import uuid
from typing import Any, Dict, Iterable, List, Optional

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import settings
from models.schemas import ExtractedChunk

log = logging.getLogger(__name__)

_NAMESPACE = uuid.UUID("6f1c2a52-8d0e-4f43-9a52-0d3f0a6c1b11")


def _batched(seq: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


class VectorStoreManager:
    """Hybrid (dense + BM25) store on Qdrant. Use `services.get_vector_store()`."""

    def __init__(self) -> None:
        if settings.USE_LOCAL_QDRANT:
            self.client = QdrantClient(path=settings.QDRANT_LOCAL_PATH)
        else:
            self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=30)

        self.dense_ef = TextEmbedding(model_name=settings.TEXT_EMBEDDING_MODEL)
        self.sparse_ef = SparseTextEmbedding(model_name=settings.SPARSE_EMBEDDING_MODEL)
        # derive the vector size from the model instead of hard-coding 384
        self.dense_dim = len(next(iter(self.dense_ef.embed(["dimension probe"]))))
        self._docs_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_lock = threading.Lock()
        self._setup_collection()

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def _setup_collection(self) -> None:
        name = settings.QDRANT_COLLECTION
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": models.VectorParams(size=self.dense_dim, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    # IDF modifier is required for BM25 scores to be meaningful
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False),
                        modifier=models.Modifier.IDF,
                    )
                },
            )
            log.info("Created collection %s (dim=%d)", name, self.dense_dim)
        else:
            info = self.client.get_collection(name)
            existing = info.config.params.vectors
            size = existing["dense"].size if isinstance(existing, dict) else existing.size
            sparse_cfg = (info.config.params.sparse_vectors or {}).get("sparse")
            if sparse_cfg is None or sparse_cfg.modifier != models.Modifier.IDF:
                log.warning(
                    "Collection '%s' was created without the BM25 IDF modifier; keyword search quality will be "
                    "poor. Use a fresh QDRANT_COLLECTION and re-upload your PDFs.", name,
                )
            if size != self.dense_dim:
                raise RuntimeError(
                    f"Collection '{name}' has dim={size} but model "
                    f"'{settings.TEXT_EMBEDDING_MODEL}' produces dim={self.dense_dim}. "
                    "Use a new QDRANT_COLLECTION name or re-index."
                )

        if not settings.USE_LOCAL_QDRANT:  # payload indexes are no-ops in local mode
            for field in ("doc_hash", "document_name", "chunk_type"):
                try:
                    self.client.create_payload_index(
                        name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
                    )
                except Exception:  # already exists
                    pass

    # ------------------------------------------------------------------ #
    # document registry
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _hash_filter(doc_hash: str) -> models.Filter:
        return models.Filter(
            must=[models.FieldCondition(key="doc_hash", match=models.MatchValue(value=doc_hash))]
        )

    def is_document_indexed(self, doc_hash: str) -> bool:
        return self.client.count(settings.QDRANT_COLLECTION, count_filter=self._hash_filter(doc_hash), exact=False).count > 0

    def get_all_indexed_documents(self) -> List[Dict[str, Any]]:
        with self._cache_lock:
            if self._docs_cache is not None:
                return list(self._docs_cache)

        docs: Dict[str, Dict[str, Any]] = {}
        offset = None
        while True:  # paginate: the old code silently stopped after 10 000 points
            points, offset = self.client.scroll(
                settings.QDRANT_COLLECTION,
                limit=1000,
                offset=offset,
                with_payload=["document_name", "doc_hash"],
                with_vectors=False,
            )
            for p in points:
                h = (p.payload or {}).get("doc_hash")
                if not h:
                    continue
                d = docs.setdefault(h, {"document_name": p.payload.get("document_name"), "doc_hash": h, "chunks": 0})
                d["chunks"] += 1
            if offset is None:
                break

        result = sorted(docs.values(), key=lambda d: (d["document_name"] or "").lower())
        with self._cache_lock:
            self._docs_cache = result
        return list(result)

    def _invalidate_cache(self) -> None:
        with self._cache_lock:
            self._docs_cache = None

    # ------------------------------------------------------------------ #
    # write path
    # ------------------------------------------------------------------ #
    @staticmethod
    def _embed_text(c: ExtractedChunk) -> str:
        # section title gives both dense and BM25 retrievers extra signal
        section = c.metadata.section if c.metadata.section not in (None, "General") else ""
        return f"{section}\n{c.content}" if section else c.content

    def index_chunks(self, chunks: List[ExtractedChunk], doc_hash: str) -> int:
        if not chunks:
            return 0
        texts = [self._embed_text(c) for c in chunks]
        bs = settings.EMBED_BATCH_SIZE
        try:
            log.info("Embedding %d chunks (dense + sparse)", len(texts))
            dense = list(self.dense_ef.embed(texts, batch_size=bs))
            sparse = list(self.sparse_ef.embed(texts, batch_size=bs))

            points = []
            for c, d, s in zip(chunks, dense, sparse):
                payload = {
                    **c.metadata.model_dump(),
                    "content": c.content,
                    "image_path": c.image_path,
                    "image_url": c.image_url,
                    "doc_hash": doc_hash,
                }
                points.append(
                    models.PointStruct(
                        # deterministic id -> re-indexing the same chunk overwrites, never duplicates
                        id=str(uuid.uuid5(_NAMESPACE, f"{doc_hash}:{c.metadata.chunk_id}")),
                        vector={
                            "dense": d.tolist(),
                            "sparse": models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
                        },
                        payload=payload,
                    )
                )

            batches = list(_batched(points, settings.UPSERT_BATCH_SIZE))
            for n, batch in enumerate(batches, 1):
                log.info("Upserting batch %d/%d", n, len(batches))
                self.client.upsert(settings.QDRANT_COLLECTION, points=batch, wait=True)
        except Exception:
            # never leave a half-indexed document that would later be reported "already_exists"
            log.exception("Indexing failed - rolling back %s", doc_hash[:12])
            self.delete_document(doc_hash)
            raise
        self._invalidate_cache()
        return len(points)

    def delete_document(self, doc_hash: str) -> int:
        before = self.client.count(settings.QDRANT_COLLECTION, count_filter=self._hash_filter(doc_hash), exact=True).count
        # remove extracted images belonging to the document
        offset = None
        while True:
            pts, offset = self.client.scroll(
                settings.QDRANT_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="doc_hash", match=models.MatchValue(value=doc_hash)),
                        models.FieldCondition(key="chunk_type", match=models.MatchValue(value="figure")),
                    ]
                ),
                limit=500,
                offset=offset,
                with_payload=["image_path"],
            )
            for p in pts:
                path = (p.payload or {}).get("image_path")
                if path and os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if offset is None:
                break
        self.client.delete(
            settings.QDRANT_COLLECTION,
            points_selector=models.FilterSelector(filter=self._hash_filter(doc_hash)),
            wait=True,
        )
        self._invalidate_cache()
        return before

    # ------------------------------------------------------------------ #
    # read path
    # ------------------------------------------------------------------ #
    def hybrid_search(
        self, query: str, top_k: int = 20, doc_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Dense + BM25 retrieval fused with Reciprocal Rank Fusion *inside* Qdrant (1 round trip)."""
        if not query.strip():
            return []

        # query_embed applies the model's retrieval instruction / query-side weighting
        q_dense = next(iter(self.dense_ef.query_embed(query))).tolist()
        q_sparse = next(iter(self.sparse_ef.query_embed(query)))

        flt = None
        if doc_filter:
            flt = models.Filter(
                should=[
                    models.FieldCondition(key="document_name", match=models.MatchAny(any=doc_filter)),
                    models.FieldCondition(key="doc_hash", match=models.MatchAny(any=doc_filter)),
                ]
            )

        prefetch_limit = max(top_k * 2, 40)
        result = self.client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            prefetch=[
                models.Prefetch(query=q_dense, using="dense", filter=flt, limit=prefetch_limit),
                models.Prefetch(
                    query=models.SparseVector(indices=q_sparse.indices.tolist(), values=q_sparse.values.tolist()),
                    using="sparse",
                    filter=flt,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        out = []
        for pt in result.points:
            payload = dict(pt.payload or {})
            payload["retrieval_score"] = pt.score
            out.append(payload)
        return out
