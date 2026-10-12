import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Configure BEFORE `config` is imported anywhere.
_TMP = tempfile.mkdtemp(prefix="rag_tests_")
os.environ.update(
    USE_LOCAL_QDRANT="true",
    QDRANT_LOCAL_PATH=str(Path(_TMP) / "qdrant"),
    IMAGE_STORE_DIR=str(Path(_TMP) / "images"),
    GEMINI_API_KEY="test-key",
    ENABLE_RERANKER="false",
    WARMUP_ON_STARTUP="false",
    QDRANT_COLLECTION="test_collection",
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeDense:
    DIM = 32

    def __init__(self, *a, **k):
        pass

    @staticmethod
    def _vec(text):
        v = np.zeros(FakeDense.DIM, dtype=np.float32)
        for w in text.lower().split():
            v[hash(w) % FakeDense.DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts, batch_size=None):
        for t in texts:
            yield self._vec(t)

    def query_embed(self, q):
        yield self._vec(q)


class FakeSparse:
    def __init__(self, *a, **k):
        pass

    @staticmethod
    def _vec(text):
        counts = {}
        for w in text.lower().split():
            counts[abs(hash(w)) % 5000] = counts.get(abs(hash(w)) % 5000, 0) + 1
        idx = sorted(counts)
        return SimpleNamespace(
            indices=np.array(idx, dtype=np.int64), values=np.array([float(counts[i]) for i in idx], dtype=np.float32)
        )

    def embed(self, texts, batch_size=None):
        for t in texts:
            yield self._vec(t)

    def query_embed(self, q):
        yield self._vec(q)


@pytest.fixture(scope="session")
def store():
    import vectorstore.qdrant_manager as qm

    qm.TextEmbedding, qm.SparseTextEmbedding = FakeDense, FakeSparse
    from config import ensure_dirs

    ensure_dirs()
    import services

    services.get_vector_store.cache_clear()
    vs = services.get_vector_store()
    yield vs
    vs.client.close()
