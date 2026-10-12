import io

import pymupdf
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(store):
    import main
    with TestClient(main.app) as c:
        yield c


def _pdf_bytes(text="Hello mining safety world. " * 20):
    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(72, 72, 520, 700), text, fontsize=11)
    return doc.tobytes()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["qdrant"] is True
    assert "x-request-id" in r.headers


def test_upload_validation(client):
    assert client.post("/api/upload", files={"file": ("a.txt", b"hi", "text/plain")}).status_code == 400
    assert client.post("/api/upload", files={"file": ("a.pdf", b"not a pdf", "application/pdf")}).status_code == 400
    assert client.post("/api/upload", files={"file": ("a.pdf", b"", "application/pdf")}).status_code == 400
    bad = b"%PDF-1.4 garbage"
    assert client.post("/api/upload", files={"file": ("a.pdf", bad, "application/pdf")}).status_code == 422


def test_upload_size_limit(client, monkeypatch):
    import main
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)
    r = client.post("/api/upload", files={"file": ("a.pdf", _pdf_bytes(), "application/pdf")})
    assert r.status_code == 413


def test_upload_lifecycle_and_dedupe(client):
    data = _pdf_bytes()
    r1 = client.post("/api/upload", files={"file": ("../../evil name.pdf", data, "application/pdf")})
    assert r1.status_code == 200, r1.text
    j = r1.json()
    assert j["status"] == "indexed" and j["chunks"] >= 1 and "/" not in j["filename"]

    r2 = client.post("/api/upload", files={"file": ("again.pdf", data, "application/pdf")})
    assert r2.json()["status"] == "already_exists" and r2.json()["doc_hash"] == j["doc_hash"]

    docs = client.get("/api/documents").json()["documents"]
    assert any(d["doc_hash"] == j["doc_hash"] for d in docs)

    assert client.delete(f"/api/documents/{j['doc_hash']}").status_code == 200
    assert client.delete(f"/api/documents/{j['doc_hash']}").status_code == 404
    assert client.delete("/api/documents/not-a-hash").status_code == 400


def test_chat_validation_and_config(client, monkeypatch):
    import main
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "   "}).status_code == 400
    monkeypatch.setattr(main.settings, "GEMINI_API_KEY", "")
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 503


def test_chat_response_contract(client, monkeypatch):
    import main
    from models.schemas import SectionContent, StructuredAnswer

    class G:
        def invoke(self, state):
            assert state["conversation_history"] == [] and state["document_ids"] == ["x"]
            return {"generated_answer": StructuredAnswer(title="t", summary="s", sections=[SectionContent(heading="h", content="c")]),
                    "reranked_chunks": [{"content": "c"}], "validation_passed": True, "retry_count": 0, "rewritten_query": "q"}

    monkeypatch.setattr(main, "app_graph", G())
    r = client.post("/api/chat", json={"message": "hello", "documents": ["x"]})
    body = r.json()
    assert r.status_code == 200
    assert set(body) == {"response", "sources", "meta"} and body["response"]["title"] == "t"
    assert body["meta"]["grounded"] is True


def test_chat_pipeline_crash_is_a_clean_500(client, monkeypatch):
    import main

    class G:
        def invoke(self, state):
            raise ValueError("secret internal detail")

    monkeypatch.setattr(main, "app_graph", G())
    r = client.post("/api/chat", json={"message": "hello"})
    assert r.status_code == 500 and "secret" not in r.text
