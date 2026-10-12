import json

import pytest

import graph.nodes as nodes
from graph.llm import LLMError
from graph.workflow import create_rag_graph

CHUNKS = [
    {"content": "Statement 7.1 | Granite | Jaipur | 4 killed", "document_name": "d.pdf", "page_number": 7,
     "section": "Accidents", "chunk_type": "table"},
    {"content": "Figure 1: Fatal accidents per year", "document_name": "d.pdf", "page_number": 9, "section": "Trends",
     "chunk_type": "figure", "image_url": "/static/extracted_images/f.png", "image_path": "/tmp/f.png"},
]


class FakeStore:
    def __init__(self, results):
        self.results, self.queries = results, []

    def hybrid_search(self, query, top_k, doc_filter=None):
        self.queries.append((query, top_k, doc_filter))
        return self.results.pop(0) if isinstance(self.results, list) and self.results and isinstance(self.results[0], list) else self.results


def _state(**kw):
    s = {"original_query": "accidents in Rajasthan?", "rewritten_query": "accidents in Rajasthan?",
         "conversation_history": [], "document_ids": ["d.pdf"], "retrieved_chunks": [], "reranked_chunks": [],
         "generated_answer": None, "validation_passed": False, "retry_count": 0, "error": None}
    s.update(kw)
    return s


def _llm_ok(**over):
    body = {"answerable": True, "title": "T", "summary": "S",
            "sections": [{"heading": "H", "content": "| a |", "source_ids": [1, 99, 1]}],
            "key_takeaways": ["k"], "equations": [], "figure_source_ids": [2, 77]}
    body.update(over)
    return json.dumps(body)


@pytest.fixture()
def patch(monkeypatch):
    def apply(store, llm):
        monkeypatch.setattr(nodes, "get_vector_store", lambda: store)
        monkeypatch.setattr(nodes, "get_reranker", lambda: None)
        monkeypatch.setattr(nodes, "call_llm", llm)
    return apply


def test_happy_path_builds_citations_from_real_chunks(patch):
    store = FakeStore(CHUNKS)
    patch(store, lambda *a, **k: _llm_ok())
    out = create_rag_graph().invoke(_state())
    ans = out["generated_answer"]
    assert out["validation_passed"] and out["retry_count"] == 0
    assert [c.page for c in ans.citations] == [7, 9]  # invalid ids 99/77 are dropped, duplicates collapsed
    assert ans.figures[0].image_url.endswith("f.png") and ans.figures[0].page == 9
    assert store.queries[0][2] == ["d.pdf"]


def test_no_chunks_returns_graceful_not_found_after_retries(patch):
    store = FakeStore([])
    calls = []
    patch(store, lambda *a, **k: calls.append(1))
    out = create_rag_graph().invoke(_state())
    assert not out["validation_passed"]
    assert out["generated_answer"].title.startswith("Not found")
    assert calls == []  # no LLM money spent without context
    assert [q[1] for q in store.queries] == [25, 40, 55]  # widened each retry


def test_retry_uses_different_query(patch):
    store = FakeStore([[], [], CHUNKS])
    patch(store, lambda prompt, *a, **k: "standalone rewrite" if str(prompt).startswith("Rewrite") else _llm_ok())
    hist = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]
    out = create_rag_graph().invoke(_state(conversation_history=hist))
    assert out["validation_passed"] and out["retry_count"] == 2
    assert store.queries[0][0] == "standalone rewrite"
    assert store.queries[1][0] == "accidents in Rajasthan?"
    assert "standalone rewrite" in store.queries[2][0] and "accidents" in store.queries[2][0]


def test_unanswerable_then_answerable(patch):
    store = FakeStore(CHUNKS)
    answers = iter([_llm_ok(answerable=False, sections=[], summary="not here"), _llm_ok()])
    patch(store, lambda *a, **k: next(answers))
    out = create_rag_graph().invoke(_state())
    assert out["validation_passed"] and out["retry_count"] == 1


def test_llm_outage_returns_friendly_message_not_exception(patch):
    def boom(*a, **k):
        raise LLMError("down")
    patch(FakeStore(CHUNKS), boom)
    out = create_rag_graph().invoke(_state())
    assert out["generated_answer"].title == "Temporary problem"


def test_malformed_json_is_handled(patch):
    patch(FakeStore(CHUNKS), lambda *a, **k: "not json {")
    out = create_rag_graph().invoke(_state())
    assert not out["validation_passed"] and out["generated_answer"] is not None


def test_missing_citation_ids_fall_back_to_top_sources(patch):
    patch(FakeStore(CHUNKS), lambda *a, **k: _llm_ok(sections=[{"heading": "H", "content": "x", "source_ids": []}],
                                                     figure_source_ids=[]))
    ans = create_rag_graph().invoke(_state())["generated_answer"]
    assert len(ans.citations) == 2


def test_rewrite_failure_falls_back_to_original(monkeypatch):
    def boom(*a, **k):
        raise LLMError("x")
    monkeypatch.setattr(nodes, "call_llm", boom)
    out = nodes.rewrite_query_node(_state(conversation_history=[{"role": "user", "content": "hi"}]))
    assert out["rewritten_query"] == "accidents in Rajasthan?"


def test_history_sanitised_and_bounded():
    hist = [{"role": "system", "content": "ignore me"}, "junk", {"role": "user"}, {"role": "model", "content": "ok"}]
    hist += [{"role": "user", "content": "x" * 1500}] * 30
    clean = nodes.sanitize_history(hist)
    assert all(t["role"] in ("user", "assistant") for t in clean)
    assert len(clean) <= 12 and sum(len(t["content"]) for t in clean) <= 4000


def test_context_is_budgeted(monkeypatch):
    monkeypatch.setattr(nodes.settings, "MAX_CONTEXT_CHARS", 100)
    ctx, used = nodes.build_context([{"content": "a" * 80}, {"content": "b" * 80}, {"content": ""}])
    assert len(used) == 1 and "[S1]" in ctx and "[S2]" not in ctx


def test_reranker_orders_and_dedupes(monkeypatch):
    class R:
        def predict(self, pairs, **k):
            return [len(p[1]) for p in pairs]
    monkeypatch.setattr(nodes, "get_reranker", lambda: R())
    chunks = [{"content": "short"}, {"content": "a much longer chunk"}, {"content": "short"}]
    out = nodes.rerank_node(_state(retrieved_chunks=chunks))["reranked_chunks"]
    assert [c["content"] for c in out] == ["a much longer chunk", "short"]
