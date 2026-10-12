from models.schemas import ChunkMetadata, ExtractedChunk


def _chunk(i, text, section="General", ctype="text", doc="d.pdf"):
    return ExtractedChunk(
        content=text,
        metadata=ChunkMetadata(document_name=doc, page_number=i, section=section, chunk_type=ctype, chunk_id=f"c{i}"),
    )


def test_index_search_filter_delete(store):
    h1, h2 = "1" * 64, "2" * 64
    store.index_chunks([_chunk(1, "granite quarry accidents in jaipur"), _chunk(2, "marble mining udaipur safety")], h1)
    store.index_chunks([_chunk(1, "quantum chromodynamics lattice", doc="e.pdf")], h2)

    assert store.is_document_indexed(h1) and store.is_document_indexed(h2)
    names = {d["doc_hash"]: d for d in store.get_all_indexed_documents()}
    assert names[h1]["chunks"] == 2 and names[h2]["document_name"] == "e.pdf"

    hits = store.hybrid_search("granite accidents", top_k=5)
    assert hits and hits[0]["content"].startswith("granite")
    assert "retrieval_score" in hits[0]

    only_h2 = store.hybrid_search("granite accidents", top_k=5, doc_filter=[h2])
    assert {h["doc_hash"] for h in only_h2} == {h2}
    by_name = store.hybrid_search("quantum", top_k=5, doc_filter=["e.pdf"])
    assert {h["document_name"] for h in by_name} == {"e.pdf"}

    # idempotent re-index (deterministic ids)
    store.index_chunks([_chunk(1, "granite quarry accidents in jaipur"), _chunk(2, "marble mining udaipur safety")], h1)
    assert {d["doc_hash"]: d for d in store.get_all_indexed_documents()}[h1]["chunks"] == 2

    assert store.delete_document(h1) == 2
    assert not store.is_document_indexed(h1)
    assert store.is_document_indexed(h2)
    store.delete_document(h2)


def test_empty_query_and_empty_chunks(store):
    assert store.hybrid_search("   ") == []
    assert store.index_chunks([], "3" * 64) == 0


def test_failed_indexing_rolls_back(store, monkeypatch):
    h = "4" * 64
    calls = {"n": 0}
    real = store.client.upsert

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return real(*a, **k)

    monkeypatch.setattr(store.client, "upsert", flaky)
    from config import settings
    monkeypatch.setattr(settings, "UPSERT_BATCH_SIZE", 2)
    import pytest
    with pytest.raises(RuntimeError):
        store.index_chunks([_chunk(i, f"text number {i}") for i in range(1, 6)], h)
    assert not store.is_document_indexed(h)  # no half-indexed document left behind
