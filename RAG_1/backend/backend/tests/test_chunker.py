from ingestion.chunker import SemanticChunker, table_rows_to_markdown


def _els(text, page=1, section="S"):
    return [{"type": "text", "page": page, "section": section, "content": text}]


def test_small_paragraphs_are_merged():
    ch = SemanticChunker(chunk_size=50, overlap=5)
    els = [{"type": "text", "page": 1, "section": "A", "content": f"para {i} " + "word " * 5} for i in range(4)]
    out = ch.process_raw_elements(els, "d.pdf", "k")
    assert len(out) == 1 and "para 3" in out[0].content


def test_never_exceeds_chunk_size_and_keeps_all_words():
    ch = SemanticChunker(chunk_size=40, overlap=8)
    text = " ".join(f"w{i}" for i in range(500))  # no punctuation at all
    out = ch.process_raw_elements(_els(text), "d.pdf", "k")
    assert all(len(c.content.split()) <= 40 for c in out)
    joined = " ".join(c.content for c in out)
    assert all(f"w{i}" in joined for i in range(500))


def test_sentence_boundaries_respected():
    ch = SemanticChunker(chunk_size=30, overlap=5)
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(20))
    out = ch.process_raw_elements(_els(text), "d.pdf", "k")
    assert len(out) > 1
    assert all(c.content.rstrip().endswith(".") for c in out)


def test_section_and_page_change_flushes():
    ch = SemanticChunker(chunk_size=100, overlap=10)
    els = _els("alpha beta", 1, "A") + _els("gamma delta", 2, "B")
    out = ch.process_raw_elements(els, "d.pdf", "k")
    assert [(c.metadata.page_number, c.metadata.section) for c in out] == [(1, "A"), (2, "B")]


def test_table_split_repeats_header():
    rows = [["Mineral", "District"]] + [[f"m{i}", f"d{i}"] for i in range(60)]
    ch = SemanticChunker()
    out = ch.process_raw_elements(
        [{"type": "table", "page": 3, "section": "S", "rows": rows, "caption": "Statement 7.1"}], "d.pdf", "k"
    )
    assert len(out) == 3 and all(c.metadata.chunk_type == "table" for c in out)
    assert all("| Mineral | District |" in c.content and "Statement 7.1" in c.content for c in out)


def test_markdown_handles_ragged_and_none_cells():
    md = table_rows_to_markdown([["a", None, "c"], ["1", "2"], [None, None]])
    assert md[0] == "| a |  | c |" and md[2] == "| 1 | 2 |  |" and len(md) == 3


def test_ids_are_deterministic_and_unique():
    ch = SemanticChunker(chunk_size=20, overlap=2)
    els = _els(" ".join(["x"] * 200))
    a = [c.metadata.chunk_id for c in ch.process_raw_elements(els, "d.pdf", "k")]
    b = [c.metadata.chunk_id for c in ch.process_raw_elements(els, "d.pdf", "k")]
    assert a == b and len(set(a)) == len(a)


def test_figure_chunk_keeps_image_paths():
    out = SemanticChunker().process_raw_elements(
        [{"type": "figure", "page": 2, "content": "Figure 1: x", "image_path": "/a.png", "image_url": "/static/a.png"}],
        "d.pdf", "k",
    )
    assert out[0].image_url == "/static/a.png" and out[0].metadata.chunk_type == "figure"
