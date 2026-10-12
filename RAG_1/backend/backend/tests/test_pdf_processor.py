import io

import pymupdf
import pytest
from PIL import Image

from ingestion.pdf_processor import EmptyPDFError, PDFProcessor


def _png(size=(300, 200)):
    import random
    img = Image.new("RGB", size)
    img.putdata([(random.randint(0, 255),) * 3 for _ in range(size[0] * size[1])])  # incompressible
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def sample_pdf(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "1. Introduction", fontsize=20)
    page.insert_textbox(pymupdf.Rect(72, 100, 520, 200),
                        "This paper studies accident statistics in mining. " * 4, fontsize=11)
    page.insert_text((72, 230), "Statement 7.1 Accident statistics", fontsize=11)
    # ruled 3x3 table
    x0, y0, cw, rh = 72, 245, 120, 22
    for r in range(4):
        page.draw_line((x0, y0 + r * rh), (x0 + 3 * cw, y0 + r * rh))
    for c in range(4):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + 3 * rh))
    rows = [["Mineral", "District", "Killed"], ["Granite", "Jaipur", "4"], ["Marble", "Udaipur", "2"]]
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            page.insert_text((x0 + c * cw + 5, y0 + r * rh + 15), val, fontsize=10)
    page.insert_image(pymupdf.Rect(72, 360, 272, 460), stream=_png())
    page.insert_text((72, 480), "Figure 1: Fatal accidents per year", fontsize=10)
    page.insert_image(pymupdf.Rect(300, 360, 310, 370), stream=_png((16, 16)))  # tiny icon: must be skipped
    path = tmp_path / "sample.pdf"
    doc.save(path)
    return str(path)


def test_extracts_text_table_figure(sample_pdf, tmp_path):
    chunks = PDFProcessor(output_image_dir=str(tmp_path / "img")).process(sample_pdf, "sample.pdf", "a" * 64)
    types = [c.metadata.chunk_type for c in chunks]
    assert "text" in types and "table" in types
    assert types.count("figure") == 1  # icon filtered out

    table = next(c for c in chunks if c.metadata.chunk_type == "table")
    assert "| Mineral | District | Killed |" in table.content and "Udaipur" in table.content
    assert "Statement 7.1" in table.content  # caption pulled in

    fig = next(c for c in chunks if c.metadata.chunk_type == "figure")
    assert "Figure 1" in fig.content and fig.image_path and fig.image_url.startswith("/static/")

    text = next(c for c in chunks if c.metadata.chunk_type == "text")
    assert text.metadata.section.startswith("1. Introduction")
    assert not any("Granite" in c.content for c in chunks if c.metadata.chunk_type == "text")  # no table duplication


def test_reindexing_is_deterministic(sample_pdf, tmp_path):
    p = PDFProcessor(output_image_dir=str(tmp_path / "img"))
    a = [c.metadata.chunk_id for c in p.process(sample_pdf, "s.pdf", "b" * 64)]
    b = [c.metadata.chunk_id for c in p.process(sample_pdf, "s.pdf", "b" * 64)]
    assert a == b


def test_blank_pdf_raises_empty(tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    path = tmp_path / "blank.pdf"
    doc.save(path)
    with pytest.raises(EmptyPDFError):
        PDFProcessor(output_image_dir=str(tmp_path / "i")).process(str(path))


def test_corrupt_pdf_raises_value_error(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 this is not really a pdf")
    with pytest.raises(ValueError):
        PDFProcessor(output_image_dir=str(tmp_path / "i")).process(str(bad))
