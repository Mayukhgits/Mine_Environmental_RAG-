import hashlib
import json
import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

from config import settings
from ingestion.chunker import SemanticChunker
from models.schemas import ExtractedChunk

log = logging.getLogger(__name__)

_CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?\.?|table|statement|chart|plate|scheme)\s*[\dIVX]+", re.I)
_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+(\.\d+){0,3}|[IVX]+\.|chapter\s+\d+|appendix\s+\w+)\b", re.I)


class EmptyPDFError(ValueError):
    pass


class PDFProcessor:
    def __init__(self, output_image_dir: Optional[str] = None, chunker: Optional[SemanticChunker] = None):
        self.output_image_dir = output_image_dir or settings.IMAGE_STORE_DIR
        os.makedirs(self.output_image_dir, exist_ok=True)
        self.chunker = chunker or SemanticChunker()

    def process(self, pdf_path: str, doc_name: Optional[str] = None, doc_hash: Optional[str] = None) -> List[ExtractedChunk]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        doc_name = doc_name or os.path.basename(pdf_path)
        doc_hash = doc_hash or self._hash(pdf_path)

        try:
            doc = pymupdf.open(pdf_path)
        except Exception as exc:
            raise ValueError(f"Cannot open PDF: {exc}") from exc

        try:
            if doc.needs_pass:
                raise ValueError("PDF is password protected")
            if doc.page_count > settings.MAX_PDF_PAGES:
                raise ValueError(f"PDF has {doc.page_count} pages (limit {settings.MAX_PDF_PAGES})")

            body_size = self._body_font_size(doc)
            elements: List[Dict[str, Any]] = []
            seen_images: set = set()
            section = "General"

            for page in doc:
                try:
                    section = self._extract_page(page, doc, doc_hash, body_size, section, elements, seen_images)
                except Exception:
                    log.exception("Failed to process page %s of %s", page.number + 1, doc_name)
        finally:
            doc.close()

        chunks = self.chunker.process_raw_elements(elements, doc_name, doc_key=doc_hash[:12])
        if not any(c.metadata.chunk_type in ("text", "table") for c in chunks):
            raise EmptyPDFError(
                "No extractable text found (scanned PDF?). Set ENABLE_OCR=true and install Tesseract."
            )
        log.info("Extracted %d chunks from %s", len(chunks), doc_name)
        return chunks

    def _extract_page(self, page, doc, doc_hash, body_size, section, elements, seen_images) -> str:
        page_num = page.number + 1

        page_items: List[Dict[str, Any]] = []
        table_boxes: List[pymupdf.Rect] = []
        text_blocks = self._text_blocks(page)
        try:
            found = page.find_tables()
            for tbl in found.tables:
                rows = tbl.extract()
                if not rows or len(rows) < 2:
                    continue
                rect = pymupdf.Rect(tbl.bbox)
                table_boxes.append(rect)
                page_items.append(
                    {
                        "type": "table",
                        "page": page_num,
                        "section": section,
                        "rows": rows,
                        "caption": self._caption_above(text_blocks, rect),
                        "_y": rect.y0,
                    }
                )
        except Exception:
            log.debug("find_tables failed on page %s", page_num, exc_info=True)

        for blk in text_blocks:
            rect = pymupdf.Rect(blk["bbox"])
            if any(rect.intersects(t) and (rect & t).get_area() > 0.5 * rect.get_area() for t in table_boxes):
                continue
            text = blk["text"]
            if self._is_heading(blk, body_size):
                page_items.append({"type": "heading", "content": text[:120], "_y": blk["bbox"][1]})
                continue
            page_items.append(
                {"type": "text", "page": page_num, "section": section, "content": text, "_y": blk["bbox"][1]}
            )

        if not any(i["type"] in ("text", "table") for i in page_items) and settings.ENABLE_OCR:
            try:
                tp = page.get_textpage_ocr(dpi=200, full=True)
                ocr_text = page.get_text("text", textpage=tp).strip()
                if ocr_text:
                    page_items.append(
                        {"type": "text", "page": page_num, "section": section, "content": ocr_text, "_y": 0}
                    )
            except Exception:
                log.warning("OCR failed on page %s (is Tesseract installed?)", page_num)

        for img in page.get_images(full=True):
            fig = self._extract_image(page, doc, img, doc_hash, page_num, section, text_blocks, seen_images)
            if fig:
                page_items.append(fig)

        page_items.sort(key=lambda e: e.get("_y", 0))
        for item in page_items:
            if item["type"] == "heading":
                section = item["content"]
                continue
            item["section"] = section
            elements.append(item)
        return section

    @staticmethod
    def _text_blocks(page) -> List[Dict[str, Any]]:
        out = []
        for b in page.get_text("dict", sort=True)["blocks"]:
            if b.get("type") != 0:
                continue
            spans = [s for ln in b["lines"] for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "\n".join(
                " ".join(s["text"] for s in ln["spans"]).strip() for ln in b["lines"]
            ).strip()
            text = re.sub(r"-\n(?=[a-z])", "", text)
            text = re.sub(r"[ \t]+", " ", text)
            if not text:
                continue
            out.append(
                {
                    "bbox": b["bbox"],
                    "text": text,
                    "size": max(s["size"] for s in spans),
                    "bold": all(bool(s["flags"] & 16) or "bold" in s["font"].lower() for s in spans),
                }
            )
        return out

    @staticmethod
    def _body_font_size(doc) -> float:
        sizes: Counter = Counter()
        for i, page in enumerate(doc):
            if i >= 15:
                break
            for b in page.get_text("dict")["blocks"]:
                for ln in b.get("lines", []):
                    for s in ln["spans"]:
                        sizes[round(s["size"], 1)] += len(s["text"])
        return sizes.most_common(1)[0][0] if sizes else 10.0

    @staticmethod
    def _is_heading(blk: Dict[str, Any], body_size: float) -> bool:
        words = blk["text"].split()
        if not words or len(words) > 18 or len(blk["text"].splitlines()) > 2:
            return False
        if _CAPTION_RE.match(blk["text"]):
            return False
        bigger = blk["size"] >= body_size * 1.15
        bold_numbered = blk["bold"] and bool(_NUMBERED_HEADING_RE.match(blk["text"]))
        return bigger or bold_numbered

    @staticmethod
    def _caption_above(text_blocks, rect, max_gap: float = 60) -> str:
        best = None
        for b in text_blocks:
            y1 = b["bbox"][3]
            if 0 <= rect.y0 - y1 <= max_gap and (best is None or y1 > best["bbox"][3]):
                best = b
        return best["text"][:300] if best else ""

    @staticmethod
    def _caption_near(text_blocks, rect, max_gap: float = 60) -> str:
        cands: List[Tuple[float, str]] = []
        for b in text_blocks:
            y0, y1 = b["bbox"][1], b["bbox"][3]
            gap = y0 - rect.y1 if y0 >= rect.y1 - 5 else rect.y0 - y1
            if 0 <= gap <= max_gap or (-5 <= gap < 0):
                score = gap + (0 if _CAPTION_RE.match(b["text"]) else 1000)
                cands.append((score, b["text"]))
        if not cands:
            return ""
        cands.sort()
        return cands[0][1][:400] if cands[0][0] < 1000 else ""

    def _extract_image(self, page, doc, img, doc_hash, page_num, section, text_blocks, seen) -> Optional[Dict[str, Any]]:
        xref = img[0]
        if xref in seen:
            return None
        try:
            pix = pymupdf.Pixmap(doc, xref)
            if pix.width < settings.MIN_IMAGE_SIDE_PX or pix.height < settings.MIN_IMAGE_SIDE_PX:
                return None
            if pix.colorspace and pix.colorspace.n > 3 or pix.alpha:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            data = pix.tobytes("png")
        except Exception:
            log.debug("Skipping undecodable image xref=%s", xref, exc_info=True)
            return None
        if len(data) < settings.MIN_IMAGE_BYTES:
            return None

        digest = hashlib.sha1(data).hexdigest()[:10]
        if digest in seen:
            return None
        seen.update({xref, digest})

        name = f"{doc_hash[:12]}_p{page_num}_{digest}.png"
        path = os.path.join(self.output_image_dir, name)
        if not os.path.exists(path):
            with open(path, "wb") as fh:
                fh.write(data)

        rects = page.get_image_rects(xref)
        rect = rects[0] if rects else pymupdf.Rect(0, 0, 0, 0)
        caption = self._caption_near(text_blocks, rect) if rects else ""
        return {
            "type": "figure",
            "page": page_num,
            "section": section,
            "content": caption or f"Figure on page {page_num} (section: {section})",
            "image_path": path,
            "image_url": f"{settings.STATIC_URL_PREFIX}/{name}",
            "_y": rect.y0,
        }

    @staticmethod
    def _hash(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()