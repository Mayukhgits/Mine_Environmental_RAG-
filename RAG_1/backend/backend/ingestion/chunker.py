import re
from typing import Any, Dict, List, Optional

from config import settings
from models.schemas import ChunkMetadata, ExtractedChunk

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def _wc(text: str) -> int:
    return len(text.split())


def _split_long(text: str, size: int, overlap: int) -> List[str]:
    pieces: List[str] = []
    for sentence in _SENTENCE_RE.split(text):
        words = sentence.split()
        if len(words) <= size:
            pieces.append(sentence)
        else:
            step = max(1, size - overlap)
            for i in range(0, len(words), step):
                pieces.append(" ".join(words[i : i + size]))
                if i + size >= len(words):
                    break
    return pieces


def table_rows_to_markdown(rows: List[List[Optional[str]]]) -> List[str]:
    cleaned = [[(c or "").replace("\n", " ").replace("|", "/").strip() for c in r] for r in rows]
    cleaned = [r for r in cleaned if any(r)]
    if not cleaned:
        return []
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    lines = ["| " + " | ".join(cleaned[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in cleaned[1:]]
    return lines


class SemanticChunker:
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_raw_elements(
        self, raw_elements: List[Dict[str, Any]], doc_name: str, doc_key: str = ""
    ) -> List[ExtractedChunk]:
        key = (doc_key or re.sub(r"\W+", "_", doc_name))[:16]
        chunks: List[ExtractedChunk] = []
        counter = 0

        def new_id(page: int, kind: str) -> str:
            nonlocal counter
            counter += 1
            return f"{key}_p{page}_{kind}_{counter}"

        def add(content, page, section, ctype, **extra):
            chunks.append(
                ExtractedChunk(
                    content=content,
                    metadata=ChunkMetadata(
                        document_name=doc_name,
                        page_number=page,
                        section=section or "General",
                        chunk_type=ctype,
                        chunk_id=new_id(page, ctype[:3]),
                    ),
                    **extra,
                )
            )

        buf_units: List[str] = []
        buf_key = None

        def flush(final: bool = True):
            nonlocal buf_units
            if not buf_units or buf_key is None:
                buf_units = []
                return
            page, section = buf_key
            for text in self._pack(buf_units):
                add(text, page, section, "text")
            buf_units = []

        for el in raw_elements:
            etype = el.get("type", "text")
            page = int(el.get("page", 1))
            section = el.get("section") or "General"

            if etype == "text":
                content = (el.get("content") or "").strip()
                if not content:
                    continue
                if buf_key != (page, section):
                    flush()
                    buf_key = (page, section)
                buf_units.append(content)

            elif etype == "table":
                flush()
                buf_key = None
                for part in self._table_parts(el):
                    add(part, page, section, "table")

            elif etype == "figure":
                flush()
                buf_key = None
                add(
                    el["content"],
                    page,
                    section,
                    "figure",
                    image_path=el.get("image_path"),
                    image_url=el.get("image_url"),
                )
        flush()
        return chunks

    def _pack(self, paragraphs: List[str]) -> List[str]:
        units: List[str] = []
        for p in paragraphs:
            if _wc(p) <= self.chunk_size:
                units.append(p)
            else:
                units.extend(_split_long(p, self.chunk_size, self.overlap))

        out: List[str] = []
        cur: List[str] = []
        cur_words = 0
        for u in units:
            w = _wc(u)
            if cur and cur_words + w > self.chunk_size:
                out.append("\n".join(cur))
                tail: List[str] = []
                tail_words = 0
                for prev in reversed(cur):
                    pw = _wc(prev)
                    if tail_words + pw > self.overlap:
                        break
                    tail.insert(0, prev)
                    tail_words += pw
                cur, cur_words = tail, tail_words
            cur.append(u)
            cur_words += w
        if cur and (not out or "\n".join(cur) != out[-1]):
            if not out or any(u not in out[-1] for u in cur):
                out.append("\n".join(cur))
        return out

    def _table_parts(self, el: Dict[str, Any]) -> List[str]:
        caption = (el.get("caption") or "").strip()
        lines = table_rows_to_markdown(el.get("rows") or [])
        if not lines:
            return []
        header, body = lines[:2], lines[2:]
        step = settings.TABLE_MAX_ROWS_PER_CHUNK
        parts = []
        groups = [body[i : i + step] for i in range(0, len(body), step)] or [[]]
        for n, grp in enumerate(groups, 1):
            title = caption + (f" (part {n}/{len(groups)})" if len(groups) > 1 else "")
            text = (title + "\n" if title else "") + "\n".join(header + grp)
            parts.append(text)
        return parts