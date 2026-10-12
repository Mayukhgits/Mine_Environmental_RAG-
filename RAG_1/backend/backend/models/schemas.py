from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ChunkType = Literal["text", "table", "figure", "formula"]


class ChunkMetadata(BaseModel):
    document_name: str
    page_number: int
    chapter: Optional[str] = "Unknown"
    section: Optional[str] = "General"
    chunk_type: ChunkType = Field(..., description="text | table | figure | formula")
    chunk_id: str


class ExtractedChunk(BaseModel):
    content: str
    metadata: ChunkMetadata
    image_path: Optional[str] = None
    image_url: Optional[str] = None


class LLMSection(BaseModel):
    heading: str
    content: str
    source_ids: List[int] = Field(default_factory=list)


class LLMAnswer(BaseModel):
    answerable: bool = True
    title: str
    summary: str
    sections: List[LLMSection] = Field(default_factory=list)
    key_takeaways: List[str] = Field(default_factory=list)
    equations: List[str] = Field(default_factory=list)
    figure_source_ids: List[int] = Field(default_factory=list)


class ContextReference(BaseModel):
    document: str
    page: int
    section: str
    snippet: str
    figure_path: Optional[str] = None


class SectionContent(BaseModel):
    heading: str
    content: str


class Figure(BaseModel):
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    caption: str = ""
    page: int = 0


class StructuredAnswer(BaseModel):
    title: str
    summary: str
    sections: List[SectionContent]
    key_takeaways: List[str] = Field(default_factory=list)
    equations: List[str] = Field(default_factory=list)
    figures: List[Figure] = Field(default_factory=list)
    citations: List[ContextReference] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: List[Dict[str, Any]] = Field(default_factory=list, max_length=100)
    documents: List[str] = Field(default_factory=list, max_length=50)