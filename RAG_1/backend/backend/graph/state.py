from typing import Any, Dict, List, Optional, TypedDict

from models.schemas import StructuredAnswer


class AgentState(TypedDict, total=False):
    original_query: str
    rewritten_query: str
    conversation_history: List[Dict[str, str]]
    document_ids: Optional[List[str]]
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    generated_answer: Optional[StructuredAnswer]
    validation_passed: bool
    retry_count: int
    error: Optional[str]
