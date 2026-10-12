from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import END, StateGraph

from config import settings
from graph.nodes import (
    finalize_failure_node,
    generate_answer_node,
    hybrid_retrieval_node,
    rerank_node,
    rewrite_query_node,
    validate_answer_node,
)
from models.schemas import StructuredAnswer


class AgentState(TypedDict, total=False):
    original_query: str
    rewritten_query: str
    is_general_query: bool
    conversation_history: List[Dict[str, str]]
    document_ids: Optional[List[str]]
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    generated_answer: Optional[StructuredAnswer]
    validation_passed: bool
    retry_count: int
    error: Optional[str]


def route_after_rewrite(state: AgentState) -> str:
    if state.get("is_general_query", False):
        return "generate"
    return "retrieve"


def route_after_validation(state: AgentState) -> str:
    if state.get("validation_passed"):
        return "end"
    if state.get("retry_count", 0) > settings.MAX_RAG_RETRIES:
        return "finalize"
    return "retrieve"


def create_rag_graph():
    wf = StateGraph(AgentState)
    wf.add_node("rewrite_query", rewrite_query_node)
    wf.add_node("retrieve", hybrid_retrieval_node)
    wf.add_node("rerank", rerank_node)
    wf.add_node("generate", generate_answer_node)
    wf.add_node("validate", validate_answer_node)
    wf.add_node("finalize", finalize_failure_node)

    wf.set_entry_point("rewrite_query")
    
    wf.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {"generate": "generate", "retrieve": "retrieve"}
    )
    
    wf.add_edge("retrieve", "rerank")
    wf.add_edge("rerank", "generate")
    wf.add_edge("generate", "validate")
    wf.add_conditional_edges(
        "validate", route_after_validation, {"end": END, "retrieve": "retrieve", "finalize": "finalize"}
    )
    wf.add_edge("finalize", END)
    return wf.compile()


app_graph = create_rag_graph()