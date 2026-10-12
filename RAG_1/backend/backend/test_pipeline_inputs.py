import os
import sys

# Ensure backend root is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.pdf_processor import PDFProcessor
from vectorstore.qdrant_manager import VectorStoreManager
from graph.nodes import hybrid_retrieval_node, rerank_node, generate_answer_node

# Step A: Define test PDF path and query
TEST_PDF_PATH = r"C:\Users\Debajyoti\Downloads\SANKETNonCoal_17042025.pdf" # Adjust path if needed
TEST_QUERY = "What are the main observations or requirements in this document?"

def test_pdf_indexing():
    print("\n--- 1. Testing PDF Processing & Qdrant Indexing ---")
    if not os.path.exists(TEST_PDF_PATH):
        print(f"[FAIL] Test PDF not found at {TEST_PDF_PATH}")
        return None, None

    qdrant_mgr = VectorStoreManager()
    pdf_processor = PDFProcessor()

    # Calculate file hash
    doc_hash = qdrant_mgr.compute_file_hash(TEST_PDF_PATH)
    print(f"Document Hash: {doc_hash}")

    # Process and Index
    chunks = pdf_processor.process(TEST_PDF_PATH)
    print(f"Extracted {len(chunks)} chunks/elements from PDF.")
    
    qdrant_mgr.index_chunks(chunks, doc_hash)
    print("[PASS] PDF successfully indexed into Qdrant.")
    return qdrant_mgr, doc_hash


def test_retrieval_and_reranking(doc_hash):
    print("\n--- 2. Testing Retrieval & Reranking Nodes ---")
    
    # State setup with doc_hash filter
    initial_state = {
        "original_query": TEST_QUERY,
        "rewritten_query": TEST_QUERY,
        "conversation_history": [],
        "document_ids": [doc_hash],
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "generated_answer": None,
        "validation_passed": False,
        "retry_count": 0
    }

    # Test Hybrid Retrieval Node
    retrieval_output = hybrid_retrieval_node(initial_state)
    retrieved_chunks = retrieval_output.get("retrieved_chunks", [])
    print(f"Hybrid Search retrieved: {len(retrieved_chunks)} chunks")
    
    if not retrieved_chunks:
        print("[FAIL] Retrieval returned 0 chunks.")
        return None

    # Test Rerank Node
    state_after_retrieval = {**initial_state, "retrieved_chunks": retrieved_chunks}
    rerank_output = rerank_node(state_after_retrieval)
    reranked_chunks = rerank_output.get("reranked_chunks", [])
    print(f"Reranker selected top {len(reranked_chunks)} chunks.")

    print("\nTop Chunk Sample Snippet:")
    print("-" * 50)
    print(reranked_chunks[0].get("content", "")[:300])
    print("-" * 50)
    print("[PASS] Retrieval & Reranking working correctly.")
    
    return reranked_chunks


def test_model_generation(reranked_chunks):
    print("\n--- 3. Testing Gemini Model QA Generation ---")
    
    state_for_generation = {
        "original_query": TEST_QUERY,
        "reranked_chunks": reranked_chunks
    }

    gen_output = generate_answer_node(state_for_generation)
    answer = gen_output.get("generated_answer")

    if answer:
        print("[PASS] Gemini structured output received successfully!")
        print("\n--- Generated Answer Summary ---")
        print(f"Title: {answer.title}")
        print(f"Summary: {answer.summary}")
        print(f"Number of Sections: {len(answer.sections)}")
        if answer.sections:
            print(f"Section 1 Heading: {answer.sections[0].heading}")
    else:
        print("[FAIL] Gemini generation returned None or invalid output.")


if __name__ == "__main__":
    qdrant_mgr, doc_hash = test_pdf_indexing()
    if doc_hash:
        reranked_chunks = test_retrieval_and_reranking(doc_hash)
        if reranked_chunks:
            test_model_generation(reranked_chunks)