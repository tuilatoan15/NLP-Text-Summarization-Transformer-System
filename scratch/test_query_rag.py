import sys
sys.path.insert(0, ".")

import logging
from backend.services.rag.service import RAGChatService
from backend.services.rag.rag_config import EMBEDDING_MODEL, RETRIEVAL_THRESHOLD

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("test_query")

service = RAGChatService()

# List documents in DB to get their IDs
docs = service.list_documents()
print("Uploaded Documents in DB:")
doc_ids = []
for d in docs:
    print(f" - ID: {d['id']}, Filename: {d['filename']}")
    doc_ids.append(d['id'])

query = "Tài liệu này đề cập đến các vấn đề chính nào?"
print(f"\nQuerying: '{query}'")
print(f"Targeting document IDs: {doc_ids}")

# Direct vector store query
query_vector = service.embedding_service.embed_query(query, EMBEDDING_MODEL)
print("\n[Step 1] Querying vector store...")
candidates = service.vector_store.query(
    query_vector=query_vector,
    top_k=30,
    document_ids=doc_ids
)
print(f"Received {len(candidates)} candidates from vector store.")
for i, c in enumerate(candidates[:5]):
    print(f"  {i+1}. Doc: {c['filename']}, Score: {c['embedding_score']:.4f}, Text preview: {c['text'][:100]}...")

# Hybrid retriever retrieve
print("\n[Step 2] Running Hybrid retrieve + Rerank...")
retrieved = service.retriever.retrieve(
    query=query,
    query_vector=query_vector,
    chunks=candidates,
    top_k=4,
    threshold=RETRIEVAL_THRESHOLD
)
print(f"Received {len(retrieved)} retrieved chunks after reranker (threshold={RETRIEVAL_THRESHOLD}).")
for i, r in enumerate(retrieved):
    print(f"  {i+1}. Doc: {r['filename']}, Combined Score: {r['combined_score']:.4f}, Rerank Score: {r.get('rerank_score', '?')}, Text preview: {r['text'][:100]}...")

# service.chat
print("\n[Step 3] Running full service.chat...")
response = service.chat(
    query=query,
    conversation_id=None,
    document_ids=doc_ids
)
print("\nChat Response:")
print("Answer:", response["answer"])
print("Confidence:", response["confidence"])
print("Fallback used:", response["fallback_used"])
print("Retrieved chunks count:", len(response["retrieved_context"]))
