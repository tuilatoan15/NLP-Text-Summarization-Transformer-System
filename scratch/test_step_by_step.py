import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(r"c:\Users\ASUS\Desktop\NLP-Text-Summarization-Transformer-System")
sys.path.insert(0, str(project_root))

sys.stdout.reconfigure(encoding='utf-8')

from backend.services.rag.service import RAGChatService

def test_step_by_step():
    service = RAGChatService()
    doc_id = "81c33bb3-ca15-46c1-b212-27b3b91d80a7"
    query = "Tóm tắt nội dung chính của tài liệu."
    
    # 1. Embed query
    embedding_model = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    query_vector = service.embedding_service.embed_query(query, embedding_model)
    
    # 2. Get candidates
    candidates = service.vector_store.query(
        query_vector=query_vector,
        top_k=30,
        document_ids=[doc_id]
    )
    
    # 3. Filter summary candidates
    summary_candidates = [c for c in candidates if c.get("metadata", {}).get("chunk_type") == "summary"]
    print(f"Candidates found: {len(candidates)}")
    print(f"Summary candidates found: {len(summary_candidates)}")
    
    # 4. Retrieve
    retrieved = service.retriever.retrieve(
        query=query,
        query_vector=query_vector,
        chunks=summary_candidates if summary_candidates else candidates,
        top_k=4,
        threshold=0.25,
        retrieval_mode="hybrid",
        use_reranking=True
    )
    print(f"Retrieved chunks count: {len(retrieved)}")
    for i, r in enumerate(retrieved):
        print(f"Retrieved {i}: ID={r['chunk_id']}, Text={repr(r['text'])}")
        
    # 5. Build answer
    ans_res = service.generator.build_answer(query, retrieved, chat_history=None)
    print("\n=== GENERATED ANSWER ===")
    print(repr(ans_res["answer"]))
    print(f"Model used: {ans_res['model_used']}")
    print(f"Fallback used: {ans_res['fallback_used']}")
    print(f"Confidence: {ans_res['confidence']}")

if __name__ == "__main__":
    test_step_by_step()
