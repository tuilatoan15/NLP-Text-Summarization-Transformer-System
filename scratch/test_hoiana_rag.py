import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(r"c:\Users\ASUS\Desktop\NLP-Text-Summarization-Transformer-System")
sys.path.insert(0, str(project_root))

sys.stdout.reconfigure(encoding='utf-8')

from backend.services.rag.service import RAGChatService

def test_hoiana_rag():
    service = RAGChatService()
    doc_id = "81c33bb3-ca15-46c1-b212-27b3b91d80a7"
    
    docs = service.list_documents()
    target_doc = None
    for doc in docs:
        if doc['id'] == doc_id:
            target_doc = doc
            break
            
    if not target_doc:
        print(f"❌ Không tìm thấy tài liệu {doc_id} trong DB!")
        return

    print(f"\n🔬 Đang chạy thử nghiệm với tài liệu: {target_doc['filename']} (ID: {doc_id})")

    # 1. Chạy chat thực tế
    print("\n--- 💬 CHẠY THỬ CHAT() ---")
    res = service.chat(
        query="Tóm tắt nội dung chính của tài liệu.",
        conversation_id=None,
        document_ids=[doc_id],
    )
    print(f"Answer:\n{res['answer']}")
    print(f"Confidence: {res['confidence']}")
    print(f"Grounded: {res['grounded']}")
    print(f"Fallback used: {res.get('fallback_used')}")
    print(f"Model used: {res.get('model_used')}")

if __name__ == "__main__":
    test_hoiana_rag()
