import requests
import json
import sys

# Ensure UTF-8 printing
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

url = "http://localhost:8000/rag/chat"
headers = {"Content-Type": "application/json"}

# Lấy danh sách documents trước để lấy ID thực tế
try:
    doc_res = requests.get("http://localhost:8000/rag/documents")
    doc_res.raise_for_status()
    docs = doc_res.json().get("items", [])
    print(f"Docs from API: {len(docs)}")
    if not docs:
        print("No documents found in API!")
        sys.exit(0)
    
    # Lấy tài liệu đầu tiên
    target_doc = docs[0]
    doc_id = target_doc["id"]
    filename = target_doc["filename"]
    print(f"Target document: {filename} (ID: {doc_id})")
    
    payload = {
        "query": "Tóm tắt ngắn gọn các nội dung cốt lõi của tài liệu.",
        "conversation_id": None,
        "document_ids": [doc_id],
        "top_k": 5,
        "threshold": 0.35,
        "retrieval_mode": "hybrid",
        "use_reranking": True,
        "embedding_model": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        "temperature": 0.2
    }
    
    print("\n--- SENDING POST REQUEST TO /rag/chat ---")
    res = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {res.status_code}")
    if res.status_code == 200:
        data = res.json()
        print(f"Answer: {data.get('answer')}")
        print(f"Confidence: {data.get('confidence')}")
        print(f"Grounded: {data.get('grounded')}")
        print(f"Retrieved context count: {len(data.get('retrieved_context', []))}")
        print(f"Evaluation: {data.get('evaluation')}")
    else:
        print(f"Error detail: {res.text}")
        
except Exception as e:
    print(f"Error calling API: {e}")
