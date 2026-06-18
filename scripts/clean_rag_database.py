#!/usr/bin/env python3
"""
scripts/clean_rag_database.py
Dọn dẹp và khôi phục cơ sở dữ liệu RAG (SQLite + Vector Store Qdrant/Chroma)
bằng cách quét và loại bỏ các prompt hệ thống bị index nhầm trong các chunk tóm tắt.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from backend.services.rag.repository import RAGRepository
from backend.services.rag.vector_store import VectorStoreManager
from backend.services.rag.embedding_service import EmbeddingService
from src.utils import logger

# Các mẫu prompt hệ thống cần loại bỏ
PROMPT_PATTERNS = [
    r"Bạn là trợ lý phân tích văn bản\.\s*Hãy viết một đoạn tóm tắt ngắn gọn,\s*mạch lạc khoảng 3-5 câu\s*bao quát toàn bộ các thông tin cốt lõi trong đoạn văn bản sau,\s*không tự bịa đặt thông tin:\s*",
    r"Hãy viết một đoạn tóm tắt ngắn gọn,\s*mạch lạc khoảng 3-5 câu\s*bao quát toàn bộ các thông tin cốt lõi trong đoạn văn bản sau,\s*không tự bịa đặt thông tin:\s*",
    r"Bạn là trợ lý phân tích văn bản\.\s*Hãy viết một đoạn tóm tắt ngắn gọn,\s*mạch lạc khoảng 3-5 câu\s*bao quát toàn bộ các thông tin cốt lõi trong đoạn văn bản sau,\s*không tự bịa đặt thông tin:\s*\n*",
    r"Bản tóm tắt tiếng Việt cô đọng:\s*"
]

def clean_text(text: str) -> tuple[str, bool]:
    """
    Quét và loại bỏ các prompt rác khỏi văn bản chunk.
    Trả về (văn bản đã dọn dẹp, True nếu có thay đổi).
    """
    cleaned = text
    changed = False
    
    for pattern in PROMPT_PATTERNS:
        matches = list(re.finditer(pattern, cleaned, re.IGNORECASE))
        if matches:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
            changed = True
            
    # Xử lý các dấu hai chấm thừa ở đầu sau khi xóa prompt
    if changed:
        cleaned = cleaned.strip()
        # Nếu bắt đầu bằng dấu hai chấm hoặc khoảng trắng
        cleaned = re.sub(r"^:\s*", "", cleaned)
        cleaned = cleaned.strip()
        
    return cleaned, changed

def main():
    logger.info("🧹 Bắt đầu tiến trình chẩn đoán và dọn dẹp cơ sở dữ liệu RAG...")
    
    rag_dir = config.DOCUMENT_INTELLIGENCE_DIR / "rag"
    db_path = rag_dir / "rag_chat.db"
    
    if not db_path.exists():
        logger.error(f"❌ Không tìm thấy cơ sở dữ liệu SQLite tại {db_path}")
        return
        
    repository = RAGRepository(db_path)
    vector_store = VectorStoreManager(rag_dir)
    embedding_service = EmbeddingService()
    
    # Lấy danh sách tất cả các tài liệu
    documents = repository.list_documents()
    logger.info(f"📁 Tìm thấy {len(documents)} tài liệu trong database.")
    
    total_chunks_scanned = 0
    total_chunks_cleaned = 0
    
    for doc in documents:
        doc_id = doc["id"]
        filename = doc["filename"]
        logger.info(f"🔍 Quét tài liệu: {filename} (ID: {doc_id})")
        
        # Lấy tất cả các chunk của tài liệu này
        chunks = repository.list_chunks(document_ids=[doc_id])
        logger.info(f"   -> Tìm thấy {len(chunks)} chunks.")
        
        cleaned_chunks = []
        cleaned_vectors = []
        
        for chunk in chunks:
            total_chunks_scanned += 1
            text = chunk["text"]
            
            cleaned_txt, is_changed = clean_text(text)
            
            if is_changed:
                total_chunks_cleaned += 1
                logger.info(f"   [!] Phát hiện chunk nhiễm prompt (ID: {chunk['id']})")
                logger.info(f"       - Trước: {text[:100]}...")
                logger.info(f"       - Sau:  {cleaned_txt[:100]}...")
                
                # Cập nhật văn bản mới vào dictionary chunk
                chunk["text"] = cleaned_txt
                
                # Tạo embedding mới phù hợp với văn bản đã dọn dẹp
                model_name = chunk.get("embedding_model", config.DEFAULT_EMBEDDING_MODEL)
                new_vector = embedding_service.embed_documents([cleaned_txt], model_name)[0]
                
                cleaned_chunks.append({
                    "id": chunk["id"],
                    "document_id": chunk["document_id"],
                    "filename": chunk["filename"],
                    "page": chunk["page"],
                    "chunk_index": chunk["chunk_index"],
                    "text": cleaned_txt,
                    "metadata": chunk.get("metadata", {})
                })
                cleaned_vectors.append(new_vector)
                
        if cleaned_chunks:
            logger.info(f"   💾 Đang cập nhật {len(cleaned_chunks)} chunks đã dọn dẹp vào SQLite và Vector Store...")
            # 1. Lưu lại vào SQLite database
            repository.save_chunks(cleaned_chunks, cleaned_vectors, config.DEFAULT_EMBEDDING_MODEL)
            # 2. Lưu lại vào Vector Store (Qdrant/Chroma)
            vector_store.upsert_chunks(cleaned_chunks, cleaned_vectors)
            logger.info(f"   ✅ Đã cập nhật xong tài liệu: {filename}")
            
    logger.info("=========================================================")
    logger.info("🎉 HOÀN THÀNH TIẾN TRÌNH DỌN DẸP RAG DATABASE!")
    logger.info(f"   - Tổng số chunks đã quét: {total_chunks_scanned}")
    logger.info(f"   - Tổng số chunks đã sửa đổi/làm sạch: {total_chunks_cleaned}")
    logger.info("=========================================================")

if __name__ == "__main__":
    main()
