"""
tests/test_rag_multi_doc.py — Integration tests for multi-document summarization in RAG.
"""
import unittest
from pathlib import Path
from backend.services.rag.service import RAGChatService

class TestRagMultiDocSummarization(unittest.TestCase):
    def test_multi_doc_summarize_empty(self):
        service = RAGChatService()
        result = service.summarize_documents(document_ids=[])
        self.assertEqual(result["word_count"], 0)
        self.assertTrue(result["fallback_used"])

    def test_multi_doc_summarize_logic(self):
        service = RAGChatService()
        
        # Insert test chunks to database/repository
        doc_a_id = "doc_a_test"
        doc_b_id = "doc_b_test"
        
        chunks_a = [
            {"id": "c_a1", "document_id": doc_a_id, "filename": "doc_a.txt", "text": "Học máy là một lĩnh vực của trí tuệ nhân tạo.", "chunk_index": 0},
        ]
        chunks_b = [
            {"id": "c_b1", "document_id": doc_b_id, "filename": "doc_b.txt", "text": "Mạng nơ-ron học sâu mô phỏng não bộ con người.", "chunk_index": 0},
        ]
        vectors_a = [[0.0] * 768]
        vectors_b = [[0.0] * 768]
        
        # Save to repo
        service.repository.save_chunks(chunks_a, vectors_a, "mock_model")
        service.repository.save_chunks(chunks_b, vectors_b, "mock_model")
        
        # Summarize both
        result = service.summarize_documents(document_ids=[doc_a_id, doc_b_id])
        
        self.assertEqual(len(result["document_ids"]), 2)
        self.assertTrue(result["word_count"] > 0)
        self.assertIsNotNone(result["summary"])
        
        # Clean up database records
        service.repository.delete_document(doc_a_id)
        service.repository.delete_document(doc_b_id)
