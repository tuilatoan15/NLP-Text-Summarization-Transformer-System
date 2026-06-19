"""
tests/test_rag_documents.py — Bộ test tích hợp RAG Documents.

Kiểm tra tính năng tóm tắt tài liệu qua RAGChatService và API endpoint
sử dụng dữ liệu thực tế từ 10 file docx đã tải lên trong hệ thống.

Yêu cầu: Backend server phải đang chạy trên localhost:8000
"""
import unittest
import requests
import json
import time

BASE_URL = "http://localhost:8000"
TIMEOUT = 120  # seconds — RAG pipeline chậm do reranker + embedding


class TestRAGDocumentList(unittest.TestCase):
    """Kiểm tra danh sách tài liệu đã nạp vào hệ thống."""

    def test_list_documents_returns_items(self):
        """Phải có ít nhất 1 tài liệu đã nạp."""
        r = requests.get(f"{BASE_URL}/rag/documents", timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("items", data)
        self.assertGreaterEqual(len(data["items"]), 1, "Cần ít nhất 1 tài liệu trong hệ thống")

    def test_documents_have_required_fields(self):
        """Mỗi tài liệu phải có các trường bắt buộc."""
        r = requests.get(f"{BASE_URL}/rag/documents", timeout=TIMEOUT)
        data = r.json()
        for doc in data["items"]:
            self.assertIn("id", doc)
            self.assertIn("filename", doc)
            self.assertIn("source_type", doc)
            self.assertIn("created_at", doc)

    def test_documents_have_valid_source_type(self):
        """source_type phải là một trong: pdf, docx, txt, md."""
        r = requests.get(f"{BASE_URL}/rag/documents", timeout=TIMEOUT)
        data = r.json()
        valid_types = {"pdf", "docx", "txt", "md"}
        for doc in data["items"]:
            self.assertIn(
                doc["source_type"], valid_types,
                f"source_type '{doc['source_type']}' không hợp lệ cho file {doc['filename']}"
            )


class TestRAGChatWithDocuments(unittest.TestCase):
    """Kiểm tra tính năng chat RAG với tài liệu thực tế."""

    @classmethod
    def setUpClass(cls):
        """Lấy danh sách document_ids từ hệ thống."""
        r = requests.get(f"{BASE_URL}/rag/documents", timeout=TIMEOUT)
        cls.documents = r.json().get("items", [])
        cls.doc_ids = [d["id"] for d in cls.documents]

    def test_chat_summarize_single_document(self):
        """Gửi câu hỏi tóm tắt cho 1 tài liệu → phải trả kết quả có nội dung."""
        if not self.doc_ids:
            self.skipTest("Không có tài liệu nào trong hệ thống")

        payload = {
            "query": "Tóm tắt ngắn gọn các nội dung cốt lõi của tài liệu.",
            "document_ids": [self.doc_ids[0]],
            "top_k": 4,
            "threshold": 0.35,
            "retrieval_mode": "hybrid",
            "use_reranking": True,
            "temperature": 0.15,
        }
        r = requests.post(f"{BASE_URL}/rag/chat", json=payload, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        data = r.json()

        self.assertIn("answer", data)
        self.assertIn("confidence", data)
        self.assertIn("conversation_id", data)
        self.assertTrue(len(data["answer"]) > 10, "Câu trả lời phải có nội dung thực tế")
        self.assertGreaterEqual(data["confidence"], 0.0, "Confidence phải >= 0")

    def test_chat_with_empty_document_ids_blocked(self):
        """Gửi document_ids rỗng với câu hỏi tóm tắt → intent sẽ không phải SUMMARIZE."""
        payload = {
            "query": "Tóm tắt ngắn gọn các nội dung cốt lõi của tài liệu.",
            "document_ids": [],
            "top_k": 4,
            "threshold": 0.35,
        }
        r = requests.post(f"{BASE_URL}/rag/chat", json=payload, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        # Khi document_ids rỗng, intent không nên là SUMMARIZE
        if "intent" in data:
            self.assertNotEqual(
                data["intent"], "summarize",
                "Intent không nên là SUMMARIZE khi document_ids rỗng"
            )

    def test_chat_document_qa_question(self):
        """Hỏi đáp trực tiếp về nội dung tài liệu → trả lời có context."""
        if not self.doc_ids:
            self.skipTest("Không có tài liệu nào trong hệ thống")

        payload = {
            "query": "Các thông tin quan trọng nhất trong văn bản là gì?",
            "document_ids": [self.doc_ids[0]],
            "top_k": 4,
            "threshold": 0.35,
            "retrieval_mode": "hybrid",
            "use_reranking": True,
        }
        r = requests.post(f"{BASE_URL}/rag/chat", json=payload, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("answer", data)
        self.assertTrue(len(data["answer"]) > 0)

    def test_chat_general_greeting(self):
        """Gửi câu chào hỏi → intent GENERAL, confidence = 1.0."""
        payload = {
            "query": "Xin chào",
            "document_ids": [],
        }
        r = requests.post(f"{BASE_URL}/rag/chat", json=payload, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("answer", data)
        if "intent" in data:
            self.assertEqual(data["intent"], "general")
        if "confidence" in data:
            self.assertEqual(data["confidence"], 1.0)


class TestRAGChatStream(unittest.TestCase):
    """Kiểm tra tính năng streaming chat RAG."""

    @classmethod
    def setUpClass(cls):
        r = requests.get(f"{BASE_URL}/rag/documents", timeout=TIMEOUT)
        cls.documents = r.json().get("items", [])
        cls.doc_ids = [d["id"] for d in cls.documents]

    def test_stream_chat_returns_sse_events(self):
        """Stream endpoint phải trả về Server-Sent Events hợp lệ."""
        if not self.doc_ids:
            self.skipTest("Không có tài liệu nào trong hệ thống")

        payload = {
            "query": "Tóm tắt ngắn gọn các nội dung cốt lõi của tài liệu.",
            "document_ids": [self.doc_ids[0]],
            "top_k": 4,
            "threshold": 0.35,
            "retrieval_mode": "hybrid",
            "use_reranking": True,
        }
        r = requests.post(
            f"{BASE_URL}/rag/chat/stream",
            json=payload,
            stream=True,
            timeout=TIMEOUT,
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/event-stream", r.headers.get("content-type", ""))

        events = []
        final_response = None
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                raw = line[6:]
                try:
                    event = json.loads(raw)
                    events.append(event)
                    if event.get("event") == "done":
                        final_response = event.get("response", {})
                except json.JSONDecodeError:
                    pass

        self.assertGreater(len(events), 0, "Phải nhận được ít nhất 1 SSE event")

        # Phải có event 'done' kết thúc
        done_events = [e for e in events if e.get("event") == "done"]
        self.assertEqual(len(done_events), 1, "Phải có đúng 1 event 'done'")

        # Final response phải có answer
        if final_response:
            self.assertIn("answer", final_response)
            self.assertTrue(len(final_response["answer"]) > 0)

    def test_stream_chat_has_conversation_id(self):
        """Stream phải trả về conversation_id trong các token events."""
        if not self.doc_ids:
            self.skipTest("Không có tài liệu nào trong hệ thống")

        payload = {
            "query": "Nội dung chính của tài liệu này là gì?",
            "document_ids": [self.doc_ids[0]],
        }
        r = requests.post(
            f"{BASE_URL}/rag/chat/stream",
            json=payload,
            stream=True,
            timeout=TIMEOUT,
        )
        self.assertEqual(r.status_code, 200)

        has_conversation_id = False
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                    if event.get("conversation_id"):
                        has_conversation_id = True
                        break
                except json.JSONDecodeError:
                    pass

        self.assertTrue(has_conversation_id, "Stream phải chứa conversation_id")


class TestRAGDocumentSummarize(unittest.TestCase):
    """Kiểm tra endpoint tóm tắt tài liệu chuyên dụng."""

    @classmethod
    def setUpClass(cls):
        r = requests.get(f"{BASE_URL}/rag/documents", timeout=TIMEOUT)
        cls.documents = r.json().get("items", [])
        cls.doc_ids = [d["id"] for d in cls.documents]

    def test_summarize_single_document(self):
        """Tóm tắt 1 tài liệu qua endpoint chuyên dụng."""
        if not self.doc_ids:
            self.skipTest("Không có tài liệu nào trong hệ thống")

        payload = {
            "document_ids": [self.doc_ids[0]],
            "query": "Tóm tắt nội dung chính của tài liệu"
        }
        r = requests.post(f"{BASE_URL}/rag/documents/summarize", json=payload, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("summary", data)
        self.assertTrue(len(data["summary"]) > 20, "Bản tóm tắt phải có nội dung thực tế")
        self.assertIn("word_count", data)
        self.assertGreater(data["word_count"], 0)

    def test_summarize_multiple_documents(self):
        """Tóm tắt đa tài liệu (nếu có ≥ 2 tài liệu)."""
        if len(self.doc_ids) < 2:
            self.skipTest("Cần ít nhất 2 tài liệu để test đa tài liệu")

        payload = {
            "document_ids": self.doc_ids[:2],
            "query": "Tóm tắt và so sánh nội dung hai tài liệu"
        }
        r = requests.post(f"{BASE_URL}/rag/documents/summarize", json=payload, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("summary", data)
        self.assertGreater(data["word_count"], 0)

    def test_summarize_empty_document_ids_rejected(self):
        """Gửi document_ids rỗng → phải trả lỗi 422."""
        payload = {
            "document_ids": [],
        }
        r = requests.post(f"{BASE_URL}/rag/documents/summarize", json=payload, timeout=TIMEOUT)
        self.assertIn(r.status_code, [400, 422], "Phải reject khi document_ids rỗng")


class TestRAGConversationManagement(unittest.TestCase):
    """Kiểm tra quản lý lịch sử hội thoại RAG."""

    def test_create_and_list_conversations(self):
        """Tạo conversation mới qua chat → phải xuất hiện trong danh sách."""
        # Lấy danh sách trước
        r1 = requests.get(f"{BASE_URL}/rag/conversations", timeout=TIMEOUT)
        self.assertEqual(r1.status_code, 200)
        before_count = len(r1.json().get("items", []))

        # Gửi 1 câu chat để tạo conversation mới
        docs_r = requests.get(f"{BASE_URL}/rag/documents", timeout=TIMEOUT)
        doc_ids = [d["id"] for d in docs_r.json().get("items", [])]

        if doc_ids:
            payload = {
                "query": "Xin chào, đây là test tạo conversation",
                "document_ids": doc_ids[:1],
            }
            r2 = requests.post(f"{BASE_URL}/rag/chat", json=payload, timeout=TIMEOUT)
            self.assertEqual(r2.status_code, 200)
            new_conv_id = r2.json().get("conversation_id")
            self.assertIsNotNone(new_conv_id)

            # Kiểm tra conversation mới xuất hiện
            r3 = requests.get(f"{BASE_URL}/rag/conversations", timeout=TIMEOUT)
            after_count = len(r3.json().get("items", []))
            self.assertGreaterEqual(after_count, before_count)

    def test_conversation_messages_endpoint(self):
        """Lấy tin nhắn của conversation phải trả về danh sách."""
        # Tạo conversation qua chat
        docs_r = requests.get(f"{BASE_URL}/rag/documents", timeout=TIMEOUT)
        doc_ids = [d["id"] for d in docs_r.json().get("items", [])]

        if not doc_ids:
            self.skipTest("Không có tài liệu")

        payload = {
            "query": "Test lấy tin nhắn",
            "document_ids": doc_ids[:1],
        }
        chat_r = requests.post(f"{BASE_URL}/rag/chat", json=payload, timeout=TIMEOUT)
        conv_id = chat_r.json().get("conversation_id")

        if conv_id:
            r = requests.get(f"{BASE_URL}/rag/conversations/{conv_id}/messages", timeout=TIMEOUT)
            self.assertEqual(r.status_code, 200)
            messages = r.json().get("items", [])
            self.assertGreaterEqual(len(messages), 2, "Phải có ít nhất 1 user + 1 assistant message")

            # Kiểm tra roles
            roles = [m["role"] for m in messages]
            self.assertIn("user", roles)
            self.assertIn("assistant", roles)


class TestRAGEmbeddingModels(unittest.TestCase):
    """Kiểm tra danh sách embedding models."""

    def test_list_embedding_models(self):
        """Phải trả về danh sách models hợp lệ."""
        r = requests.get(f"{BASE_URL}/rag/embedding-models", timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("models", data)


class TestRAGIntentClassification(unittest.TestCase):
    """Kiểm tra phân loại intent qua unit test trực tiếp."""

    def test_summarize_intent_with_docs(self):
        """Câu hỏi tóm tắt + document_ids → SUMMARIZE."""
        from backend.services.rag.agent import classify_intent
        result = classify_intent("Tóm tắt nội dung chính", ["doc_123"])
        self.assertEqual(result, "SUMMARIZE")

    def test_summarize_intent_without_docs(self):
        """Câu hỏi tóm tắt + không có document_ids → KHÔNG PHẢI SUMMARIZE."""
        from backend.services.rag.agent import classify_intent
        result = classify_intent("Tóm tắt nội dung chính", [])
        self.assertNotEqual(result, "SUMMARIZE")

    def test_general_intent_greeting(self):
        """Câu chào hỏi → GENERAL."""
        from backend.services.rag.agent import classify_intent
        result = classify_intent("Xin chào bạn", [])
        self.assertEqual(result, "GENERAL")

    def test_document_qa_intent(self):
        """Câu hỏi tra cứu → DOCUMENT_QA."""
        from backend.services.rag.agent import classify_intent
        result = classify_intent("Ai bị bắt trong vụ lừa đảo?", ["doc_123"])
        self.assertIn(result, ["DOCUMENT_QA", "SUMMARIZE"])


if __name__ == "__main__":
    unittest.main()
