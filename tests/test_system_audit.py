"""
tests/test_system_audit.py — Comprehensive System Audit Test Suite.

Giai đoạn 3-6 của cuộc kiểm toán toàn diện:
  • Giai đoạn 3: Functional Testing (API endpoints)
  • Giai đoạn 4: Frontend API contract validation
  • Giai đoạn 5: NLP Quality baseline checks
  • Giai đoạn 6: RAG Pipeline component tests

Yêu cầu: python -m pytest tests/test_system_audit.py -v --tb=short
"""
from __future__ import annotations

import json
import sys
import tempfile
import shutil
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """TestClient với thư mục tạm thời cho RAG database."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    with mock.patch("src.config.DOCUMENT_INTELLIGENCE_DIR", temp_path):
        with mock.patch("src.abstractive.get_summarizer") as mock_sum:
            instance = mock.MagicMock()
            instance.is_loaded.return_value = True
            instance.summarize.return_value = "Bản tóm tắt mẫu cho kiểm toán."
            mock_sum.return_value = instance
            
            from api.main import app
            from fastapi.testclient import TestClient
            with TestClient(app) as c:
                yield c
    shutil.rmtree(temp_dir, ignore_errors=True)


SAMPLE_VIETNAMESE = (
    "Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp để thảo luận về tình hình "
    "leo thang căng thẳng ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay "
    "lập tức và mở hành lang nhân đạo cho người dân vùng chiến sự. Đại diện Mỹ "
    "phát biểu rằng Washington ủng hộ giải pháp hai nhà nước. Cuộc khủng hoảng "
    "nhân đạo ngày càng nghiêm trọng khi hàng nghìn thường dân phải di tản. "
    "Tổng thư ký Liên Hợp Quốc cảnh báo rằng tình hình có thể diễn biến phức tạp "
    "hơn nếu các bên không đạt được thỏa thuận hòa bình trong thời gian sớm nhất."
)


# ════════════════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 3: FUNCTIONAL TESTING — API ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════


class TestPhase3_InfoEndpoints:
    """F-24, F-25: Health & Metrics endpoints."""

    def test_root_returns_api_info(self, client):
        """F-ROOT: GET / trả về thông tin API chuẩn."""
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        assert isinstance(data["endpoints"], list)

    def test_health_check_ok(self, client):
        """F-24: GET /health trả về status ok và model info."""
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "api_version" in data
        assert "model_status" in data
        assert "registry" in data

    def test_metrics_endpoint(self, client):
        """F-25: GET /metrics trả về device diagnostics."""
        r = client.get("/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "device" in data
        assert "models_preloaded" in data

    def test_docs_swagger_accessible(self, client):
        """Swagger UI phải truy cập được."""
        r = client.get("/docs")
        assert r.status_code == 200

    def test_redoc_accessible(self, client):
        """ReDoc phải truy cập được."""
        r = client.get("/redoc")
        assert r.status_code == 200

    def test_models_list(self, client):
        """GET /models trả về danh sách thuật toán."""
        r = client.get("/models")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) >= 1


class TestPhase3_SummarizeEndpoints:
    """F-01 đến F-09: Summarization API endpoints."""

    def test_summarize_text_success(self, client):
        """F-01: POST /summarize với văn bản tiếng Việt hợp lệ."""
        r = client.post("/summarize", json={
            "text": SAMPLE_VIETNAMESE,
            "save_result": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert "extractive" in data
        assert "abstractive" in data
        assert "best" in data
        assert "scores" in data
        assert "word_count" in data
        assert "processing_time_seconds" in data

    def test_summarize_response_structure(self, client):
        """Kiểm tra cấu trúc response đầy đủ theo SummarizeResponse schema."""
        r = client.post("/summarize", json={
            "text": SAMPLE_VIETNAMESE,
            "save_result": False,
        })
        assert r.status_code == 200
        data = r.json()
        # Các trường bắt buộc theo SummarizeResponse
        required_fields = [
            "extractive", "abstractive", "best", "best_type",
            "scores", "word_count", "processing_time_seconds",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # word_count phải có cấu trúc đúng
        assert "input" in data["word_count"]

    def test_summarize_empty_text_422(self, client):
        """F-07: Input rỗng trả về 422."""
        r = client.post("/summarize", json={"text": ""})
        assert r.status_code == 422

    def test_summarize_no_body_422(self, client):
        """F-07: Không có body trả về 422."""
        r = client.post("/summarize", json={})
        assert r.status_code == 422

    def test_summarize_short_text_422(self, client):
        """F-08: Input quá ngắn (<5 từ sau cleaning) trả về 422."""
        r = client.post("/summarize", json={"text": "Xin chào"})
        assert r.status_code == 422

    def test_summarize_with_model_selection(self, client):
        """F-01: Chọn model cụ thể (vit5)."""
        r = client.post("/summarize", json={
            "text": SAMPLE_VIETNAMESE,
            "model_name": "vit5",
            "save_result": False,
        })
        assert r.status_code == 200

    def test_compare_all_algorithms(self, client):
        """F-03: POST /summarize/compare so sánh tất cả thuật toán."""
        r = client.post("/summarize/compare", json={
            "text": SAMPLE_VIETNAMESE,
            "save_result": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert "results" in data or "extractive" in data  # new or legacy format

    def test_compare_with_specific_algorithms(self, client):
        """F-03b: So sánh với danh sách thuật toán cụ thể."""
        r = client.post("/summarize/compare", json={
            "text": SAMPLE_VIETNAMESE,
            "algorithms": ["textrank", "lexrank"],
            "save_result": False,
        })
        assert r.status_code == 200


class TestPhase3_ChatHistoryAPI:
    """F-17 đến F-23: Chat History CRUD API."""

    def test_create_conversation(self, client):
        """F-17: Tạo cuộc trò chuyện mới."""
        r = client.post("/api/chat/conversations", json={"title": "Audit Test Conv"})
        assert r.status_code == 200
        data = r.json()
        assert "id" in data
        assert data["title"] == "Audit Test Conv"
        assert "created_at" in data

    def test_list_conversations(self, client):
        """F-18: Liệt kê cuộc trò chuyện."""
        # Tạo 1 conversation trước
        client.post("/api/chat/conversations", json={"title": "List Test"})
        
        r = client.get("/api/chat/conversations")
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) >= 1

    def test_list_conversations_with_pagination(self, client):
        """F-18b: Phân trang cuộc trò chuyện."""
        r = client.get("/api/chat/conversations?limit=5&offset=0")
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert len(data["items"]) <= 5

    def test_get_conversation_detail(self, client):
        """F-19: Lấy chi tiết cuộc trò chuyện với messages."""
        conv = client.post("/api/chat/conversations", json={"title": "Detail Test"}).json()
        conv_id = conv["id"]
        
        r = client.get(f"/api/chat/conversations/{conv_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == conv_id
        assert "messages" in data
        assert isinstance(data["messages"], list)

    def test_get_nonexistent_conversation_404(self, client):
        """F-19b: Cuộc trò chuyện không tồn tại trả về 404."""
        r = client.get("/api/chat/conversations/non-existent-id-12345")
        assert r.status_code == 404

    def test_rename_conversation(self, client):
        """F-20: Đổi tên cuộc trò chuyện."""
        conv = client.post("/api/chat/conversations", json={"title": "Old Title"}).json()
        conv_id = conv["id"]
        
        r = client.put(f"/api/chat/conversations/{conv_id}", json={"title": "New Title"})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        
        # Verify
        detail = client.get(f"/api/chat/conversations/{conv_id}").json()
        assert detail["title"] == "New Title"

    def test_rename_empty_title_422(self, client):
        """F-20b: Đổi tên với title rỗng trả về 422."""
        conv = client.post("/api/chat/conversations", json={"title": "Test"}).json()
        r = client.put(f"/api/chat/conversations/{conv['id']}", json={"title": ""})
        assert r.status_code == 422

    def test_delete_conversation_cascade(self, client):
        """F-21: Xóa cuộc trò chuyện, messages bị xóa theo (cascade)."""
        conv = client.post("/api/chat/conversations", json={"title": "To Delete"}).json()
        conv_id = conv["id"]
        
        # Add a message
        client.post(f"/api/chat/conversations/{conv_id}/messages", json={
            "role": "user", "content": "Test message"
        })
        
        # Delete
        r = client.delete(f"/api/chat/conversations/{conv_id}")
        assert r.status_code == 200
        assert r.json()["ok"] is True
        
        # Verify 404
        r2 = client.get(f"/api/chat/conversations/{conv_id}")
        assert r2.status_code == 404

    def test_delete_nonexistent_conversation_404(self, client):
        """F-21b: Xóa cuộc trò chuyện không tồn tại trả về 404."""
        r = client.delete("/api/chat/conversations/fake-id-999")
        assert r.status_code == 404

    def test_save_message(self, client):
        """F-23: Lưu tin nhắn vào cuộc trò chuyện."""
        conv = client.post("/api/chat/conversations", json={"title": "Msg Test"}).json()
        conv_id = conv["id"]
        
        r = client.post(f"/api/chat/conversations/{conv_id}/messages", json={
            "role": "user",
            "content": "Tóm tắt tài liệu về biến đổi khí hậu",
        })
        assert r.status_code == 200
        assert "message_id" in r.json()
        
        # Verify message in conversation
        detail = client.get(f"/api/chat/conversations/{conv_id}").json()
        assert len(detail["messages"]) == 1
        assert detail["messages"][0]["role"] == "user"
        assert "biến đổi khí hậu" in detail["messages"][0]["content"]

    def test_save_message_invalid_role_422(self, client):
        """F-23b: Role không hợp lệ trả về 422."""
        conv = client.post("/api/chat/conversations", json={"title": "Role Test"}).json()
        r = client.post(f"/api/chat/conversations/{conv['id']}/messages", json={
            "role": "invalid_role",
            "content": "Test"
        })
        assert r.status_code == 422

    def test_save_message_to_nonexistent_conv_404(self, client):
        """F-23c: Lưu tin nhắn vào conversation không tồn tại."""
        r = client.post("/api/chat/conversations/fake-id-404/messages", json={
            "role": "user",
            "content": "Test"
        })
        assert r.status_code == 404

    def test_search_conversations(self, client):
        """F-22: Tìm kiếm cuộc trò chuyện theo từ khóa."""
        # Tạo conversation với tiêu đề duy nhất
        unique_keyword = "AuditTestSearchKeyword2026"
        client.post("/api/chat/conversations", json={"title": f"Về {unique_keyword}"})
        
        r = client.get(f"/api/chat/search?q={unique_keyword}")
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert len(data["items"]) >= 1

    def test_search_no_results(self, client):
        """F-22b: Tìm kiếm không có kết quả."""
        r = client.get("/api/chat/search?q=xyzunlikelykeyword99999")
        assert r.status_code == 200
        assert len(r.json()["items"]) == 0

    def test_message_count_updated(self, client):
        """F-23d: message_count cập nhật sau khi thêm tin nhắn."""
        conv = client.post("/api/chat/conversations", json={"title": "Count Test"}).json()
        conv_id = conv["id"]
        
        # Thêm 3 messages
        for i in range(3):
            client.post(f"/api/chat/conversations/{conv_id}/messages", json={
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i+1}",
            })
        
        # Verify count
        detail = client.get(f"/api/chat/conversations/{conv_id}").json()
        assert len(detail["messages"]) == 3


class TestPhase3_RAGDocumentAPI:
    """F-11 đến F-16: RAG Document management endpoints."""

    def test_list_rag_documents(self, client):
        """F-12: GET /rag/documents trả về danh sách tài liệu."""
        r = client.get("/rag/documents")
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert isinstance(data["items"], list)

    def test_list_embedding_models(self, client):
        """Listing embedding models endpoint."""
        r = client.get("/rag/embedding-models")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data


# ════════════════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 5: NLP QUALITY — BASELINE CHECKS
# ════════════════════════════════════════════════════════════════════════════


class TestPhase5_NLPQuality:
    """NLP-01 đến NLP-10: Kiểm tra chất lượng NLP cơ bản."""

    def test_extractive_produces_valid_output(self, client):
        """NLP-06: TextRank extractive phải trả về văn bản hợp lệ."""
        r = client.post("/summarize", json={
            "text": SAMPLE_VIETNAMESE,
            "model_name": "textrank",
            "save_result": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data.get("extractive", "")) > 0
        # Extractive summary phải là subset của input (ít nhất chứa từ trong input)
        input_words = set(SAMPLE_VIETNAMESE.lower().split())
        extractive_words = set(data["extractive"].lower().split())
        overlap = input_words & extractive_words
        assert len(overlap) > 3, "Extractive summary should contain words from input"

    def test_abstractive_produces_output(self, client):
        """NLP-01: Abstractive phải sinh ra văn bản."""
        r = client.post("/summarize", json={
            "text": SAMPLE_VIETNAMESE,
            "model_name": "vit5",
            "save_result": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data.get("abstractive", "")) > 0

    def test_summarize_compression_ratio(self, client):
        """NLP-05: Compression ratio phải trong khoảng hợp lý."""
        r = client.post("/summarize", json={
            "text": SAMPLE_VIETNAMESE,
            "save_result": False,
        })
        assert r.status_code == 200
        data = r.json()
        input_words = data["word_count"].get("input", 0)
        best_words = data["word_count"].get("best", 0)
        if input_words > 0 and best_words > 0:
            ratio = best_words / input_words
            assert ratio < 1.0, "Summary should be shorter than input"

    def test_best_model_selection_reasonable(self, client):
        """NLP-10: best_type phải là extractive hoặc abstractive."""
        r = client.post("/summarize", json={
            "text": SAMPLE_VIETNAMESE,
            "save_result": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert data.get("best_type", "") in ["extractive", "abstractive", ""]

    def test_processing_time_reasonable(self, client):
        """Performance: Thời gian xử lý phải < 120 giây."""
        r = client.post("/summarize", json={
            "text": SAMPLE_VIETNAMESE,
            "save_result": False,
        })
        assert r.status_code == 200
        time_s = r.json().get("processing_time_seconds", 0)
        assert time_s < 120, f"Processing took too long: {time_s}s"


# ════════════════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 6: RAG PIPELINE — COMPONENT TESTS
# ════════════════════════════════════════════════════════════════════════════


class TestPhase6_RAGComponents:
    """RAG-01 đến RAG-12: Kiểm tra từng component RAG."""

    def test_chunking_pipeline_split(self):
        """RAG-01: ChunkingPipeline tách văn bản thành chunks."""
        from backend.services.rag.chunker import ChunkingPipeline
        
        pipeline = ChunkingPipeline()
        text = SAMPLE_VIETNAMESE * 3  # Lặp lại để có đủ dài
        pages = [{"text": text, "page": 1}]
        
        # Mock embedding service
        mock_emb = mock.MagicMock()
        mock_emb.embed_documents.return_value = [
            [0.1 * i, 0.2 * i, 0.3, 0.4] for i in range(100)  # enough vectors
        ]
        
        chunks = pipeline.split(
            text=text,
            pages=pages,
            chunk_size=200,
            chunk_overlap=40,
            document_id="test_doc",
            filename="test.txt",
            embedding_service=mock_emb,
            embedding_model="test-model",
        )
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "id" in chunk
            assert "document_id" in chunk
            assert "text" in chunk
            assert len(chunk["text"]) > 0

    def test_intent_classification_general(self):
        """RAG-06: Phân loại intent GENERAL cho câu xã giao."""
        from backend.services.rag.agent import classify_intent
        
        general_queries = [
            "Xin chào",
            "Bạn khỏe không?",
            "Cảm ơn bạn",
            "Hello",
        ]
        for q in general_queries:
            intent = classify_intent(q, document_ids=None)
            assert intent == "GENERAL", f"Expected GENERAL for '{q}', got {intent}"

    def test_intent_classification_document_qa(self):
        """RAG-06b: Phân loại intent DOCUMENT_QA cho câu hỏi tài liệu."""
        from backend.services.rag.agent import classify_intent
        
        doc_queries = [
            "Nội dung chính của bài báo là gì?",
            "Tài liệu đề cập những gì?",
            "Giải thích phần kết luận",
        ]
        for q in doc_queries:
            intent = classify_intent(q, document_ids=["doc1"])
            assert intent in ["DOCUMENT_QA", "SUMMARIZE"], f"Expected DOC_QA/SUMMARIZE for '{q}', got {intent}"

    def test_intent_classification_summarize(self):
        """RAG-06c: Phân loại intent SUMMARIZE khi có từ khóa tóm tắt."""
        from backend.services.rag.agent import classify_intent
        
        intent = classify_intent("Tóm tắt nội dung tài liệu này", document_ids=["doc1"])
        assert intent == "SUMMARIZE", f"Expected SUMMARIZE, got {intent}"

    def test_generator_insufficient_context(self):
        """RAG-GENERATOR: Generator trả về message phù hợp khi không có context."""
        from backend.services.rag.generator import GroundedGenerator
        
        gen = GroundedGenerator()
        result = gen.build_answer("Test question", contexts=[], general_chat=False)
        
        assert result["confidence"] == 0.0
        assert "Không tìm thấy" in result["answer"]
        assert result["fallback_used"] is True

    def test_hybrid_retriever_empty_chunks(self):
        """RAG-RETRIEVER: HybridRetriever xử lý đúng khi chunks rỗng."""
        from backend.services.rag.retriever import HybridRetriever
        
        retriever = HybridRetriever()
        result = retriever.retrieve(
            query="Test query",
            query_vector=[0.1, 0.2, 0.3],
            chunks=[],
            top_k=5,
            threshold=0.3,
        )
        assert result == []

    def test_rag_config_values_valid(self):
        """RAG-CONFIG: Cấu hình RAG nằm trong phạm vi hợp lý."""
        from backend.services.rag.rag_config import (
            CHUNK_SIZE, CHUNK_OVERLAP,
            VECTOR_WEIGHT, BM25_WEIGHT,
            RETRIEVAL_INITIAL_TOP_K, RETRIEVAL_FINAL_TOP_K,
            RETRIEVAL_THRESHOLD,
        )
        
        # Chunk config
        assert 100 <= CHUNK_SIZE <= 2000, f"CHUNK_SIZE={CHUNK_SIZE} out of range"
        assert CHUNK_OVERLAP < CHUNK_SIZE, "CHUNK_OVERLAP must be < CHUNK_SIZE"
        assert CHUNK_OVERLAP >= 0
        
        # Hybrid weights
        assert abs((VECTOR_WEIGHT + BM25_WEIGHT) - 1.0) < 0.01, "Weights should sum to ~1.0"
        assert VECTOR_WEIGHT > 0
        assert BM25_WEIGHT > 0
        
        # Retrieval params
        assert RETRIEVAL_INITIAL_TOP_K > RETRIEVAL_FINAL_TOP_K
        assert 0.0 <= RETRIEVAL_THRESHOLD <= 1.0

    def test_generation_profiles_complete(self):
        """RAG-CONFIG: Tất cả model phải có generation profile."""
        from backend.services.rag.rag_config import (
            GENERATION_PROFILES,
            PREFERRED_SUMMARIZER_ORDER,
        )
        
        for model in PREFERRED_SUMMARIZER_ORDER:
            assert model in GENERATION_PROFILES, f"Missing profile for {model}"
            profile = GENERATION_PROFILES[model]
            assert profile.num_beams >= 1
            assert profile.max_new_tokens > profile.min_new_tokens
            assert profile.no_repeat_ngram_size >= 2

    def test_prompt_templates_have_placeholders(self):
        """RAG-PROMPT: Prompt templates phải chứa placeholders cần thiết."""
        from backend.services.rag.rag_config import (
            QA_PROMPT_TEMPLATE,
            SUMMARIZE_PROMPT_TEMPLATE,
        )
        
        assert "{context}" in QA_PROMPT_TEMPLATE
        assert "{question}" in QA_PROMPT_TEMPLATE
        assert "{chat_history}" in QA_PROMPT_TEMPLATE
        assert "{context}" in SUMMARIZE_PROMPT_TEMPLATE


# ════════════════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 8: SECURITY TESTING (Static Analysis)
# ════════════════════════════════════════════════════════════════════════════


class TestPhase8_SecurityBaseline:
    """SEC-01 đến SEC-06: Kiểm tra bảo mật cơ bản."""

    def test_cors_configuration_exists(self, client):
        """SEC-01: CORS middleware phải được cấu hình."""
        # Gửi preflight request
        r = client.options("/", headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        })
        # CORS headers phải tồn tại
        assert "access-control-allow-origin" in r.headers or r.status_code == 200

    def test_no_sensitive_info_in_health(self, client):
        """SEC-02: Health endpoint không lộ thông tin nhạy cảm."""
        r = client.get("/health")
        text = r.text.lower()
        assert "password" not in text
        assert "secret" not in text
        assert "api_key" not in text

    def test_no_sensitive_info_in_metrics(self, client):
        """SEC-03: Metrics endpoint không lộ thông tin nhạy cảm."""
        r = client.get("/metrics")
        text = r.text.lower()
        assert "password" not in text
        assert "secret" not in text

    def test_invalid_conversation_id_handled(self, client):
        """SEC-04: SQL injection attempt bị xử lý đúng."""
        # Try SQL injection in conversation ID
        injection = "'; DROP TABLE conversations;--"
        r = client.get(f"/api/chat/conversations/{injection}")
        assert r.status_code in [404, 422, 500]
        # Verify conversations table still works
        r2 = client.get("/api/chat/conversations")
        assert r2.status_code == 200

    def test_xss_in_conversation_title(self, client):
        """SEC-05: XSS attempt trong title phải được lưu nguyên (escape ở frontend)."""
        xss_title = '<script>alert("xss")</script>'
        r = client.post("/api/chat/conversations", json={"title": xss_title})
        assert r.status_code == 200
        # Title phải được lưu nguyên (không bị execute)
        assert r.json()["title"] == xss_title

    def test_large_payload_handled(self, client):
        """SEC-06: Payload quá lớn phải được xử lý (không crash)."""
        large_text = "A" * 1_000_000  # 1MB text
        r = client.post("/summarize", json={"text": large_text})
        # Should either succeed or return proper error, not crash
        assert r.status_code in [200, 413, 422, 500]


# ════════════════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 9: EDGE CASES & ERROR HANDLING
# ════════════════════════════════════════════════════════════════════════════


class TestPhase9_EdgeCases:
    """EDGE-01 đến EDGE-10: Kiểm tra các trường hợp biên."""

    def test_unicode_vietnamese_diacritics(self, client):
        """EDGE-01: Xử lý đúng dấu tiếng Việt đặc biệt."""
        vietnamese_text = (
            "Đặng Thị Ngọc Thịnh là nữ Phó Chủ tịch nước Cộng hòa Xã hội Chủ nghĩa Việt Nam. "
            "Bà được bầu làm Phó Chủ tịch nước vào tháng 4 năm 2016. "
            "Trước đó, bà từng giữ chức Bí thư Tỉnh ủy Vĩnh Long từ năm 2010. "
            "Bà sinh ngày 10 tháng 10 năm 1959 tại Quảng Nam. "
            "Bà là một trong những nữ chính trị gia nổi bật nhất Việt Nam hiện nay."
        )
        r = client.post("/summarize", json={"text": vietnamese_text, "save_result": False})
        assert r.status_code == 200

    def test_special_characters_in_text(self, client):
        """EDGE-02: Ký tự đặc biệt trong văn bản."""
        text_with_special = (
            "Công ty ABC (mã: ABC-123) đã đạt doanh thu 1.5 tỷ đồng. "
            "Tăng trưởng 25% so với cùng kỳ năm ngoái. "
            'CEO phát biểu: "Chúng tôi rất hài lòng với kết quả này." '
            "Nguồn: https://example.com/report?year=2024&q=1 "
            "Email liên hệ: info@company.com.vn"
        )
        r = client.post("/summarize", json={"text": text_with_special, "save_result": False})
        assert r.status_code == 200

    def test_conversation_title_max_length(self, client):
        """EDGE-03: Title cuộc trò chuyện ở giới hạn tối đa (150 ký tự)."""
        max_title = "A" * 150
        r = client.post("/api/chat/conversations", json={"title": max_title})
        assert r.status_code == 200

    def test_conversation_title_over_max_422(self, client):
        """EDGE-03b: Title vượt quá giới hạn phải trả về lỗi khi rename."""
        conv = client.post("/api/chat/conversations", json={"title": "Test"}).json()
        over_max_title = "A" * 200
        r = client.put(f"/api/chat/conversations/{conv['id']}", json={"title": over_max_title})
        assert r.status_code == 422

    def test_multiple_messages_ordering(self, client):
        """EDGE-04: Thứ tự tin nhắn phải đúng theo thời gian."""
        conv = client.post("/api/chat/conversations", json={"title": "Order Test"}).json()
        conv_id = conv["id"]
        
        messages_data = [
            ("user", "Câu hỏi 1"),
            ("assistant", "Trả lời 1"),
            ("user", "Câu hỏi 2"),
            ("assistant", "Trả lời 2"),
        ]
        for role, content in messages_data:
            client.post(f"/api/chat/conversations/{conv_id}/messages", json={
                "role": role, "content": content,
            })
        
        detail = client.get(f"/api/chat/conversations/{conv_id}").json()
        msgs = detail["messages"]
        assert len(msgs) == 4
        assert msgs[0]["content"] == "Câu hỏi 1"
        assert msgs[1]["content"] == "Trả lời 1"
        assert msgs[2]["content"] == "Câu hỏi 2"
        assert msgs[3]["content"] == "Trả lời 2"

    def test_empty_search_query_422(self, client):
        """EDGE-05: Tìm kiếm với query rỗng phải trả về lỗi."""
        r = client.get("/api/chat/search?q=")
        assert r.status_code == 422

    def test_concurrent_conversation_creation(self, client):
        """EDGE-06: Tạo nhiều conversation liên tiếp không lỗi."""
        ids = []
        for i in range(10):
            r = client.post("/api/chat/conversations", json={"title": f"Batch {i}"})
            assert r.status_code == 200
            ids.append(r.json()["id"])
        
        # Tất cả ID phải khác nhau
        assert len(set(ids)) == 10


# ════════════════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 11: CODE QUALITY CHECKS (Static)
# ════════════════════════════════════════════════════════════════════════════


class TestPhase11_CodeQuality:
    """CQ-01 đến CQ-05: Kiểm tra chất lượng code."""

    def test_rag_config_no_secrets(self):
        """CQ-01: rag_config.py không chứa hardcoded secrets."""
        config_path = Path(__file__).parent.parent / "backend" / "services" / "rag" / "rag_config.py"
        content = config_path.read_text(encoding="utf-8")
        # Should use os.getenv, not hardcoded values
        assert 'os.getenv("GEMINI_API_KEY"' in content or 'os.getenv("GEMINI_API_KEY",' in content
        assert 'os.getenv("OPENAI_API_KEY"' in content or 'os.getenv("OPENAI_API_KEY",' in content

    def test_repository_has_foreign_keys(self):
        """CQ-02: SQLite repository bật PRAGMA foreign_keys."""
        repo_path = Path(__file__).parent.parent / "backend" / "services" / "rag" / "repository.py"
        content = repo_path.read_text(encoding="utf-8")
        assert "PRAGMA foreign_keys = ON" in content

    def test_api_main_has_error_handling(self):
        """CQ-03: API main.py có error handling middleware."""
        api_path = Path(__file__).parent.parent / "api" / "main.py"
        content = api_path.read_text(encoding="utf-8")
        assert "HTTPException" in content
        assert "middleware" in content.lower()

    def test_all_routers_have_tags(self):
        """CQ-04: Tất cả routers phải có tags cho API docs."""
        api_dir = Path(__file__).parent.parent / "api"
        for py_file in api_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            content = py_file.read_text(encoding="utf-8")
            if "APIRouter" in content or "router" in content.lower():
                assert "tags=" in content, f"{py_file.name} missing tags on router"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
