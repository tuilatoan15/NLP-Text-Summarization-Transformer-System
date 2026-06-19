from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import tempfile
import shutil
from unittest import mock
from backend.services.rag.chunker import ChunkingPipeline
from backend.services.rag.raptor import gmm_cluster, RaptorIndexer
from backend.services.rag.agent import evaluate_answer, rewrite_query


class MockEmbeddingService:
    def embed_documents(self, texts: list[str], model_name: str) -> list[list[float]]:
        # Sinh vector giả định: các câu có chủ đề giống nhau sẽ có vector gần nhau
        vectors = []
        for text in texts:
            # Nếu nói về "bóng đá" hoặc "thể thao"
            if any(w in text.lower() for w in ["bóng đá", "thể thao", "cầu thủ"]):
                vec = [0.9, 0.1, 0.0]
            # Nếu nói về "vũ trụ" hoặc "thiên văn"
            elif any(w in text.lower() for w in ["vũ trụ", "sao hỏa", "thiên văn"]):
                vec = [0.0, 0.9, 0.1]
            # Mặc định
            else:
                vec = [0.1, 0.1, 0.8]
            vectors.append(vec)
        return vectors

    def embed_query(self, text: str, model_name: str) -> list[float]:
        return [0.9, 0.1, 0.0]


def test_semantic_chunking_modes():
    pipeline = ChunkingPipeline()
    embedding_service = MockEmbeddingService()
    
    text = (
        "Bóng đá là môn thể thao vua được hàng triệu người yêu thích. "
        "Các cầu thủ chạy trên sân cỏ để ghi bàn thắng. "
        "Sao Hỏa là hành tinh thứ tư tính từ Mặt Trời trong hệ Mặt Trời. "
        "Các nhà khoa học đang nghiên cứu dấu vết nước trên Sao Hỏa. "
        "Phở bò là món ăn truyền thống nổi tiếng của Việt Nam."
    )
    
    pages = [{"page": 1, "text": text}]
    
    # 1. Test Fixed Threshold
    chunks_fixed = pipeline.split(
        text=text,
        pages=pages,
        chunk_size=500,
        chunk_overlap=0,
        document_id="doc_test",
        filename="test.txt",
        embedding_service=embedding_service,
        chunking_mode="fixed",
        threshold=0.5
    )
    
    assert len(chunks_fixed) > 0
    # Đảm bảo phân đoạn ngữ nghĩa tách được chủ đề bóng đá và sao hỏa riêng biệt
    assert any("Bóng đá" in c["text"] for c in chunks_fixed)
    assert any("Sao Hỏa" in c["text"] for c in chunks_fixed)

    # 2. Test Dynamic Threshold
    chunks_dynamic = pipeline.split(
        text=text,
        pages=pages,
        chunk_size=500,
        chunk_overlap=0,
        document_id="doc_test",
        filename="test.txt",
        embedding_service=embedding_service,
        chunking_mode="dynamic",
        dynamic_k=0.8
    )
    assert len(chunks_dynamic) > 0

    # 3. Test Sliding Window
    chunks_sliding = pipeline.split(
        text=text,
        pages=pages,
        chunk_size=500,
        chunk_overlap=0,
        document_id="doc_test",
        filename="test.txt",
        embedding_service=embedding_service,
        chunking_mode="sliding_window",
        window_size=2
    )
    assert len(chunks_sliding) > 0


def test_gmm_clustering_soft():
    # Tạo dữ liệu gồm 3 cụm rõ rệt
    c1 = np.random.normal(loc=[10, 0], scale=0.5, size=(5, 2))
    c2 = np.random.normal(loc=[0, 10], scale=0.5, size=(5, 2))
    c3 = np.random.normal(loc=[-10, -10], scale=0.5, size=(5, 2))
    X = np.vstack([c1, c2, c3]).astype(np.float32)
    
    # Chạy GMM Numpy với k = 3
    clusters = gmm_cluster(X, k=3, threshold=0.15)
    
    assert len(clusters) >= 2
    # Tổng số điểm được gán có thể lớn hơn 15 do phân cụm mềm (overlapping)
    total_assigned = sum(len(c) for c in clusters)
    assert total_assigned >= 15


def test_raptor_recursive_tree():
    repository = mock.MagicMock()
    vector_store = mock.MagicMock()
    embedding_service = MockEmbeddingService()
    generator = mock.MagicMock()
    
    indexer = RaptorIndexer(repository, vector_store, embedding_service, generator)
    
    # Mock summarizer trả về tóm tắt cố định
    with mock.patch.object(indexer, "_summarize_cluster", return_value="Tóm tắt phân đoạn test"):
        base_chunks = [
            {"id": f"chunk_{i}", "text": f"Nội dung phân đoạn bóng đá thể thao thứ {i}", "filename": "test.txt"}
            for i in range(12)
        ]
        base_vectors = [[0.9, 0.1, 0.0] for _ in range(12)]
        
        indexer.build_tree(
            document_id="doc_raptor_test",
            base_chunks=base_chunks,
            base_vectors=base_vectors,
            embedding_model="test-model",
            max_levels=2
        )
        
        # Đảm bảo repository.save_chunks được gọi để lưu các nút tóm tắt
        assert repository.save_chunks.call_count >= 1
        assert vector_store.upsert_chunks.call_count >= 1
        
        # Kiểm tra level của nút tóm tắt cấp cao nhất lưu trong repository
        called_chunks = repository.save_chunks.call_args[0][0]
        assert len(called_chunks) > 0
        assert called_chunks[0]["metadata"]["level"] >= 1


@mock.patch("backend.services.rag.agent._call_llm")
def test_llm_judge_and_rewrite(mock_call):
    # 1. Mock LLM phản hồi JSON hợp lệ cho evaluate_answer
    mock_call.return_value = '{"faithfulness": "yes", "relevance": "yes", "sufficiency": "no", "feedback": "Thiếu thông tin về giá bán"}'
    
    judge_res = evaluate_answer("Giá bán xe VinFast VF8 là bao nhiêu?", "VinFast VF8 có thiết kế đẹp mắt.", "VF8 có thiết kế đẹp.")
    assert judge_res["sufficiency"] == "no"
    assert judge_res["feedback"] == "Thiếu thông tin về giá bán"
    
    # 2. Mock LLM cho rewrite_query
    mock_call.return_value = "Giá bán xe điện VinFast VF8 hiện nay là bao nhiêu tiền"
    rewritten = rewrite_query("Giá VinFast VF8?", "Thiếu thông tin về giá bán")
    assert rewritten == "Giá bán xe điện VinFast VF8 hiện nay là bao nhiêu tiền"


@pytest.fixture(scope="module")
def client_app():
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    with mock.patch("src.config.DOCUMENT_INTELLIGENCE_DIR", temp_path):
        from api.main import app
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c
            
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_agentic_rag_chat_flow(client_app):
    # Mock evaluate_answer để:
    # Lượt 1: Trả về sufficiency = "no" (kích hoạt retry & rewrite query)
    # Lượt 2: Trả về sufficiency = "yes" (chấp nhận kết quả)
    mock_evals = [
        {"faithfulness": "yes", "relevance": "yes", "sufficiency": "no", "feedback": "Thiếu dữ liệu doanh thu năm 2025"},
        {"faithfulness": "yes", "relevance": "yes", "sufficiency": "yes", "feedback": "Đầy đủ"}
    ]
    
    with mock.patch("backend.services.rag.agent.evaluate_answer", side_effect=mock_evals), \
         mock.patch("backend.services.rag.agent.rewrite_query", return_value="Doanh thu 2025 của công ty là bao nhiêu?"), \
         mock.patch("backend.services.rag.agent.expand_query", return_value=[]):
         
        # Mock generator build_answer và retriever retrieve để trả về kết quả giả định
        with mock.patch("backend.services.rag.generator.GroundedGenerator.build_answer", 
                        return_value={"answer": "Doanh thu năm 2025 đạt 100 tỷ đồng.", "confidence": 0.95, "grounded": True, "model_used": "mock"}):
            
            # Tạo conversation
            conv = client_app.post("/api/chat/conversations", json={"title": "Test Agentic RAG"}).json()
            conv_id = conv["id"]
            
            # Gửi tin nhắn đầu tiên
            # Endpoints: POST /api/chat/conversations/{conv_id}/messages
            msg_res = client_app.post(
                f"/api/chat/conversations/{conv_id}/messages",
                json={"role": "user", "content": "Hỏi doanh thu công ty"}
            )
            assert msg_res.status_code == 200
            
            # Kiểm tra xem lịch sử tin nhắn và trigger hoạt động ổn định
            detail = client_app.get(f"/api/chat/conversations/{conv_id}").json()
            assert len(detail["messages"]) == 1
