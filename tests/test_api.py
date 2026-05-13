"""
tests/test_api.py — Integration tests cho FastAPI endpoints.
Dùng TestClient của FastAPI để test không cần server thật.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

# Import app nhưng skip model load để test nhanh
import unittest.mock as mock

@pytest.fixture(scope="module")
def client():
    """Tạo TestClient với model bị mock để test nhanh."""
    with mock.patch("src.abstractive.get_summarizer") as mock_sum:
        # Mock summarizer luôn trả về kết quả
        instance = mock.MagicMock()
        instance.is_loaded.return_value = True
        instance.summarize.return_value = "Bản tóm tắt diễn giải mẫu."
        mock_sum.return_value = instance

        from api.main import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_root_ok(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "name" in r.json()

    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"

    def test_docs_accessible(self, client):
        r = client.get("/docs")
        assert r.status_code == 200


class TestSummarizeEndpoint:
    SAMPLE = (
        "Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp để thảo luận về tình hình "
        "leo thang căng thẳng ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay "
        "lập tức và mở hành lang nhân đạo cho người dân vùng chiến sự. Đại diện Mỹ "
        "phát biểu rằng Washington ủng hộ giải pháp hai nhà nước. Cuộc khủng hoảng "
        "nhân đạo ngày càng nghiêm trọng khi hàng nghìn thường dân phải di tản."
    )

    def test_summarize_text(self, client):
        r = client.post("/summarize", json={"text": self.SAMPLE, "save_result": False})
        assert r.status_code == 200
        data = r.json()
        assert "extractive" in data
        assert "abstractive" in data
        assert "best" in data
        assert "scores" in data

    def test_missing_input_422(self, client):
        r = client.post("/summarize", json={})
        assert r.status_code == 422

    def test_response_has_word_count(self, client):
        r = client.post("/summarize", json={"text": self.SAMPLE, "save_result": False})
        assert r.status_code == 200
        assert "word_count" in r.json()


class TestModelsEndpoint:
    def test_models_list(self, client):
        r = client.get("/models")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert isinstance(data["models"], list)
