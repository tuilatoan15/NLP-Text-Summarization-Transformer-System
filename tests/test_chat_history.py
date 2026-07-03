from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
import shutil
from unittest import mock
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Tạo TestClient với thư mục RAG database tạm thời để tránh ảnh hưởng đến dữ liệu thực."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Mock config.DOCUMENT_INTELLIGENCE_DIR để database được ghi vào thư mục tạm
    with mock.patch("src.config.DOCUMENT_INTELLIGENCE_DIR", temp_path):
        from api.main import app
        with TestClient(app) as c:
            yield c
            
    # Xóa thư mục tạm sau khi kết thúc test
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_create_and_list_conversations(client):
    # 1. Tạo cuộc trò chuyện mới với tiêu đề mặc định
    response = client.post("/api/chat/conversations", json={"title": "Học máy cơ bản"})
    assert response.status_code == 200
    conv = response.json()
    assert "id" in conv
    assert conv["title"] == "Học máy cơ bản"
    assert conv["message_count"] == 0
    
    conv_id = conv["id"]
    
    # 2. Lấy danh sách các cuộc trò chuyện
    list_response = client.get("/api/chat/conversations")
    assert list_response.status_code == 200
    items = list_response.json()["items"]
    assert len(items) >= 1
    
    # Tìm kiếm cuộc trò chuyện vừa tạo trong danh sách
    found = next((item for item in items if item["id"] == conv_id), None)
    assert found is not None
    assert found["title"] == "Học máy cơ bản"


def test_get_conversation_detail_and_not_found(client):
    # Tạo cuộc trò chuyện
    response = client.post("/api/chat/conversations", json={"title": "Test Detail"})
    conv_id = response.json()["id"]
    
    # Lấy chi tiết cuộc trò chuyện
    detail_response = client.get(f"/api/chat/conversations/{conv_id}")
    assert detail_response.status_code == 200
    conv_detail = detail_response.json()
    assert conv_detail["id"] == conv_id
    assert conv_detail["title"] == "Test Detail"
    assert "messages" in conv_detail
    assert isinstance(conv_detail["messages"], list)
    assert len(conv_detail["messages"]) == 0

    # Lấy cuộc trò chuyện không tồn tại (UUID ngẫu nhiên)
    fake_id = "00000000-0000-0000-0000-000000000000"
    not_found_response = client.get(f"/api/chat/conversations/{fake_id}")
    assert not_found_response.status_code == 404


def test_rename_conversation(client):
    # Tạo cuộc trò chuyện
    response = client.post("/api/chat/conversations", json={"title": "Tiêu đề cũ"})
    conv_id = response.json()["id"]
    
    # Đổi tên
    rename_response = client.put(
        f"/api/chat/conversations/{conv_id}", 
        json={"title": "Tiêu đề mới hoàn toàn"}
    )
    assert rename_response.status_code == 200
    assert rename_response.json()["ok"] is True
    assert rename_response.json()["title"] == "Tiêu đề mới hoàn toàn"
    
    # Lấy chi tiết xem đã đổi tên chưa
    detail_response = client.get(f"/api/chat/conversations/{conv_id}")
    assert detail_response.json()["title"] == "Tiêu đề mới hoàn toàn"


def test_delete_conversation_cascade(client):
    # Tạo cuộc trò chuyện
    response = client.post("/api/chat/conversations", json={"title": "Sẽ bị xóa"})
    conv_id = response.json()["id"]
    
    # Thêm tin nhắn vào cuộc trò chuyện này
    msg_response = client.post(
        f"/api/chat/conversations/{conv_id}/messages",
        json={"role": "user", "content": "Câu hỏi số một"}
    )
    assert msg_response.status_code == 200
    
    # Xóa cuộc trò chuyện
    delete_response = client.delete(f"/api/chat/conversations/{conv_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["ok"] is True
    
    # Lấy lại cuộc trò chuyện này sẽ thấy 404
    get_response = client.get(f"/api/chat/conversations/{conv_id}")
    assert get_response.status_code == 404


def test_save_message_and_auto_title_trigger(client):
    # Mock LLM sinh tiêu đề luôn trả về tiêu đề cố định để test auto-title hoạt động
    mock_title = "Đánh giá mô hình BARTPho"
    
    with mock.patch("backend.services.rag.service.generate_title_for_conversation", return_value=mock_title):
        # 1. Tạo cuộc trò chuyện
        response = client.post("/api/chat/conversations", json={"title": "New chat"})
        conv_id = response.json()["id"]
        
        # 2. Gửi tin nhắn 1 (user)
        msg1 = client.post(
            f"/api/chat/conversations/{conv_id}/messages",
            json={"role": "user", "content": "Fine-tune BARTPho như thế nào?"}
        )
        assert msg1.status_code == 200
        
        # Vì chỉ mới có 1 tin nhắn, tiêu đề chưa đổi
        detail = client.get(f"/api/chat/conversations/{conv_id}").json()
        assert detail["title"] == "New chat"
        assert len(detail["messages"]) == 1
        
        # 3. Gửi tin nhắn 2 (assistant) -> Tổng số tin nhắn = 2, kích hoạt auto title
        msg2 = client.post(
            f"/api/chat/conversations/{conv_id}/messages",
            json={"role": "assistant", "content": "Bạn cần chuẩn bị dataset VietNews và thư viện HuggingFace..."}
        )
        assert msg2.status_code == 200
        
        # Tiêu đề cuộc trò chuyện cần được tự động đặt lại thông qua mock_title
        # (auto-title chạy async trong background thread)
        import time
        updated_detail = client.get(f"/api/chat/conversations/{conv_id}").json()
        for _ in range(50):
            if updated_detail["title"] == mock_title:
                break
            time.sleep(0.05)
            updated_detail = client.get(f"/api/chat/conversations/{conv_id}").json()
        assert updated_detail["title"] == mock_title
        assert len(updated_detail["messages"]) == 2


def test_search_conversations(client):
    # Tạo cuộc trò chuyện 1 có chứa từ khóa đặc biệt "BARTPho"
    client.post("/api/chat/conversations", json={"title": "Đánh giá BARTPho VietNews"})
    
    # Tạo cuộc trò chuyện 2 không chứa từ khóa đó
    client.post("/api/chat/conversations", json={"title": "Học máy cơ bản tuần 1"})
    
    # Tạo cuộc trò chuyện 3 có tin nhắn chứa nội dung "BARTPho"
    conv3 = client.post("/api/chat/conversations", json={"title": "Hội thoại 3"}).json()
    conv3_id = conv3["id"]
    client.post(
        f"/api/chat/conversations/{conv3_id}/messages",
        json={"role": "user", "content": "Tìm tài liệu về BARTPho"}
    )
    
    # Thực hiện tìm kiếm từ khóa "BARTPho"
    search_response = client.get("/api/chat/search?q=BARTPho")
    assert search_response.status_code == 200
    results = search_response.json()["items"]
    
    # Kết quả tìm kiếm cần phải trả về ít nhất 2 cuộc trò chuyện (cuộc trò chuyện 1 và cuộc trò chuyện 3)
    assert len(results) >= 2
    titles = [item["title"] for item in results]
    assert "Đánh giá BARTPho VietNews" in titles
    assert "Hội thoại 3" in titles
