# Changelog

Tất cả các thay đổi quan trọng trong dự án liên quan đến hệ thống quản lý lịch sử cuộc trò chuyện Chatbot (ChatGPT-like) sẽ được ghi nhận tại đây.

## [1.0.0] - 2026-06-19

### Added
- **Database Layer**:
  - Bảng `conversations` mới trong SQLite với các trường: `id` (UUID), `title`, `created_at`, `updated_at`, `message_count`, `is_archived`, `user_id`.
  - Bảng `messages` mới với các trường: `id` (UUID), `conversation_id`, `role`, `content`, `created_at`, `metadata_json` (và các cột đặc trưng RAG: `citations_json`, `confidence`, `retrieval_threshold`, `model_used`, `evaluation`).
  - Hỗ trợ ràng buộc `ON DELETE CASCADE` tự động xóa toàn bộ tin nhắn khi xóa cuộc trò chuyện tương ứng.
  - Các index tối ưu trên `messages(conversation_id)` và `conversations(updated_at)`.
  - Cơ chế tự động di chuyển dữ liệu (Auto Migration) từ các bảng cũ `rag_chat_conversations`/`rag_chat_messages` sang cấu trúc mới khi khởi động ứng dụng.
- **Backend API (`api/chat.py`)**:
  - `GET /api/chat/conversations`: Lấy danh sách các cuộc trò chuyện (hỗ trợ phân trang).
  - `GET /api/chat/conversations/{id}`: Lấy chi tiết cuộc trò chuyện kèm theo lịch sử tin nhắn.
  - `POST /api/chat/conversations`: Tạo cuộc trò chuyện mới.
  - `PUT /api/chat/conversations/{id}`: Đổi tên cuộc trò chuyện.
  - `DELETE /api/chat/conversations/{id}`: Xóa cuộc trò chuyện.
  - `POST /api/chat/conversations/{id}/messages`: Lưu tin nhắn mới và kích hoạt auto-title.
  - `GET /api/chat/search?q=`: Tìm kiếm cuộc trò chuyện dựa trên tiêu đề hoặc nội dung tin nhắn.
- **Auto Title Generation**:
  - Logic tự động đặt tiêu đề thông minh cho cuộc hội thoại mới khi đạt từ 2 đến 4 tin nhắn đầu tiên sử dụng LLM API hoặc mô hình Transformer cục bộ.
  - Cơ chế Fallback an toàn tự động lấy 50 ký tự đầu tiên của câu hỏi đầu tiên làm tiêu đề nếu AI gặp lỗi.
- **Frontend UI & State Management**:
  - Tích hợp **TanStack React Query** để quản lý trạng thái, hỗ trợ Cache Invalidation, Auto Refresh, và Optimistic Updates khi người dùng đổi tên/xóa cuộc trò chuyện.
  - Sidebar Left Panel được thiết kế tab-switcher gồm hai tab: "Hội thoại" (Lịch sử chat) và "Tài liệu" (Nạp file).
  - Phân nhóm cuộc trò chuyện theo thời gian trên Sidebar: *Hôm nay, Hôm qua, 7 ngày gần nhất, 30 ngày gần nhất, Cũ hơn*.
  - Chức năng đổi tên trực tiếp (inline rename) và nút xóa nhanh trên Sidebar khi hover.
  - Modal xác nhận xóa cuộc trò chuyện (Delete Confirmation Modal) tránh thao tác nhầm.
- **Tests**:
  - File `tests/test_chat_history.py` kiểm thử tích hợp đầy đủ các API CRUD hội thoại, lưu tin nhắn, tìm kiếm và kích hoạt tự động sinh tiêu đề.

### Fixed
- Lỗi thiếu import `re` trong `backend/services/rag/service.py` khi xử lý chuẩn hóa tiêu đề hội thoại sinh ra từ AI.
- Loại bỏ dropdown selector cũ trên Chat Header để hiển thị tiêu đề cuộc trò chuyện đang hoạt động một cách tinh giản.

## [1.1.0] - 2026-06-19

### Added
- **Semantic Chunking**:
  - Hỗ trợ 3 giải thuật ngắt ngữ nghĩa thông minh: Fixed Threshold, Dynamic Threshold, và Sliding Window Semantic Chunking tính theo cosine similarity.
- **Agentic RAG**:
  - Tác nhân tự phản hồi và tự sửa lỗi (Self-Correction loop) thông qua LLM Judge đánh giá 3 tiêu chí: Faithfulness, Relevance, Sufficiency.
  - Tích hợp Query Rewriting để tự động viết lại câu hỏi tập trung vào thông tin bị thiếu và thử lại (Retrieval Retry) tối đa 3 lần.
- **RAPTOR với GMM**:
  - Phân cụm mềm (Soft Clustering) bằng thuật toán Gaussian Mixture Model (GMM) tự viết bằng Numpy.
  - Xây dựng chỉ mục cây tri thức đệ quy nhiều tầng (Recursive Tree Construction) từ Level 0 đến Level 3.
  - Hỗ trợ Multi-Level Retrieval truy xuất kết hợp cả base chunks và summary nodes cấp cao.
- **Tests**:
  - Thêm file [test_rag_advanced_nextgen.py](file:///c:/Users/ASUS/Desktop/NLP-Text-Summarization-Transformer-System/tests/test_rag_advanced_nextgen.py) kiểm thử thành công 100% tất cả các tính năng RAG nâng cao.
