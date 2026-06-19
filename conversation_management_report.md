# BÁO CÁO TRIỂN KHAI HỆ THỐNG QUẢN LÝ CUỘC TRÒ CHUYỆN CHATBOT

Báo cáo chi tiết về thiết kế kiến trúc, mô hình dữ liệu, đặc tả API và giao diện của tính năng quản lý hội thoại thông minh (ChatGPT-like) được tích hợp trong hệ thống Hỏi đáp tài liệu ChatRAG.

---

## 1. Thiết Kế Cơ Sở Dữ Liệu (Database Schema)

Hệ thống đã nâng cấp cấu trúc SQLite từ bảng phẳng cũ sang hai bảng chuẩn hóa có quan hệ chặt chẽ:

### Bảng `conversations`
Lưu trữ thông tin tổng quan của từng phiên trò chuyện:
*   `id` (TEXT PRIMARY KEY): UUID định danh duy nhất.
*   `title` (TEXT NOT NULL): Tiêu đề cuộc trò chuyện (mặc định ban đầu là "New chat", sau đó tự động đặt tiêu đề).
*   `created_at` (TEXT NOT NULL): Thời gian tạo dạng ISO.
*   `updated_at` (TEXT NOT NULL): Thời gian cập nhật tin nhắn cuối cùng hoặc đổi tên (dùng để sắp xếp Sidebar và đánh index tối ưu).
*   `message_count` (INTEGER DEFAULT 0): Số lượng tin nhắn trong cuộc hội thoại để tối ưu hiển thị.
*   `is_archived` (INTEGER DEFAULT 0): Trạng thái lưu trữ (0/1).
*   `user_id` (TEXT): Định danh người dùng nếu có phân quyền.

### Bảng `messages`
Lưu trữ chi tiết toàn bộ các tin nhắn trong cuộc trò chuyện:
*   `id` (TEXT PRIMARY KEY): UUID định danh duy nhất.
*   `conversation_id` (TEXT NOT NULL): Khóa ngoại liên kết tới bảng `conversations`.
*   `role` (TEXT NOT NULL): Vai trò gửi tin (`user` hoặc `assistant`).
*   `content` (TEXT NOT NULL): Nội dung văn bản tin nhắn.
*   `confidence` (REAL): Độ tin cậy của mô hình RAG đối với câu trả lời (chỉ dành cho assistant).
*   `retrieval_threshold` (REAL): Ngưỡng tương đồng RAG sử dụng.
*   `citations_json` (TEXT NOT NULL): Chuỗi JSON chứa danh sách các đoạn trích dẫn nguồn tham chiếu chi tiết (filename, chunk_id, page, score).
*   `created_at` (TEXT NOT NULL): Thời gian tạo tin nhắn.
*   `model_used` (TEXT): Tên mô hình AI được sử dụng để sinh câu trả lời.
*   `evaluation_json` (TEXT): Kết quả đánh giá tự động (hallucination risk, consistency, coverage).
*   `metadata_json` (TEXT): Các cấu hình hoặc thông tin bổ sung khác.

### Ràng buộc & Chỉ mục (Constraints & Indexes)
1.  **CASCADE DELETE**: Sử dụng ràng buộc khóa ngoại `FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE`. Để đảm bảo tính năng này hoạt động trên SQLite, hệ thống tự động chạy `PRAGMA foreign_keys = ON;` trên mọi kết nối cơ sở dữ liệu. Khi xóa cuộc trò chuyện, toàn bộ tin nhắn liên quan sẽ tự động bị xóa triệt để khỏi đĩa cứng.
2.  **Indexes**:
    *   Tạo index trên `messages(conversation_id)` để tăng tốc độ truy vấn chi tiết lịch sử tin nhắn.
    *   Tạo index trên `conversations(updated_at)` để sắp xếp nhanh danh sách lịch sử trò chuyện mới nhất lên trên.

---

## 2. Thiết Kế API Backend

Hệ thống cung cấp bộ RESTful API hoàn chỉnh dưới prefix `/api/chat` tại file `api/chat.py`:

1.  **Lấy danh sách cuộc trò chuyện**: `GET /api/chat/conversations?limit=50&offset=0`
2.  **Lấy chi tiết cuộc trò chuyện kèm tin nhắn**: `GET /api/chat/conversations/{id}`
3.  **Tạo cuộc trò chuyện mới**: `POST /api/chat/conversations` (Chấp nhận body dạng `{ "title": "...", "user_id": "..." }`)
4.  **Đổi tên cuộc trò chuyện**: `PUT /api/chat/conversations/{id}` (Body `{ "title": "..." }`)
5.  **Xóa cuộc trò chuyện**: `DELETE /api/chat/conversations/{id}`
6.  **Lưu tin nhắn mới**: `POST /api/chat/conversations/{id}/messages` (Body chứa chi tiết tin nhắn, citations, confidence, v.v. và kích hoạt logic auto-title).
7.  **Tìm kiếm hội thoại**: `GET /api/chat/search?q={query}` (Tìm kiếm theo tiêu đề cuộc trò chuyện hoặc nội dung tin nhắn bất kỳ chứa từ khóa).

---

## 3. Cơ Chế Tự Động Sinh Tiêu Đề (Auto Title Generation)

Để tối ưu hóa trải nghiệm người dùng giống như ChatGPT, hệ thống tự động đặt tiêu đề thông minh:
*   **Trigger**: Khi một cuộc trò chuyện mới tích lũy đủ từ 2 đến 4 tin nhắn (sau lượt hỏi đáp đầu tiên), hệ thống kích hoạt chạy ngầm logic auto-title.
*   **Thuật toán sinh**: Sử dụng LLM đang cấu hình (Google Gemini, OpenAI GPT, Ollama) hoặc mô hình sinh văn bản cục bộ để tóm tắt câu hỏi của người dùng thành tiêu đề tiếng Việt ngắn gọn.
*   **Ràng buộc tiêu đề**: Tiêu đề có độ dài lý tưởng từ 5 đến 10 từ (giới hạn 3 đến 15 từ, không chứa ký tự đặc biệt, không dài quá 80 ký tự).
*   **Cơ chế Fallback**: Nếu AI sinh tiêu đề bị lỗi hoặc không đáp ứng ràng buộc độ dài, hệ thống sẽ tự động lấy 50 ký tự đầu tiên của tin nhắn hỏi đầu tiên của user làm tiêu đề cuộc trò chuyện.

---

## 4. Tích Hợp Frontend & Trải Nghiệm Người Dùng (UX/UI)

Giao diện Sidebar và Quản lý trạng thái đã được tái cấu trúc hoàn toàn tại `frontend/src/pages/Chat.tsx`:

1.  **Sidebar Tab Switcher**: Cho phép chuyển đổi nhanh chóng giữa tab "Hội thoại" (quản lý lịch sử chat) và tab "Tài liệu" (tải lên tài liệu và chọn phạm vi RAG).
2.  **Phân nhóm thời gian**: Lịch sử hội thoại được phân tích và nhóm thành các mục trực quan:
    *   *Hôm nay*
    *   *Hôm qua*
    *   *7 ngày gần nhất*
    *   *30 ngày gần nhất*
    *   *Cũ hơn*
3.  **Quản lý trạng thái nâng cao với React Query**:
    *   Sử dụng `useQuery` để fetch danh sách và tìm kiếm cuộc trò chuyện.
    *   Sử dụng `useMutation` cho các tác vụ tạo mới, đổi tên và xóa cuộc trò chuyện.
    *   Áp dụng **Optimistic Updates**: Khi đổi tên hoặc xóa cuộc trò chuyện, giao diện UI lập tức phản hồi và cập nhật danh sách Sidebar trước khi API trả về kết quả thành công, mang lại cảm giác mượt mà tức thì cho người dùng.
    *   **Cache Invalidation**: Tự động vô hiệu hóa cache và refresh danh sách cuộc trò chuyện sau khi thêm tin nhắn mới, đổi tên hoặc xóa.
4.  **Inline Rename & Confirm Delete Modal**:
    *   Nhấp đúp hoặc bấm nút Rename (hover) sẽ hiển thị ô nhập liệu trực tiếp ngay trên Sidebar, nhấn Enter để lưu hoặc Escape để hủy bỏ.
    *   Bấm nút Delete sẽ kích hoạt một Modal overlay tuyệt đẹp yêu cầu xác nhận xóa cuộc trò chuyện để đảm bảo người dùng không vô tình xóa mất dữ liệu.
5.  **Tìm kiếm thời gian thực**: Thanh tìm kiếm nằm ở Sidebar cho phép lọc danh sách hội thoại theo từ khóa tức thì khi đang gõ chữ.
