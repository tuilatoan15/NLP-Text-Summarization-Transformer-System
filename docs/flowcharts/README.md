# Flowcharts — NLP Text Summarization Transformer System

Sơ đồ luồng xử lý (draw.io) cho các API và workflow chính. Mở tại [app.diagrams.net](https://app.diagrams.net) → File → Import.

**Quy ước:** Tất cả sơ đồ dùng nhãn tiếng Việt, đánh số bước tuần tự (`1.`, `2.`, `3.` …) và phong cách đen trắng (nền trắng, viền/chữ đen).

| File | Mô tả |
|------|-------|
| [00-full-project-sequence.drawio](./00-full-project-sequence.drawio) | **Sơ đồ tuần tự tổng thể (UML)** — một luồng master (39 bước): khởi động, Playground so sánh tóm tắt, RAG upload/chat, Document Intelligence, Dataset Analytics, Dashboard/Benchmark. [SVG](./00-full-project-sequence.svg) |
| [00-main-sequence-diagram.drawio](./00-main-sequence-diagram.drawio) | **Sơ đồ sequence chính (UML)** — 2 tab: (A) Playground + POST /summarize/compare/stream + alt /summarize; (B) Chat.tsx + POST /rag/chat/stream + alt sync. |
| [01-text-summarize.drawio](./01-text-summarize.drawio) | Luồng POST /summarize — tóm tắt đơn (sync hoặc Celery async). |
| [02-compare-summarize-stream.drawio](./02-compare-summarize-stream.drawio) | Luồng POST /summarize/compare/stream — so sánh thuật toán qua SSE. |
| [03-compare-summarize-sync.drawio](./03-compare-summarize-sync.drawio) | Luồng POST /summarize/compare — so sánh sync hoặc Celery job. |
| [04-file-upload-summarize.drawio](./04-file-upload-summarize.drawio) | Luồng POST /summarize/files* — upload file, trích xuất và tóm tắt. |
| [05-summarize-history.drawio](./05-summarize-history.drawio) | Luồng GET/DELETE /summarize/history — lịch sử kết quả so sánh. |
| [06-rag-document-upload.drawio](./06-rag-document-upload.drawio) | Luồng POST /rag/documents/upload — ingest tài liệu RAG. |
| [07-rag-chat-stream.drawio](./07-rag-chat-stream.drawio) | Luồng POST /rag/chat/stream — hỏi đáp RAG streaming SSE. |
| [08-rag-chat-sync.drawio](./08-rag-chat-sync.drawio) | Luồng POST /rag/chat — hỏi đáp RAG đồng bộ. |
| [09-rag-conversation-crud.drawio](./09-rag-conversation-crud.drawio) | Luồng CRUD /api/chat/conversations/* — quản lý hội thoại. |
| [10-document-intelligence-ingest.drawio](./10-document-intelligence-ingest.drawio) | Luồng POST /documents/ingest — nạp tài liệu Document Intelligence. |
| [11-document-semantic-search.drawio](./11-document-semantic-search.drawio) | Luồng POST /documents/{id}/search — tìm kiếm ngữ nghĩa. |
| [12-document-compare-summaries.drawio](./12-document-compare-summaries.drawio) | Luồng POST /documents/{id}/compare — so sánh tóm tắt trên tài liệu. |
| [13-document-explainability.drawio](./13-document-explainability.drawio) | Luồng GET /documents/{id}/explainability — giải thích extractive. |
| [14-document-hierarchical-summarize.drawio](./14-document-hierarchical-summarize.drawio) | Luồng POST /documents/{id}/summarize/hierarchical — tóm tắt phân cấp. |
| [15-document-report-export.drawio](./15-document-report-export.drawio) | Luồng GET /documents/{id}/report/* — xuất báo cáo HTML/Markdown. |
| [16-analytics-dashboard.drawio](./16-analytics-dashboard.drawio) | Luồng GET /analytics/dashboard — dashboard metrics từ kết quả đã lưu. |
| [17-dataset-analytics.drawio](./17-dataset-analytics.drawio) | Luồng GET/POST /analytics/dataset/* — phân tích bộ VietNews. |
| [18-research-benchmark.drawio](./18-research-benchmark.drawio) | Luồng /research/* — leaderboard, samples, benchmark run. |
| [19-overview-dashboard.drawio](./19-overview-dashboard.drawio) | Luồng Overview page — gộp health, metrics, dashboard, system APIs. |
| [20-system-config-gpu.drawio](./20-system-config-gpu.drawio) | Luồng /config và /system/* — cấu hình, GPU, node, models. |
| [21-dashboard-search.drawio](./21-dashboard-search.drawio) | Luồng GET /search — tìm kiếm toàn dashboard. |
| [22-api-startup.drawio](./22-api-startup.drawio) | Luồng lifespan api/main.py — preload models, warm cache, shutdown. |
