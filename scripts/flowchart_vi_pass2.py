#!/usr/bin/env python3
"""Second pass: Vietnamese label fixes without renumbering."""

import re
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent / "docs" / "flowcharts"

# (old, new) — applied to value="..." content only
VALUE_REPLACEMENTS = [
    # 00 sequence participants (&#10; is newline in draw.io)
    ("FastAPI&#10;api/main.py", "API FastAPI&#10;(main.py)"),
    ("dashboard_service&#10;stream_compare / summarize_all", "Dịch vụ dashboard&#10;stream_compare / summarize_all"),
    ("Models&#10;TextRank / ViT5 / mT5 / BARTPho", "Mô hình&#10;TextRank / ViT5 / mT5 / BARTPho"),
    ("Storage&#10;storage/results", "Lưu trữ&#10;storage/results"),
    ("Celery&#10;(async_mode)", "Celery&#10;(chế độ bất đồng bộ)"),
    ("Chat.tsx&#10;ragApi.ts", "Giao diện Chat&#10;(ragApi.ts)"),
    ("FastAPI&#10;document_chat.py", "API FastAPI&#10;(document_chat.py)"),
    ("RAGChatService&#10;service.py", "Dịch vụ RAG Chat&#10;(service.py)"),
    ("RAGRepository&#10;rag_chat.db", "Kho hội thoại&#10;(rag_chat.db)"),
    ("VectorStore&#10;Chroma", "Vector store&#10;(Chroma)"),
    ("HybridRetriever&#10;BM25+RRF+Rerank", "Bộ truy xuất lai&#10;BM25 + RRF + Rerank"),
    ("GroundedGenerator&#10;generator.py", "Bộ sinh có căn cứ&#10;(generator.py)"),
    ("LLM / Model&#10;Gemini·Ollama·ViT5", "Mô hình LLM&#10;Gemini / Ollama / ViT5"),
    # Sequence messages
    ("2. handleRun() → streamCompareSummaries()", "2. handleRun() → gọi streamCompareSummaries()"),
    ("5. StreamingResponse(stream_compare(...))", "5. Phản hồi streaming (stream_compare)"),
    ("14. Cập nhật AlgorithmCard + ComparisonTable", "14. Cập nhật thẻ thuật toán và bảng so sánh"),
    ("5. StreamingResponse → service.stream_chat()", "5. Phản hồi streaming → service.stream_chat()"),
    ("2. setMessages(user + assistant placeholder)", "2. setMessages(tin nhắn người dùng + placeholder trợ lý)"),
    ("10. summary + ROUGE/BERTScore metrics", "10. tóm tắt + chỉ số ROUGE/BERTScore"),
    ("13. SSE «finished» (data + storage.result_id)", "13. SSE «finished» (dữ liệu + storage.result_id)"),
    # Titles
    ("Luồng chính: RAG Chat — Chat.tsx → POST /rag/chat/stream (SSE) | alt sync POST /rag/chat",
     "Luồng chính: RAG Chat — Giao diện Chat → POST /rag/chat/stream (SSE) | nhánh đồng bộ POST /rag/chat"),
    ("Luồng chính: Tóm tắt văn bản — Playground (SSE compare) + POST /summarize (alt)",
     "Luồng chính: Tóm tắt văn bản — Playground (so sánh SSE) + POST /summarize (nhánh thay thế)"),
    # Flowchart common
    ("Client:", "Người dùng:"),
    ("HTTP 200 response", "Phản hồi HTTP 200"),
    ("HTTP 200 JSON", "Phản hồi HTTP 200 (JSON)"),
    ("HTTP 200 dashboard JSON", "Phản hồi HTTP 200 (JSON dashboard)"),
    ("HTTP 200 diagnostics JSON", "Phản hồi HTTP 200 (JSON chẩn đoán)"),
    ("FastAPI lifespan startup", "Khởi động lifespan FastAPI"),
    ("Server lắng nghe / tắt gracefully", "Máy chủ lắng nghe / tắt an toàn"),
    ("Kết thúc SSE stream", "Kết thúc luồng SSE"),
    ("Trả compare payload hoặc job status", "Trả payload so sánh hoặc trạng thái job"),
    ("Celery worker:", "Worker Celery:"),
    ("Sync:", "Đồng bộ:"),
    ("StreamingResponse", "Phản hồi streaming"),
    ("ThreadPoolExecutor", "ThreadPoolExecutor (luồng song song)"),
    ("Endpoint nào?", "Điểm cuối (endpoint) nào?"),
    ("Xây danh sách algorithms", "Xây danh sách thuật toán"),
    ("append_message + auto title", "append_message + tự đặt tiêu đề"),
    ("delete conversation(s)", "xóa hội thoại"),
    ("create_conversation", "tạo hội thoại"),
    ("list_conversations", "liệt kê hội thoại"),
    ("search_conversations", "tìm kiếm hội thoại"),
    ("get_conversation + list_messages", "lấy hội thoại + liệt kê tin nhắn"),
]

# Regex replacements preserving step numbers: N. text
REGEX_REPLACEMENTS = [
    (r"(\d+\. )HTTP 200 ", r"\1Phản hồi HTTP 200 "),
    (r"(\d+\. )GET /", r"\1GET /"),  # keep API paths
]


def main():
    for path in sorted(DIR.glob("*.drawio")):
        text = path.read_text(encoding="utf-8")
        original = text
        for old, new in VALUE_REPLACEMENTS:
            text = text.replace(f'value="{old}"', f'value="{new}"')
            # also inside longer values
            text = text.replace(old, new)
        for pat, repl in REGEX_REPLACEMENTS:
            text = re.sub(pat, repl, text)
        # dedupe duplicate style fragments from pass 1
        text = re.sub(r"(endArrow=none;dashed=1;html=1;){2,}", r"\1", text)
        text = re.sub(r"(opacity=40;[^;]*;opacity=60)", "opacity=60", text)
        if text != original:
            path.write_text(text, encoding="utf-8")
            print(f"patched: {path.name}")
    print("done")


if __name__ == "__main__":
    main()
