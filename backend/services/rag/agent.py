"""
agent.py — Agentic RAG Router & Query Expansion.

Phân loại ý định người dùng (Intent Routing) và Mở rộng truy vấn (Query Expansion)
hỗ trợ cho hệ thống RAG để tối ưu kết quả tìm kiếm đa tài liệu.
"""
from __future__ import annotations

import logging
import re
from typing import Any
import requests

from .rag_config import (
    RAG_GENERATOR_TYPE,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    GEMINI_MODEL,
    OPENAI_MODEL,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)


def _call_llm(prompt: str) -> str:
    """Gọi LLM API theo cấu hình hiện tại để lấy câu trả lời dạng văn bản thô."""
    generator_type = RAG_GENERATOR_TYPE.lower()
    try:
        if generator_type == "gemini":
            if not GEMINI_API_KEY:
                return ""
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 150}
            }
            res = requests.post(url, json=payload, headers=headers, timeout=10)
            res.raise_for_status()
            return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

        elif generator_type == "openai":
            if not OPENAI_API_KEY:
                return ""
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            payload = {
                "model": OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 150
            }
            res = requests.post(url, json=payload, headers=headers, timeout=10)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"].strip()

        elif generator_type == "ollama":
            url = OLLAMA_API_URL
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            }
            res = requests.post(url, json=payload, headers=headers, timeout=15)
            res.raise_for_status()
            return res.json()["response"].strip()

    except Exception as exc:
        logger.warning("⚠️ Agent LLM call failed, fallback to rule-based logic: %s", exc)
    return ""


def classify_intent(query: str, document_ids: list[str] | None = None) -> str:
    """
    Phân loại ý định của người dùng:
    - SUMMARIZE: Yêu cầu tóm tắt (cần có document_ids).
    - GENERAL: Trò chuyện xã giao, không liên quan tài liệu.
    - DOCUMENT_QA: Hỏi đáp, tra cứu thông tin dựa trên tài liệu.
    """
    query_clean = query.strip()
    if not query_clean:
        return "GENERAL"

    # 1. Rule-based checks trước tiên
    # Kiểm tra ý định tóm tắt
    summarize_keywords = ["tóm tắt", "summarize", "tóm lược", "khái quát", "bản tóm tắt"]
    is_asking_summary = any(kw in query_clean.lower() for kw in summarize_keywords)
    if is_asking_summary and document_ids:
        # Nếu hỏi tóm tắt và có chọn tài liệu -> Chuyển sang tóm tắt
        return "SUMMARIZE"

    # Kiểm tra ý định xã giao / trò chuyện thông thường
    general_patterns = [
        r"^(chào|hello|hi|xin chào|chào bạn|chào trợ lý|chào bot|greetings)\b",
        r"^(bạn là ai|tên bạn là gì|who are you|what is your name)\b",
        r"^(cảm ơn|cám ơn|thank|thanks|thank you)\b",
        r"^(tạm biệt|bye|goodbye)\b"
    ]
    is_general = any(re.search(pat, query_clean.lower()) for pat in general_patterns)
    if is_general and not (document_ids and len(document_ids) > 0):
        return "GENERAL"

    # 2. Gọi LLM để phân loại (nếu cấu hình API key đầy đủ)
    if RAG_GENERATOR_TYPE in {"gemini", "openai", "ollama"}:
        prompt = f"""Bạn là một bộ định tuyến ý định câu hỏi (Intent Router).
Nhiệm vụ của bạn là phân loại câu hỏi của người dùng vào 1 trong 3 nhóm duy nhất:
1. "GENERAL": Chào hỏi, cảm ơn, tạm biệt hoặc trò chuyện xã giao không liên quan đến dữ liệu tài liệu cụ thể.
2. "SUMMARIZE": Yêu cầu tóm tắt toàn bộ tài liệu hoặc các chương phần lớn.
3. "DOCUMENT_QA": Câu hỏi tra cứu thông tin chi tiết, phân tích, đối chiếu hoặc so sánh nội dung tài liệu.

Chỉ trả về đúng một từ duy nhất: GENERAL, SUMMARIZE hoặc DOCUMENT_QA. Không giải thích thêm.

Câu hỏi: "{query_clean}"
Phân loại:"""
        llm_response = _call_llm(prompt)
        # Chuẩn hóa kết quả trả về từ LLM
        for val in ["GENERAL", "SUMMARIZE", "DOCUMENT_QA"]:
            if val in llm_response.upper():
                # Nếu là SUMMARIZE nhưng người dùng không chọn tài liệu nào, chuyển về GENERAL hoặc DOCUMENT_QA
                if val == "SUMMARIZE" and not document_ids:
                    return "DOCUMENT_QA"
                return val

    # 3. Fallback mặc định: Nếu không phải trò chuyện xã giao, mặc định là hỏi đáp tài liệu
    return "DOCUMENT_QA"


def expand_query(query: str) -> list[str]:
    """
    Thực hiện Query Expansion (Mở rộng câu hỏi) để tạo ra thêm 1-2 câu hỏi phụ,
    giúp tăng tỷ lệ Recall khi tìm kiếm trên nhiều tài liệu.
    """
    query_clean = query.strip()
    if not query_clean:
        return []

    # 1. Gọi LLM để sinh các câu hỏi tương đương
    if RAG_GENERATOR_TYPE in {"gemini", "openai", "ollama"}:
        prompt = f"""Bạn là chuyên gia viết lại câu hỏi để tối ưu hóa tìm kiếm (Query Expansion).
Dựa trên câu hỏi hiện tại, hãy viết lại thành tối đa 2 câu hỏi tương đương bằng tiếng Việt để tìm từ khóa tốt hơn.
Yêu cầu:
- Mỗi câu hỏi viết trên một dòng.
- Không đánh số thứ tự, không thêm lời dẫn giải thích.
- Giữ nguyên ý nghĩa cốt lõi của câu hỏi ban đầu.

Câu hỏi gốc: "{query_clean}"
Các câu hỏi mở rộng:"""
        
        llm_response = _call_llm(prompt)
        if llm_response:
            expanded = []
            for line in llm_response.split("\n"):
                line_clean = re.sub(r"^\d+[\.\-\s]+", "", line.strip()) # Xóa đánh số nếu có
                line_clean = line_clean.strip('"-* ')
                if line_clean and line_clean.lower() != query_clean.lower():
                    expanded.append(line_clean)
            if expanded:
                return expanded[:2]

    # 2. Heuristic fallback: Tách cụm từ khóa hoặc trả về rỗng nếu không gọi được LLM
    return []
