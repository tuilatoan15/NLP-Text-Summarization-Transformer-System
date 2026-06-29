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


import threading
import time as _time

# ─── Global Rate Limiter: tối đa 10 requests / 60 giây ───────────────────────
_rate_lock = threading.Lock()
_rate_timestamps: list[float] = []
_RATE_LIMIT_RPM = 10          # requests per minute (dưới ngưỡng 15 RPM free tier)
_RATE_WINDOW = 60.0           # seconds


def _wait_for_rate_limit():
    """Block cho đến khi có slot trong cửa sổ rate-limit."""
    while True:
        now = _time.monotonic()
        with _rate_lock:
            # Loại bỏ timestamps cũ hơn cửa sổ
            _rate_timestamps[:] = [t for t in _rate_timestamps if now - t < _RATE_WINDOW]
            if len(_rate_timestamps) < _RATE_LIMIT_RPM:
                _rate_timestamps.append(now)
                return
            # Tính thời gian chờ đến khi slot cũ nhất hết hạn
            wait = _RATE_WINDOW - (now - _rate_timestamps[0]) + 0.1
        logger.debug("⏳ Rate limiter: chờ %.1fs trước khi gọi API...", wait)
        _time.sleep(wait)


def _execute_llm_request(
    prompt: str,
    generator_type: str,
    temperature: float = 0.2,
    max_tokens: int = 800
) -> str:
    """Gọi LLM API với retry cho lỗi 429 và tự động fallback chéo giữa Gemini và OpenAI."""
    import time
    
    # Xác định danh sách các API provider dự phòng
    providers = [generator_type]
    if generator_type == "gemini" and OPENAI_API_KEY:
        providers.append("openai")
    elif generator_type == "openai" and GEMINI_API_KEY:
        providers.append("gemini")

    MAX_RETRIES = 1  # Chỉ retry 1 lần để tránh làm nghẽn UI quá lâu
    BASE_DELAY = 2   # Giảm delay cơ sở từ 5s xuống 2s

    for provider in providers:
        for attempt in range(MAX_RETRIES + 1):
            try:
                _wait_for_rate_limit()

                if provider == "gemini":
                    if not GEMINI_API_KEY:
                        raise ValueError("GEMINI_API_KEY chưa cấu hình")
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
                    headers = {"Content-Type": "application/json"}
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": temperature,
                            "maxOutputTokens": max_tokens
                        }
                    }
                    res = requests.post(url, json=payload, headers=headers, timeout=25)
                    res.raise_for_status()
                    return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

                elif provider == "openai":
                    if not OPENAI_API_KEY:
                        raise ValueError("OPENAI_API_KEY chưa cấu hình")
                    url = "https://api.openai.com/v1/chat/completions"
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {OPENAI_API_KEY}"
                    }
                    payload = {
                        "model": OPENAI_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                    res = requests.post(url, json=payload, headers=headers, timeout=25)
                    res.raise_for_status()
                    return res.json()["choices"][0]["message"]["content"].strip()

                elif provider == "ollama":
                    url = OLLAMA_API_URL
                    headers = {"Content-Type": "application/json"}
                    payload = {
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": temperature}
                    }
                    res = requests.post(url, json=payload, headers=headers, timeout=35)
                    res.raise_for_status()
                    return res.json()["response"].strip()

            except requests.exceptions.HTTPError as exc:
                is_rate_limit = exc.response is not None and exc.response.status_code == 429
                if is_rate_limit and attempt < MAX_RETRIES:
                    delay = BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "⏳ Rate-limited [%s] — retry %d/%d sau %ds...",
                        provider, attempt + 1, MAX_RETRIES, delay
                    )
                    time.sleep(delay)
                    continue
                
                logger.error(
                    "❌ Gọi LLM API [%s] lỗi HTTP (status %s): %s",
                    provider,
                    exc.response.status_code if exc.response else "N/A",
                    exc
                )
                break  # Bị lỗi khác hoặc hết lượt retry -> thử provider tiếp theo

            except Exception as exc:
                logger.error("❌ Gọi LLM API [%s] lỗi: %s", provider, exc)
                break  # Lỗi kết nối -> thử provider tiếp theo

    return ""


def _call_llm(prompt: str) -> str:
    """Gọi LLM API theo cấu hình hiện tại thông qua _execute_llm_request."""
    return _execute_llm_request(
        prompt, 
        RAG_GENERATOR_TYPE.lower(), 
        temperature=0.1, 
        max_tokens=150
    )


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
        r"^(tạm biệt|bye|goodbye)\b",
        r"^(bạn khỏe không|bạn có khỏe|bạn thế nào|how are you)\b",
        r"^(bạn có thể làm gì|bạn giúp gì được|help me|giúp tôi)\b",
        r"^(ok|okay|được rồi|vâng|đồng ý|tốt lắm|hay quá)\b",
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


def rewrite_query(original_query: str, missing_info_feedback: str) -> str:
    """
    Gọi LLM viết lại câu hỏi gốc tập trung vào phần thông tin còn thiếu
    dựa trên đánh giá phản hồi từ LLM Judge.
    """
    prompt = (
        "Bạn là trợ lý ảo tối ưu hóa tìm kiếm câu hỏi. Dưới đây là câu hỏi gốc và phản hồi về phần thông tin bị thiếu trong kết quả tìm kiếm hiện tại.\n"
        "Hãy viết lại câu hỏi gốc thành một câu hỏi mới chi tiết, tập trung tìm kiếm chính xác phần thông tin bị thiếu đó để cải thiện kết quả truy xuất tiếp theo.\n\n"
        f"Câu hỏi gốc: \"{original_query}\"\n"
        f"Thông tin bị thiếu: \"{missing_info_feedback}\"\n\n"
        "Chỉ trả về duy nhất câu hỏi mới bằng tiếng Việt, không thêm lời dẫn giải giải thích.\n"
        "Câu hỏi mới:"
    )
    res = _call_llm(prompt)
    return res.strip() if res else original_query


def evaluate_answer(query: str, context: str, answer: str) -> dict[str, Any]:
    """
    Gọi LLM Judge đánh giá tính chân thực (faithfulness), tính liên quan (relevance)
    và tính đầy đủ (sufficiency) của câu trả lời dựa trên ngữ cảnh.
    """
    import json
    prompt = f"""Bạn là một Thẩm phán AI (LLM Judge) đánh giá chất lượng hệ thống Hỏi đáp.
Nhiệm vụ của bạn là phân tích ba yếu tố sau và trả về kết quả đánh giá dưới dạng JSON:
1. "faithfulness": Câu trả lời có hoàn toàn bám sát vào ngữ cảnh được cung cấp không? (Trả về "yes" nếu đúng, "no" nếu câu trả lời chứa thông tin tự suy diễn/bịa đặt không có trong ngữ cảnh).
2. "relevance": Câu trả lời có tập trung trả lời đúng trọng tâm câu hỏi gốc không? (Trả về "yes" nếu đúng, "no" nếu trả lời lạc đề hoặc tránh né).
3. "sufficiency": Ngữ cảnh được cung cấp có đủ thông tin để trả lời câu hỏi gốc một cách trọn vẹn không? (Trả về "yes" nếu đủ, "no" nếu thông tin trong ngữ cảnh quá sơ sài hoặc không đề cập đến nội dung câu hỏi).
4. "feedback": Giải thích ngắn gọn lý do đánh giá, đặc biệt là nếu có tiêu chí nào bị đánh giá là "no".

Yêu cầu định dạng đầu ra: Chỉ trả về duy nhất chuỗi JSON hợp lệ theo cấu trúc sau, không giải thích gì thêm ngoài JSON:
{{
  "faithfulness": "yes" hoặc "no",
  "relevance": "yes" hoặc "no",
  "sufficiency": "yes" hoặc "no",
  "feedback": "lý do..."
}}

---
CÂU HỎI: "{query}"
NGỮ CẢNH: "{context}"
CÂU TRẢ LỜI: "{answer}"
---
KẾT QUẢ ĐÁNH GIÁ (JSON):"""

    res = _call_llm(prompt)
    try:
        # Tìm cụm JSON trong text
        match = re.search(r"\{.*\}", res, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return {
                "faithfulness": data.get("faithfulness", "yes").lower(),
                "relevance": data.get("relevance", "yes").lower(),
                "sufficiency": data.get("sufficiency", "yes").lower(),
                "feedback": data.get("feedback", "")
            }
    except Exception as e:
        logger.warning("Failed to parse LLM Judge evaluation JSON: %s. Raw response: %s", e, res)
        
    return {
        "faithfulness": "yes",
        "relevance": "yes",
        "sufficiency": "yes",
        "feedback": "Default pass due to parsing error"
    }
