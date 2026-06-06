"""
summarizer.py — Transformer-based summarizer cho RAG pipeline.

Dùng lại các model đã được load trong hệ thống (BARTPho / ViT5 / mT5)
thông qua `src.model_loader.get_loaded_model` — tránh load lại, tiết kiệm RAM.

GenerationConfig được HARDCODE theo rag_config.py — không cho phép override từ ngoài.
"""
from __future__ import annotations

import logging
import unicodedata
from typing import Any

from .rag_config import (
    GENERATION_PROFILES,
    PREFERRED_SUMMARIZER_ORDER,
    SUMMARIZE_PROMPT_TEMPLATE,
    QA_PROMPT_TEMPLATE,
    GenerationProfile,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_gen_kwargs(profile: GenerationProfile) -> dict[str, Any]:
    """
    Chuyển GenerationProfile → dict kwargs cho model.generate().
    Tự động loại bỏ các param không tương thích:
      - early_stopping + do_sample=True gây RuntimeWarning
      - temperature/top_p không có nghĩa khi do_sample=False
    """
    kwargs: dict[str, Any] = {
        "num_beams": profile.num_beams,
        "no_repeat_ngram_size": profile.no_repeat_ngram_size,
        "repetition_penalty": profile.repetition_penalty,
        "length_penalty": profile.length_penalty,
        "min_new_tokens": profile.min_new_tokens,
        "max_new_tokens": profile.max_new_tokens,
    }
    if profile.do_sample:
        kwargs["do_sample"] = True
        if profile.temperature is not None:
            kwargs["temperature"] = profile.temperature
        if profile.top_p is not None:
            kwargs["top_p"] = profile.top_p
        # early_stopping không hợp lệ khi do_sample=True
    else:
        kwargs["do_sample"] = False
        kwargs["early_stopping"] = profile.early_stopping

    return kwargs


def _pick_available_model() -> str | None:
    """
    Trả về key của model đầu tiên đã được load trong registry.
    Ưu tiên theo PREFERRED_SUMMARIZER_ORDER = [bartpho, vit5, mt5].
    """
    try:
        from ai_models.model_loader import _registry  # type: ignore

        for key in PREFERRED_SUMMARIZER_ORDER:
            if _registry.is_loaded(key):
                logger.debug("Dùng model đã load: %s", key)
                return key
    except Exception as exc:
        logger.warning("Không truy cập được model registry: %s", exc)
    return None


def _clean_incomplete_sentence(text: str) -> str:
    """
    Tự động tìm kiếm dấu chấm câu cuối cùng trong văn bản tóm tắt sinh ra 
    và loại bỏ phần chữ thừa bị dở dang phía sau dấu chấm đó do chạm trần token.
    """
    text = (text or "").strip()
    if not text:
        return ""
    import re
    # Kiểm tra xem chuỗi đã kết thúc bằng một dấu chấm câu chuẩn (. ! ? … ” ") hay chưa
    if re.search(r'[.!?…]["”]?\s*$', text):
        return text
    # Tìm kiếm tất cả các vị trí kết thúc câu trong chuỗi
    ends = list(re.finditer(r'[.!?…]["”]?', text))
    if not ends:
        return text
    # Cắt đến dấu kết thúc câu cuối cùng
    return text[:ends[-1].end()].strip()


def _run_transformer_generate(
    model_key: str,
    input_text: str,
    profile: GenerationProfile,
) -> str:
    """
    Chạy model.generate() với GenerationConfig cứng.
    Trả về string đã clean, hoặc "" nếu thất bại.
    """
    try:
        import torch
        from src.model_loader import get_loaded_model  # type: ignore
        from src import config  # type: ignore

        loaded = get_loaded_model(model_key)
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.device
        use_fp16 = loaded.fp16

        # Thêm prefix "summarize:" cho T5-family models
        if model_key in {"vit5", "mt5"}:
            input_text = f"summarize: {input_text}"

        encoded = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_INPUT_TOKENS,
            padding=False,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        gen_kwargs = _build_gen_kwargs(profile)

        with torch.inference_mode():
            if use_fp16 and device.type == "cuda":
                import torch.amp
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    output_ids = model.generate(**encoded, **gen_kwargs)
            else:
                output_ids = model.generate(**encoded, **gen_kwargs)

        is_t5 = model_key in {"vit5", "mt5"}
        decoded = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=is_t5,
        )
        raw = decoded[0] if decoded else ""
        raw = unicodedata.normalize("NFC", raw)

        # Loại bỏ prefix "summarize:" nếu model sinh lại
        for prefix in ("summarize:", "summarize :", "tóm tắt:"):
            if raw.lower().startswith(prefix):
                raw = raw[len(prefix):].strip()

        # Dọn dẹp câu dở dang do chạm trần token
        raw = _clean_incomplete_sentence(raw)

        return raw.strip()

    except Exception as exc:
        logger.error("❌ Transformer generate [%s] lỗi: %s", model_key, exc)
        return ""
def _run_llm_api(prompt: str, generator_type: str) -> str:
    """Gọi LLM API tương ứng để sinh văn bản."""
    import requests
    from .rag_config import (
        GEMINI_API_KEY,
        OPENAI_API_KEY,
        GEMINI_MODEL,
        OPENAI_MODEL,
        OLLAMA_API_URL,
        OLLAMA_MODEL,
    )

    try:
        if generator_type == "gemini":
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY chưa được cấu hình trong .env")
            
            # Gemini API Endpoint
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 800
                }
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            res_data = response.json()
            return res_data["candidates"][0]["content"]["parts"][0]["text"].strip()

        elif generator_type == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY chưa được cấu hình trong .env")
            
            # OpenAI Chat Completion Endpoint
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            payload = {
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 800
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            res_data = response.json()
            return res_data["choices"][0]["message"]["content"].strip()

        elif generator_type == "ollama":
            # Ollama API Endpoint
            url = OLLAMA_API_URL
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2
                }
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=45)
            response.raise_for_status()
            res_data = response.json()
            return res_data["response"].strip()

    except Exception as exc:
        logger.error("❌ Gọi LLM API [%s] lỗi: %s", generator_type, exc)
    
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────────

class RAGTransformerSummarizer:
    """
    Summarizer tích hợp vào RAG pipeline.

    Có 2 chế độ:
      1. summarize_context() — tóm tắt tài liệu từ các chunks đã retrieve
      2. answer_question()   — trả lời câu hỏi dựa trên context

    Mọi GenerationConfig đều HARDCODE từ rag_config.py.
    """

    def summarize_context(
        self,
        contexts: list[dict[str, Any]],
        *,
        max_context_chars: int = 4000,
    ) -> dict[str, Any]:
        """
        Tóm tắt toàn bộ context từ các chunks đã được retrieve + rerank.

        Args:
            contexts:          Danh sách chunks đã rerank (có text, filename, page...)
            max_context_chars: Cắt context nếu quá dài để tránh OOM

        Returns:
            dict gồm: summary, model_used, fallback_used, word_count
        """
        if not contexts:
            return {
                "summary": "Không tìm thấy nội dung phù hợp trong tài liệu.",
                "model_used": None,
                "fallback_used": True,
                "word_count": 0,
            }

        # Ghép context từ các chunks (theo thứ tự rank)
        context_parts = []
        for i, chunk in enumerate(contexts, start=1):
            filename = chunk.get("filename", "?")
            page = chunk.get("page")
            page_info = f" trang {page}" if page else ""
            context_parts.append(
                f"[Nguồn {i} — {filename}{page_info}]\n{chunk['text']}"
            )
        full_context = "\n\n".join(context_parts)

        # Cắt nếu quá dài
        if len(full_context) > max_context_chars:
            full_context = full_context[:max_context_chars] + "..."

        # Build prompt
        prompt = SUMMARIZE_PROMPT_TEMPLATE.format(context=full_context)

        # Thử generator từ cấu hình
        from .rag_config import RAG_GENERATOR_TYPE
        summary = ""
        fallback_used = False
        model_key = ""

        if RAG_GENERATOR_TYPE in {"gemini", "openai", "ollama"}:
            logger.info("Dùng LLM API [%s] để tóm tắt", RAG_GENERATOR_TYPE)
            summary = _run_llm_api(prompt, RAG_GENERATOR_TYPE)
            if summary:
                model_key = f"{RAG_GENERATOR_TYPE}_api"
            else:
                logger.warning("⚠️ LLM API [%s] failed — fallback sang local/extractive", RAG_GENERATOR_TYPE)

        if not summary:
            model_key = _pick_available_model()
            if model_key:
                profile = GENERATION_PROFILES[model_key]
                summary = _run_transformer_generate(model_key, prompt, profile)

        if not summary or len(summary.split()) < 10:
            # Fallback: extractive — ghép các câu quan trọng nhất
            logger.warning("⚠️ Transformer summarizer failed — dùng extractive fallback")
            summary = self._extractive_fallback(contexts, max_sentences=8)
            fallback_used = True
            model_key = "extractive_fallback"

        word_count = len(summary.split())
        return {
            "summary": summary,
            "model_used": model_key,
            "fallback_used": fallback_used,
            "word_count": word_count,
        }

    def answer_question(
        self,
        question: str,
        contexts: list[dict[str, Any]],
        *,
        chat_history: list[dict[str, Any]] | None = None,
        max_context_chars: int = 3000,
        general_chat: bool = False,
    ) -> dict[str, Any]:
        """
        Trả lời câu hỏi dựa trên context đã retrieve và lịch sử trò chuyện.

        Returns:
            dict gồm: answer, confidence, grounded, model_used, fallback_used
        """
        if general_chat:
            history_text = "Không có"
            if chat_history:
                history_lines = []
                for msg in chat_history[-4:]:
                    role = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
                    history_lines.append(f"{role}: {msg.get('content', '')}")
                if history_lines:
                    history_text = "\n".join(history_lines)
            prompt = (
                "Bạn là trợ lý ảo AI tiếng Việt thông minh và thân thiện.\n"
                "Hãy trò chuyện hoặc trả lời câu hỏi dưới đây một cách lịch sự, tự nhiên và hữu ích.\n"
                "Hãy tham khảo LỊCH SỬ HỘI THOẠI (nếu có) để cuộc trò chuyện được tiếp tục mạch lạc.\n\n"
                f"LỊCH SỬ HỘI THOẠI:\n{history_text}\n\n"
                f"CÂU HỎI HIỆN TẠI: {question}\n\n"
                "TRẢ LỜI:"
            )
        else:
            if not contexts:
                return {
                    "answer": "Không tìm thấy thông tin trong tài liệu.",
                    "confidence": 0.0,
                    "grounded": True,
                    "model_used": None,
                    "fallback_used": True,
                }

            # Ghép context
            context_parts = []
            for i, chunk in enumerate(contexts, start=1):
                filename = chunk.get("filename", "?")
                context_parts.append(f"[{i}] {chunk['text']}")
            full_context = "\n\n".join(context_parts)

            if len(full_context) > max_context_chars:
                full_context = full_context[:max_context_chars] + "..."

            # Định dạng chat_history
            history_text = "Không có"
            if chat_history:
                history_lines = []
                for msg in chat_history[-4:]:  # Lấy tối đa 4 tin nhắn gần nhất (2 lượt hỏi-đáp)
                    role = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
                    history_lines.append(f"{role}: {msg.get('content', '')}")
                if history_lines:
                    history_text = "\n".join(history_lines)

            prompt = QA_PROMPT_TEMPLATE.format(context=full_context, chat_history=history_text, question=question)

        from .rag_config import RAG_GENERATOR_TYPE
        answer = ""
        fallback_used = False
        model_key = ""

        if RAG_GENERATOR_TYPE in {"gemini", "openai", "ollama"}:
            logger.info("Dùng LLM API [%s] để trả lời câu hỏi", RAG_GENERATOR_TYPE)
            answer = _run_llm_api(prompt, RAG_GENERATOR_TYPE)
            if answer:
                model_key = f"{RAG_GENERATOR_TYPE}_api"
            else:
                logger.warning("⚠️ LLM API [%s] failed — fallback sang local/extractive", RAG_GENERATOR_TYPE)

        if not answer:
            model_key = _pick_available_model()
            if model_key:
                profile = GENERATION_PROFILES[model_key]
                answer = _run_transformer_generate(model_key, prompt, profile)

        if not answer or len(answer.split()) < 3:
            # Fallback: trả về câu liên quan nhất từ context
            answer = self._sentence_fallback(question, contexts)
            fallback_used = True
            model_key = "extractive_fallback"

        confidence = min(0.99, contexts[0]["combined_score"]) if contexts else 0.0
        if contexts[0].get("rerank_score") is not None:
            confidence = min(0.99, contexts[0]["rerank_score"])

        return {
            "answer": answer,
            "confidence": round(confidence, 4),
            "grounded": True,
            "model_used": model_key,
            "fallback_used": fallback_used,
        }

    # ─────────────────────────── Fallback methods ───────────────────────────

    def _extractive_fallback(
        self,
        contexts: list[dict[str, Any]],
        max_sentences: int = 8,
    ) -> str:
        """Ghép các câu từ top chunks theo thứ tự xuất hiện tự nhiên."""
        import re
        sentences: list[str] = []
        for chunk in contexts:
            for sent in re.split(r"(?<=[.!?])\s+", chunk["text"]):
                s = sent.strip()
                if s and len(s) > 15:
                    sentences.append(s)
            if len(sentences) >= max_sentences:
                break
        return " ".join(sentences[:max_sentences])

    def _sentence_fallback(
        self,
        question: str,
        contexts: list[dict[str, Any]],
    ) -> str:
        """Trả về câu liên quan nhất với question từ context."""
        import re
        q_terms = set(re.findall(r"\w+", question.lower()))
        best_score = -1.0
        best_sentence = ""
        for chunk in contexts:
            for sent in re.split(r"(?<=[.!?])\s+", chunk["text"]):
                s = sent.strip()
                if not s:
                    continue
                s_terms = set(re.findall(r"\w+", s.lower()))
                score = len(q_terms & s_terms) / max(len(q_terms), 1)
                if score > best_score:
                    best_score = score
                    best_sentence = s
        return best_sentence or "Không tìm thấy thông tin phù hợp trong tài liệu."
