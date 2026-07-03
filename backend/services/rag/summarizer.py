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
    resolve_generation_profile,
    RAG_SUMMARIZE_BATCH_SIZE,
)
from .context_compression import CompressedContext

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
        from src.model_loader import _registry  # type: ignore

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
        encoded = {k: v.to(device, non_blocking=device.type == "cuda") for k, v in encoded.items()}
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


def _run_transformer_generate_batch(
    model_key: str,
    input_texts: list[str],
    profile: GenerationProfile,
) -> list[str]:
    """Batch inference cho nhiều prompt — giảm overhead tokenizer/GPU."""
    if not input_texts:
        return []
    if len(input_texts) == 1:
        return [_run_transformer_generate(model_key, input_texts[0], profile)]

    try:
        import torch
        from src.model_loader import get_loaded_model
        from src import config

        loaded = get_loaded_model(model_key)
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.device
        use_fp16 = loaded.fp16

        prepared = []
        for text in input_texts:
            if model_key in {"vit5", "mt5"}:
                prepared.append(f"summarize: {text}")
            else:
                prepared.append(text)

        encoded = tokenizer(
            prepared,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_INPUT_TOKENS,
            padding=True,
        )
        encoded = {k: v.to(device, non_blocking=device.type == "cuda") for k, v in encoded.items()}
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
        results: list[str] = []
        for raw in decoded:
            raw = unicodedata.normalize("NFC", raw or "")
            for prefix in ("summarize:", "summarize :", "tóm tắt:"):
                if raw.lower().startswith(prefix):
                    raw = raw[len(prefix):].strip()
            results.append(_clean_incomplete_sentence(raw.strip()))
        return results

    except Exception as exc:
        logger.warning("Batch generate thất bại [%s]: %s — fallback tuần tự", model_key, exc)
        return [_run_transformer_generate(model_key, t, profile) for t in input_texts]


def _run_transformer_generate_stream(
    model_key: str,
    input_text: str,
    profile: GenerationProfile,
):
    """Yield token từ TextIteratorStreamer (local transformer)."""
    try:
        import torch
        from threading import Thread
        from transformers import TextIteratorStreamer
        from src.model_loader import get_loaded_model
        from src import config

        loaded = get_loaded_model(model_key)
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.device
        use_fp16 = loaded.fp16

        if model_key in {"vit5", "mt5"}:
            input_text = f"summarize: {input_text}"

        encoded = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_INPUT_TOKENS,
            padding=False,
        )
        encoded = {k: v.to(device, non_blocking=device.type == "cuda") for k, v in encoded.items()}
        gen_kwargs = _build_gen_kwargs(profile)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        def _generate():
            with torch.inference_mode():
                if use_fp16 and device.type == "cuda":
                    import torch.amp
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        model.generate(**encoded, **gen_kwargs)
                else:
                    model.generate(**encoded, **gen_kwargs)

        thread = Thread(target=_generate, daemon=True)
        thread.start()
        for token in streamer:
            if token:
                yield token
        thread.join(timeout=120)
    except Exception as exc:
        logger.error("❌ Transformer stream [%s] lỗi: %s", model_key, exc)


def _run_llm_api(prompt: str, generator_type: str) -> str:
    """Gọi LLM API tương ứng để sinh văn bản (có retry + rate-limit throttling và fallback)."""
    from .agent import _execute_llm_request
    return _execute_llm_request(prompt, generator_type, temperature=0.2, max_tokens=800)


def _run_llm_api_stream(prompt: str, generator_type: str):
    """Stream token từ LLM API."""
    from .agent import _execute_llm_request_stream
    yield from _execute_llm_request_stream(prompt, generator_type, temperature=0.2, max_tokens=800)


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
                profile = resolve_generation_profile(model_key)
                summary = _run_transformer_generate(model_key, full_context, profile)

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
        compressed_context: CompressedContext | None = None,
    ) -> dict[str, Any]:
        """
        Trả lời câu hỏi dựa trên context đã retrieve hoặc CompressedContext.

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
            effective_contexts = contexts
        else:
            use_compression = compressed_context is not None and compressed_context.enabled
            effective_contexts = (
                compressed_context.top_original_chunks
                if use_compression
                else contexts
            )
            if not effective_contexts and not use_compression:
                return {
                    "answer": "Không tìm thấy thông tin trong tài liệu.",
                    "confidence": 0.0,
                    "grounded": True,
                    "model_used": None,
                    "fallback_used": True,
                }

            from .generator import GroundedGenerator

            prompt = GroundedGenerator().compose_prompt(
                question,
                contexts,
                chat_history=chat_history,
                compressed_context=compressed_context,
            )
            full_context = compressed_context.effective_context_text() if use_compression else "\n\n".join(
                f"[{i}] {chunk['text']}" for i, chunk in enumerate(contexts, start=1)
            )
            if len(full_context) > max_context_chars:
                full_context = full_context[:max_context_chars] + "..."

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
                profile = resolve_generation_profile(model_key)
                # Tối giản prompt cho local model (BARTPho/ViT5/mT5) để tránh lỗi lặp lại system instructions
                local_prompt = (
                    f"Ngữ cảnh:\n{full_context}\n\n"
                    f"Hãy trích xuất và tóm tắt thông tin từ ngữ cảnh trên để trả lời câu hỏi: {question}\n"
                    "Trả lời:"
                )
                answer = _run_transformer_generate(model_key, local_prompt, profile)

        if not answer or len(answer.split()) < 3:
            # Fallback: trả về câu liên quan nhất từ context
            fallback_ctx = effective_contexts if not general_chat else contexts
            answer = self._sentence_fallback(question, fallback_ctx)
            fallback_used = True
            model_key = "extractive_fallback"

        confidence_source = effective_contexts if not general_chat else contexts
        confidence = min(0.99, confidence_source[0]["combined_score"]) if confidence_source else 0.0
        if confidence_source and confidence_source[0].get("rerank_score") is not None:
            confidence = min(0.99, confidence_source[0]["rerank_score"])

        return {
            "answer": answer,
            "confidence": round(confidence, 4),
            "grounded": True,
            "model_used": model_key,
            "fallback_used": fallback_used,
        }

    def stream_answer_tokens(
        self,
        question: str,
        contexts: list[dict[str, Any]],
        *,
        chat_history: list[dict[str, Any]] | None = None,
        max_context_chars: int = 3000,
        general_chat: bool = False,
        compressed_context: CompressedContext | None = None,
    ):
        """Generator token thật cho streaming SSE."""
        from .rag_config import RAG_GENERATOR_TYPE

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
            full_context = ""
        else:
            use_compression = compressed_context is not None and compressed_context.enabled
            effective_contexts = (
                compressed_context.top_original_chunks
                if use_compression
                else contexts
            )
            if not effective_contexts and not use_compression:
                yield "Không tìm thấy thông tin trong tài liệu."
                return
            from .generator import GroundedGenerator

            prompt = GroundedGenerator().compose_prompt(
                question,
                contexts,
                chat_history=chat_history,
                compressed_context=compressed_context,
            )
            full_context = (
                compressed_context.effective_context_text()
                if use_compression
                else "\n\n".join(
                    f"[{i}] {chunk['text']}" for i, chunk in enumerate(contexts, start=1)
                )
            )
            if len(full_context) > max_context_chars:
                full_context = full_context[:max_context_chars] + "..."

        if RAG_GENERATOR_TYPE in {"gemini", "openai", "ollama"}:
            yielded = False
            for token in _run_llm_api_stream(prompt, RAG_GENERATOR_TYPE):
                yielded = True
                yield token
            if yielded:
                return

        model_key = _pick_available_model()
        if model_key:
            profile = resolve_generation_profile(model_key)
            if general_chat:
                stream_input = prompt
            else:
                stream_input = (
                    f"Ngữ cảnh:\n{full_context}\n\n"
                    f"Hãy trích xuất và tóm tắt thông tin từ ngữ cảnh trên để trả lời câu hỏi: {question}\n"
                    "Trả lời:"
                )
            for token in _run_transformer_generate_stream(model_key, stream_input, profile):
                yield token
            return

        fallback_ctx = effective_contexts if not general_chat else contexts
        yield (
            self._sentence_fallback(question, fallback_ctx)
            if fallback_ctx
            else "Xin lỗi, tôi không thể trả lời lúc này."
        )

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
