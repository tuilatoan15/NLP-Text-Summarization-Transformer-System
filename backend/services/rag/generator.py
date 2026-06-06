"""
generator.py — GroundedGenerator tích hợp RAGTransformerSummarizer.

Luồng ưu tiên:
  1. Transformer summarizer (BARTPho/ViT5/mT5 đã load) → câu trả lời mạch lạc
  2. Extractive fallback → câu liên quan nhất từ context
  3. "Không tìm thấy thông tin" nếu context rỗng

GenerationConfig hoàn toàn HARDCODE từ rag_config.py.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from .rag_config import QA_PROMPT_TEMPLATE, SUMMARIZE_PROMPT_TEMPLATE
from .summarizer import RAGTransformerSummarizer

logger = logging.getLogger(__name__)


class GroundedGenerator:
    insufficient_context_message = "Không tìm thấy thông tin trong tài liệu."

    def __init__(self) -> None:
        self._summarizer = RAGTransformerSummarizer()

    # ─────────────────────────── Chat Q&A ────────────────────────────────────

    def build_answer(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        *,
        chat_history: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,  # Giữ param để tương thích API, không dùng (hardcoded)
        general_chat: bool = False,
    ) -> dict[str, Any]:
        """
        Sinh câu trả lời cho query dựa trên contexts đã retrieve + rerank.

        temperature bị ignore — GenerationConfig đã hardcode trong rag_config.py.
        """
        if not contexts and not general_chat:
            return {
                "answer": self.insufficient_context_message,
                "confidence": 0.0,
                "grounded": True,
                "model_used": None,
                "fallback_used": True,
                "temperature_used": temperature,
            }

        result = self._summarizer.answer_question(
            query, contexts, chat_history=chat_history, general_chat=general_chat
        )

        return {
            "answer": result["answer"],
            "confidence": result["confidence"],
            "grounded": not general_chat,
            "model_used": result.get("model_used"),
            "fallback_used": result.get("fallback_used", False),
            "temperature_used": temperature,
        }

    # ─────────────────────────── Document Summary ─────────────────────────────

    def build_document_summary(
        self,
        contexts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Tóm tắt toàn bộ tài liệu từ các chunks đã retrieve.
        Dùng cho endpoint /summarize của document.
        """
        return self._summarizer.summarize_context(contexts)

    # ─────────────────────────── Prompt template ──────────────────────────────

    def prompt_template(self, contexts: list[dict[str, Any]], question: str, chat_history: str = "Không có") -> str:
        """Render prompt template tiếng Việt chuẩn (dùng để debug/log)."""
        blocks = []
        for idx, c in enumerate(contexts, start=1):
            filename = c.get("filename", "?")
            page = c.get("page")
            rerank = c.get("rerank_score")
            score_info = f"rerank={rerank:.3f}" if rerank is not None else f"score={c.get('combined_score', 0):.3f}"
            blocks.append(
                f"[Nguồn {idx} — {filename}"
                + (f" trang {page}" if page else "")
                + f" | {score_info}]\n{c['text']}"
            )
        context_text = "\n\n".join(blocks)
        return QA_PROMPT_TEMPLATE.format(context=context_text, chat_history=chat_history, question=question)

    def summarize_prompt_template(self, contexts: list[dict[str, Any]]) -> str:
        """Render summarize prompt template để debug."""
        blocks = []
        for idx, c in enumerate(contexts, start=1):
            blocks.append(
                f"[Nguồn {idx} — {c.get('filename', '?')}]\n{c['text']}"
            )
        return SUMMARIZE_PROMPT_TEMPLATE.format(context="\n\n".join(blocks))
