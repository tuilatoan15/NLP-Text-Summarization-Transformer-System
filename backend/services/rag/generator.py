"""
generator.py — GroundedGenerator tích hợp RAGTransformerSummarizer + Hybrid Context Compression.

Luồng ưu tiên:
  1. Transformer summarizer (BARTPho/ViT5/mT5 đã load) → câu trả lời mạch lạc
  2. Extractive fallback → câu liên quan nhất từ context
  3. "Không tìm thấy thông tin" nếu context rỗng

GenerationConfig hoàn toàn HARDCODE từ rag_config.py.
"""
from __future__ import annotations

import logging
from typing import Any

from .context_compression import CompressedContext
from .rag_config import (
    ADAPTIVE_QA_PROMPT_TEMPLATE,
    COMPRESSED_QA_PROMPT_TEMPLATE,
    QA_PROMPT_TEMPLATE,
    SUMMARIZE_PROMPT_TEMPLATE,
)
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
        temperature: float = 0.2,
        general_chat: bool = False,
        compressed_context: CompressedContext | None = None,
    ) -> dict[str, Any]:
        """
        Sinh câu trả lời cho query dựa trên contexts đã retrieve + rerank
        hoặc CompressedContext (Hybrid Summary + Top-N passages).
        """
        if not contexts and not general_chat and not (
            compressed_context and compressed_context.enabled
        ):
            return {
                "answer": self.insufficient_context_message,
                "confidence": 0.0,
                "grounded": True,
                "model_used": None,
                "fallback_used": True,
                "temperature_used": temperature,
            }

        result = self._summarizer.answer_question(
            query,
            contexts,
            chat_history=chat_history,
            general_chat=general_chat,
            compressed_context=compressed_context,
        )

        return {
            "answer": result["answer"],
            "confidence": result["confidence"],
            "grounded": not general_chat,
            "model_used": result.get("model_used"),
            "fallback_used": result.get("fallback_used", False),
            "temperature_used": temperature,
        }

    def stream_answer(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        *,
        chat_history: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        general_chat: bool = False,
        compressed_context: CompressedContext | None = None,
    ):
        """Stream token thật từ summarizer."""
        if not contexts and not general_chat and not (
            compressed_context and compressed_context.enabled
        ):
            yield self.insufficient_context_message
            return
        yield from self._summarizer.stream_answer_tokens(
            query,
            contexts,
            chat_history=chat_history,
            general_chat=general_chat,
            compressed_context=compressed_context,
        )

    # ─────────────────────────── Document Summary ─────────────────────────────

    def build_document_summary(
        self,
        contexts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Tóm tắt toàn bộ tài liệu từ các chunks đã retrieve."""
        return self._summarizer.summarize_context(contexts)

    # ─────────────────────────── Prompt composer ──────────────────────────────

    @staticmethod
    def format_original_passages(chunks: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for idx, c in enumerate(chunks, start=1):
            filename = c.get("filename", "?")
            page = c.get("page")
            rerank = c.get("rerank_score")
            score_info = (
                f"rerank={rerank:.3f}"
                if rerank is not None
                else f"score={c.get('combined_score', 0):.3f}"
            )
            blocks.append(
                f"[Đoạn {idx} — {filename}"
                + (f" trang {page}" if page else "")
                + f" | {score_info}]\n{c.get('text', '')}"
            )
        return "\n\n".join(blocks) if blocks else "(Không có đoạn gốc bổ sung)"

    @staticmethod
    def format_chat_history(chat_history: list[dict[str, Any]] | None) -> str:
        if not chat_history:
            return "Không có"
        lines: list[str] = []
        for msg in chat_history[-4:]:
            role = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
            lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines) if lines else "Không có"

    def _render_compressed_prompt(
        self,
        question: str,
        compressed_context: CompressedContext,
        *,
        history_text: str,
    ) -> str:
        passages = self.format_original_passages(compressed_context.top_original_chunks)
        if compressed_context.mode == "adaptive":
            return ADAPTIVE_QA_PROMPT_TEMPLATE.format(
                query_focus=compressed_context.query_focus or question,
                document_summary=(
                    compressed_context.hybrid_summary
                    or "(Ngữ cảnh ngắn — không cần tóm tắt, ưu tiên VERIFIED ORIGINAL PASSAGES)"
                ),
                original_passages=passages,
                chat_history=history_text,
                question=question,
            )
        return COMPRESSED_QA_PROMPT_TEMPLATE.format(
            document_summary=compressed_context.hybrid_summary,
            original_passages=passages,
            chat_history=history_text,
            question=question,
        )

    def compose_prompt(
        self,
        question: str,
        contexts: list[dict[str, Any]],
        *,
        chat_history: list[dict[str, Any]] | None = None,
        compressed_context: CompressedContext | None = None,
    ) -> str:
        """Prompt Composer — chọn template theo compression state."""
        history_text = self.format_chat_history(chat_history)

        if compressed_context and compressed_context.enabled:
            return self._render_compressed_prompt(
                question, compressed_context, history_text=history_text,
            )

        blocks = []
        for idx, c in enumerate(contexts, start=1):
            filename = c.get("filename", "?")
            page = c.get("page")
            rerank = c.get("rerank_score")
            score_info = (
                f"rerank={rerank:.3f}"
                if rerank is not None
                else f"score={c.get('combined_score', 0):.3f}"
            )
            blocks.append(
                f"[Nguồn {idx} — {filename}"
                + (f" trang {page}" if page else "")
                + f" | {score_info}]\n{c['text']}"
            )
        context_text = "\n\n".join(blocks)
        return QA_PROMPT_TEMPLATE.format(
            context=context_text,
            chat_history=history_text,
            question=question,
        )

    def prompt_template(
        self,
        contexts: list[dict[str, Any]],
        question: str,
        chat_history: str = "Không có",
        compressed_context: CompressedContext | None = None,
    ) -> str:
        """Render prompt template tiếng Việt chuẩn (dùng để debug/log)."""
        if compressed_context and compressed_context.enabled:
            return self._render_compressed_prompt(
                question, compressed_context, history_text=chat_history,
            )
        blocks = []
        for idx, c in enumerate(contexts, start=1):
            filename = c.get("filename", "?")
            page = c.get("page")
            rerank = c.get("rerank_score")
            score_info = (
                f"rerank={rerank:.3f}"
                if rerank is not None
                else f"score={c.get('combined_score', 0):.3f}"
            )
            blocks.append(
                f"[Nguồn {idx} — {filename}"
                + (f" trang {page}" if page else "")
                + f" | {score_info}]\n{c['text']}"
            )
        context_text = "\n\n".join(blocks)
        return QA_PROMPT_TEMPLATE.format(
            context=context_text, chat_history=chat_history, question=question
        )

    def summarize_prompt_template(self, contexts: list[dict[str, Any]]) -> str:
        """Render summarize prompt template để debug."""
        blocks = []
        for idx, c in enumerate(contexts, start=1):
            blocks.append(
                f"[Nguồn {idx} — {c.get('filename', '?')}]\n{c['text']}"
            )
        return SUMMARIZE_PROMPT_TEMPLATE.format(context="\n\n".join(blocks))
