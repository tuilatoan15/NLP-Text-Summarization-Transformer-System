"""
context_compression.py — Hybrid Context Compression cho ChatRAG pipeline.

Tạo một Hybrid Summary từ toàn bộ retrieved chunks (reuse HybridSummarizer)
và giữ Top-N đoạn gốc theo CrossEncoder score để đưa vào prompt LLM.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .perf import StageTimer
from .rag_config import (
    RAG_CONTEXT_COMPRESSION,
    RAG_CONTEXT_COMPRESSION_THRESHOLD,
    RAG_SUMMARY_FOR_LONG_CONTEXT_ONLY,
    RAG_TOP_ORIGINAL_CHUNKS,
)

logger = logging.getLogger(__name__)


@dataclass
class CompressedContext:
    """Kết quả nén ngữ cảnh — truyền sang generator / prompt composer."""

    enabled: bool
    skipped_reason: str | None = None
    hybrid_summary: str = ""
    top_original_chunks: list[dict[str, Any]] = field(default_factory=list)
    all_retrieved: list[dict[str, Any]] = field(default_factory=list)
    compression_ratio: float = 1.0
    input_chars: int = 0
    output_chars: int = 0
    model_used: str | None = None
    hybrid_algo_key: str | None = None
    latency_s: float = 0.0
    # Adaptive Context Builder fields (optional, backward compatible)
    mode: str = "legacy"
    query_intent: str | None = None
    query_focus: str | None = None
    compression_tier: str | None = None
    dynamic_chunks_kept: int = 0
    summary_tokens: int = 0
    input_tokens_est: int = 0
    output_tokens_est: int = 0
    facts_preserved_count: int = 0
    citations: list[dict[str, Any]] = field(default_factory=list)
    context_details: dict[str, Any] = field(default_factory=dict)
    latency_saving_estimate: float = 0.0

    def effective_context_text(self) -> str:
        """Chuỗi ngữ cảnh dùng cho judge / hallucination audit."""
        parts: list[str] = []
        if self.hybrid_summary:
            parts.append(self.hybrid_summary)
        for chunk in self.top_original_chunks:
            parts.append(chunk.get("text", ""))
        if parts:
            return "\n\n".join(parts)
        return "\n\n".join(c.get("text", "") for c in self.all_retrieved)


def _chunk_score(chunk: dict[str, Any]) -> float:
    rerank = chunk.get("rerank_score")
    if rerank is not None:
        return float(rerank)
    return float(chunk.get("combined_score", 0.0))


def compute_total_context_chars(chunks: list[dict[str, Any]]) -> int:
    return sum(len(c.get("text", "")) for c in chunks)


def select_top_original_chunks(
    chunks: list[dict[str, Any]],
    n: int,
) -> list[dict[str, Any]]:
    """Giữ Top-N chunks gốc theo CrossEncoder score (không random)."""
    if not chunks or n <= 0:
        return []
    ranked = sorted(chunks, key=_chunk_score, reverse=True)
    return ranked[:n]


def pick_best_hybrid_key() -> tuple[str, str, str]:
    """
    Chọn hybrid algorithm tốt nhất trong số đã load.
    Returns: (hybrid_key, extractive_algo, abstractive_backbone)
    """
    from ai_models.model_registry import HYBRID_ALGORITHMS
    from .summarizer import _pick_available_model

    backbone = _pick_available_model() or "bartpho"
    extractive_prefs = ("textrank", "tfidf", "lexrank", "lsa")
    for ext in extractive_prefs:
        key = f"{ext}-{backbone}"
        if key in HYBRID_ALGORITHMS:
            return key, ext, backbone
    fallback = f"textrank-{backbone}"
    ext_algo, abs_algo = fallback.split("-", 1)
    return fallback, ext_algo, abs_algo


def _generate_hybrid_summary(
    chunks: list[dict[str, Any]],
    query: str,
) -> tuple[str, str | None, str | None]:
    """Sinh Hybrid Summary từ tất cả chunks — reuse pipeline.hybrid_summarizer."""
    from pipeline.hybrid_summarizer import summarize_retrieved_chunks

    return summarize_retrieved_chunks(chunks, query=query)


def should_compress_context(total_chars: int) -> bool:
    if not RAG_CONTEXT_COMPRESSION:
        return False
    if RAG_SUMMARY_FOR_LONG_CONTEXT_ONLY:
        return total_chars >= RAG_CONTEXT_COMPRESSION_THRESHOLD
    return True


def compress_retrieved_context(
    retrieved: list[dict[str, Any]],
    query: str,
    timer: StageTimer | None = None,
) -> CompressedContext:
    """
    Adaptive Hybrid Context Compression.
    - RAG_CONTEXT_COMPRESSION=0 → passthrough (backward compatible)
    - Ngữ cảnh ngắn (< threshold) → chỉ dùng chunks gốc
    - Ngữ cảnh dài → Hybrid Summary + Top-N passages
    """
    if not retrieved:
        return CompressedContext(
            enabled=False,
            skipped_reason="empty",
            all_retrieved=[],
        )

    input_chars = compute_total_context_chars(retrieved)

    if not RAG_CONTEXT_COMPRESSION:
        return CompressedContext(
            enabled=False,
            skipped_reason="disabled",
            all_retrieved=retrieved,
            input_chars=input_chars,
            output_chars=input_chars,
            compression_ratio=1.0,
        )

    if not should_compress_context(input_chars):
        return CompressedContext(
            enabled=False,
            skipped_reason="short_context",
            top_original_chunks=retrieved,
            all_retrieved=retrieved,
            input_chars=input_chars,
            output_chars=input_chars,
            compression_ratio=1.0,
        )

    t0 = time.perf_counter()
    if timer:
        timer.start("hybrid_summary")

    hybrid_key, _, _ = pick_best_hybrid_key()
    try:
        summary, model_used, algo_key = _generate_hybrid_summary(retrieved, query)
    except Exception as exc:
        logger.error("Hybrid context compression failed: %s", exc, exc_info=True)
        summary, model_used, algo_key = "", None, hybrid_key

    if timer:
        timer.stop("hybrid_summary")

    top_n = select_top_original_chunks(retrieved, RAG_TOP_ORIGINAL_CHUNKS)

    if not summary or len(summary.split()) < 5:
        logger.warning("Hybrid summary rỗng/ngắn — fallback chỉ dùng top original chunks")
        elapsed = time.perf_counter() - t0
        output_chars = compute_total_context_chars(top_n)
        return CompressedContext(
            enabled=False,
            skipped_reason="summary_failed",
            top_original_chunks=top_n or retrieved,
            all_retrieved=retrieved,
            input_chars=input_chars,
            output_chars=output_chars,
            compression_ratio=round(output_chars / max(input_chars, 1), 4),
            latency_s=elapsed,
        )

    output_chars = len(summary) + compute_total_context_chars(top_n)
    elapsed = time.perf_counter() - t0

    logger.info(
        "📦 Context compression: %d → %d chars (%.1f%%), top_%d originals, algo=%s",
        input_chars,
        output_chars,
        100.0 * output_chars / max(input_chars, 1),
        len(top_n),
        algo_key or hybrid_key,
    )

    return CompressedContext(
        enabled=True,
        hybrid_summary=summary,
        top_original_chunks=top_n,
        all_retrieved=retrieved,
        compression_ratio=round(output_chars / max(input_chars, 1), 4),
        input_chars=input_chars,
        output_chars=output_chars,
        model_used=model_used,
        hybrid_algo_key=algo_key or hybrid_key,
        latency_s=elapsed,
        mode="legacy",
    )


def build_retrieved_context(
    retrieved: list[dict[str, Any]],
    query: str,
    timer: StageTimer | None = None,
    *,
    document_ids: list[str] | None = None,
    stage_callback=None,
) -> CompressedContext:
    """
    Unified entry — Adaptive Context Builder khi RAG_ADAPTIVE_CONTEXT=1,
    ngược lại fallback Context Compression legacy.
    """
    from .rag_config import RAG_ADAPTIVE_CONTEXT

    if RAG_ADAPTIVE_CONTEXT:
        from .adaptive_context_builder import build_adaptive_context
        return build_adaptive_context(
            retrieved,
            query,
            timer,
            document_ids=document_ids,
            stage_callback=stage_callback,
        )
    return compress_retrieved_context(retrieved, query, timer)
