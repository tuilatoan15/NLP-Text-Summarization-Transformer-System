"""
reranker.py — Cross-Encoder Reranker dùng BAAI/bge-reranker-v2-m3.

Kiến trúc:
  - Lần đầu gọi: tải model xuống (lazy load, cache vào singleton)
  - Sau đó: inference nhanh, không tải lại
  - Fallback: nếu model chưa có / không đủ RAM → dùng term_coverage scoring
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from .rag_config import (
    RERANKER_MODEL,
    RERANKER_MODEL_FALLBACK,
    RETRIEVAL_FINAL_TOP_K,
    RETRIEVAL_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Singleton — Cross-Encoder được load 1 lần, cache suốt vòng đời ứng dụng
# ─────────────────────────────────────────────────────────────────────────────
_reranker_lock = threading.Lock()
_reranker_instance: Any | None = None
_reranker_available: bool | None = None  # None = chưa thử, True/False = kết quả


def _get_reranker():
    """Lấy singleton Cross-Encoder, tải lần đầu nếu chưa có."""
    global _reranker_instance, _reranker_available

    if _reranker_available is not None:
        return _reranker_instance  # đã biết kết quả, trả ngay

    with _reranker_lock:
        if _reranker_available is not None:
            return _reranker_instance  # double-check sau khi lấy lock

        for model_name in (RERANKER_MODEL, RERANKER_MODEL_FALLBACK):
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                logger.info("🔄 Đang tải Cross-Encoder reranker: %s ...", model_name)
                model = CrossEncoder(model_name, max_length=512)
                _reranker_instance = model
                _reranker_available = True
                logger.info("✅ Cross-Encoder reranker đã sẵn sàng: %s", model_name)
                return model
            except Exception as exc:
                logger.warning(
                    "⚠️  Không thể tải reranker %s: %s — thử fallback...",
                    model_name, exc,
                )

        logger.warning(
            "⚠️  Không tải được Cross-Encoder reranker. "
            "Sẽ dùng term_coverage scoring thay thế."
        )
        _reranker_available = False
        _reranker_instance = None
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
class CrossEncoderReranker:
    """
    Reranker dùng Cross-Encoder để chấm điểm lại từng cặp (query, chunk).

    Luồng:
      Input:  top_k_initial chunks (thường 10) từ HybridRetriever
      Output: top_k_final chunks (thường 5) đã rerank, có rerank_score
    """

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = RETRIEVAL_FINAL_TOP_K,
        threshold: float = RETRIEVAL_THRESHOLD,
    ) -> list[dict[str, Any]]:
        """
        Rerank danh sách chunks theo Cross-Encoder score.

        Args:
            query:     Câu hỏi / query người dùng
            chunks:    Danh sách chunk từ hybrid retrieval (đã có combined_score)
            top_k:     Số chunk tối đa trả về sau rerank
            threshold: Ngưỡng rerank_score tối thiểu (0.0–1.0)

        Returns:
            Danh sách chunk đã được rerank, sắp xếp theo rerank_score giảm dần
        """
        if not chunks:
            return []

        reranker = _get_reranker()

        if reranker is not None:
            return self._rerank_with_cross_encoder(reranker, query, chunks, top_k, threshold)
        else:
            return self._rerank_fallback(query, chunks, top_k, threshold)

    def _rerank_with_cross_encoder(
        self,
        reranker,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Dùng Cross-Encoder thực sự để chấm điểm lại."""
        try:
            # Tạo danh sách cặp (query, chunk_text) cho Cross-Encoder
            pairs = [(query, chunk["text"]) for chunk in chunks]

            # Predict trả về raw logit (unbounded), dùng sigmoid để normalize về [0, 1]
            import torch  # type: ignore

            scores_raw = reranker.predict(pairs, convert_to_numpy=True)

            # Sigmoid normalize
            import numpy as np
            scores_normalized = 1.0 / (1.0 + np.exp(-scores_raw))

            # Gắn rerank_score vào mỗi chunk
            ranked: list[dict[str, Any]] = []
            for chunk, score in zip(chunks, scores_normalized):
                rerank_score = float(score)
                if rerank_score >= threshold:
                    entry = dict(chunk)
                    entry["rerank_score"] = round(rerank_score, 6)
                    ranked.append(entry)

            # Sắp xếp theo rerank_score giảm dần
            ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            result = ranked[:top_k]

            # Cập nhật rank
            for i, item in enumerate(result, start=1):
                item["rank"] = i

            logger.debug(
                "✅ Cross-Encoder reranked %d→%d chunks (threshold=%.2f)",
                len(chunks), len(result), threshold,
            )
            return result

        except Exception as exc:
            logger.error("❌ Cross-Encoder inference lỗi: %s — fallback term_coverage", exc)
            return self._rerank_fallback(query, chunks, top_k, threshold)

    def _rerank_fallback(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """
        Fallback khi Cross-Encoder chưa sẵn sàng.
        Dùng kết hợp combined_score (từ hybrid retrieval) + term_coverage.
        """
        import re

        query_terms = set(re.findall(r"\w+", query.lower()))

        scored: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_terms = set(re.findall(r"\w+", chunk["text"].lower()))
            coverage = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            # Kết hợp 60% combined_score + 40% term_coverage
            rerank_score = 0.6 * chunk["combined_score"] + 0.4 * coverage
            if rerank_score >= threshold * 0.7:  # nới lỏng ngưỡng vì đây là fallback
                entry = dict(chunk)
                entry["rerank_score"] = round(rerank_score, 6)
                scored.append(entry)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        result = scored[:top_k]
        for i, item in enumerate(result, start=1):
            item["rank"] = i
        return result

    @staticmethod
    def is_available() -> bool:
        """Trả về True nếu Cross-Encoder đã được load thành công."""
        return _get_reranker() is not None
