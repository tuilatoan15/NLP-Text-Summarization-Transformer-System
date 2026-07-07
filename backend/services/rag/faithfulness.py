"""
faithfulness.py — Tính faithfulness (grounding) và retrieval confidence cho Chat RAG.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from .rag_config import RAG_EVALUATE_HALLUCINATION

logger = logging.getLogger(__name__)


def compute_retrieval_confidence(retrieved: list[dict[str, Any]]) -> float:
    """Độ khớp retrieval: điểm rerank/combined của chunk hàng đầu."""
    if not retrieved:
        return 0.0
    top = retrieved[0].get("rerank_score")
    if top is None:
        top = retrieved[0].get("combined_score", 0.0)
    return round(min(0.99, float(top)), 4)


def _fast_grounding_coverage(answer: str, chunks: list[dict[str, Any]]) -> float:
    from utils.metrics import lexical_overlap
    from src.preprocess import split_sentences

    sentences = [s for s in split_sentences(answer) if len(s) >= 8]
    if not sentences or not chunks:
        return 0.0
    supported = 0
    for sentence in sentences:
        best = max(
            (lexical_overlap(sentence, c.get("text", "")) for c in chunks),
            default=0.0,
        )
        if best >= 0.35:
            supported += 1
    return round(supported / max(1, len(sentences)), 4)


def _fast_faithfulness_heuristic(
    answer: str,
    source_text: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    from evaluation.metrics import compute_faithfulness_score
    from src.fact_check import check_consistency

    consistency = check_consistency(answer, source_text, mode="fast")
    consistency_score = float(consistency.get("consistency_score", 0.0))
    grounding_coverage = _fast_grounding_coverage(answer, chunks)
    semantic_score = float(compute_faithfulness_score(answer, source_text))
    faithfulness = round(
        0.35 * consistency_score + 0.35 * grounding_coverage + 0.30 * semantic_score,
        4,
    )
    if grounding_coverage >= 0.7 and consistency_score >= 0.55:
        risk = "low"
    elif grounding_coverage >= 0.4 or consistency_score >= 0.45:
        risk = "medium"
    else:
        risk = "high"
    return {
        "faithfulness": faithfulness,
        "consistency_score": round(consistency_score, 4),
        "grounding_coverage": grounding_coverage,
        "semantic_coverage": round(semantic_score, 4),
        "hallucination_risk": risk,
    }


def compute_chat_faithfulness(
    answer: str,
    source_text: str,
    chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Tính faithfulness cho mỗi câu trả lời.
    Dùng audit đầy đủ khi RAG_EVALUATE_HALLUCINATION=1, ngược lại heuristic nhanh.
    """
    if not answer or not source_text:
        return {
            "faithfulness": 0.0,
            "consistency_score": 0.0,
            "grounding_coverage": 0.0,
            "semantic_coverage": 0.0,
            "hallucination_risk": "high",
        }

    if "không tìm thấy thông tin" in answer.lower():
        return {
            "faithfulness": 1.0,
            "consistency_score": 1.0,
            "grounding_coverage": 1.0,
            "semantic_coverage": 1.0,
            "hallucination_risk": "low",
        }

    formatted_chunks = [
        {"chunk_id": c.get("chunk_id") or c.get("id"), "text": c.get("text", "")}
        for c in (chunks or [])
    ]

    if RAG_EVALUATE_HALLUCINATION:
        try:
            from evaluation.hallucination import audit_summary

            audit_res = audit_summary(
                answer,
                source_text,
                chunks=formatted_chunks,
                mode="fast",
            )
            consistency_score = float(audit_res.get("consistency_score", 0.0))
            grounding_coverage = float(audit_res.get("grounding_coverage", 0.0))
            semantic_coverage = float(audit_res.get("semantic_coverage", 0.0))
            faithfulness_score = float(audit_res.get("faithfulness_score", 0.0))
            faithfulness = round(
                0.30 * faithfulness_score
                + 0.35 * consistency_score
                + 0.35 * grounding_coverage,
                4,
            )
            return {
                "faithfulness": faithfulness,
                "consistency_score": round(consistency_score, 4),
                "grounding_coverage": round(grounding_coverage, 4),
                "semantic_coverage": round(semantic_coverage, 4),
                "hallucination_risk": str(audit_res.get("hallucination_risk", "low")),
            }
        except Exception as exc:
            logger.error("Hallucination audit failed, falling back to heuristic: %s", exc)

    return _fast_faithfulness_heuristic(answer, source_text, formatted_chunks)


def is_comparison_query(query: str) -> bool:
    """Phát hiện câu hỏi so sánh / tổng hợp đa tài liệu."""
    q = query.lower()
    patterns = (
        r"\bso sánh\b",
        r"\bkhác nhau\b",
        r"\btương đồng\b",
        r"\bgiống\b",
        r"\bkhác\b",
        r"\bcả hai\b",
        r"\btất cả\b",
        r"\bmỗi (tài liệu|file|báo cáo)\b",
        r"\bgiữa (các |hai )?",
    )
    return any(re.search(p, q) for p in patterns)
