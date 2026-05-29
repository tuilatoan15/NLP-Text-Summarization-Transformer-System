"""
retriever.py — Hybrid Retriever tối ưu: 0.7 Vector + 0.3 BM25 + Cross-Encoder Reranker.

Pipeline cứng (không cho user thay đổi):
  1. Vector similarity (cosine) — bắt ngữ nghĩa toàn cục
  2. BM25 — giữ từ khóa chuyên ngành, tên riêng, số liệu
  3. Hybrid merge: 0.70V + 0.30BM25 → top 10
  4. Cross-Encoder reranker → top 5 (threshold > 0.40)
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .rag_config import (
    BM25_WEIGHT,
    VECTOR_WEIGHT,
    RETRIEVAL_PRE_RERANK_TOP_K,
    RETRIEVAL_FINAL_TOP_K,
    RETRIEVAL_THRESHOLD,
)
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None

token_pattern = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    if not text:
        return []
    if ViTokenizer is not None:
        try:
            text = ViTokenizer.tokenize(text)
        except Exception:
            pass
    return [t.lower() for t in token_pattern.findall(text)]


class HybridRetriever:
    """
    Hybrid Retriever chuẩn công nghiệp.

    Mọi trọng số và ngưỡng đều được hardcode từ rag_config.py.
    Tham số retrieval_mode và use_reranking trong API signature được giữ lại
    để tương thích ngược, nhưng luôn bị override bởi cấu hình cứng.
    """

    def __init__(self) -> None:
        self._reranker = CrossEncoderReranker()

    def retrieve(
        self,
        *,
        query: str,
        query_vector: list[float],
        chunks: list[dict[str, Any]],
        top_k: int = RETRIEVAL_FINAL_TOP_K,
        threshold: float = RETRIEVAL_THRESHOLD,
        retrieval_mode: str = "hybrid",      # Luôn dùng hybrid, param giữ để tương thích
        use_reranking: bool = True,           # Luôn bật reranking
    ) -> list[dict[str, Any]]:
        """
        Retrieval hoàn chỉnh: Hybrid scoring → Cross-Encoder reranking.

        Trả về top-K chunks đã được rerank, sắp xếp theo rerank_score.
        """
        if not chunks:
            return []

        # ── Bước 1: Tính BM25 scores ──────────────────────────────────────
        bm25_scores = self._bm25_scores(query, chunks)

        # ── Bước 2: Tính vector similarity (cosine) ───────────────────────
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q) or 1.0

        # ── Bước 3: Hybrid scoring với weight cứng 0.7V + 0.3BM25 ─────────
        scored: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            if "embedding_score" in chunk:
                sim = float(chunk["embedding_score"])
            else:
                vec = np.array(chunk.get("vector", []), dtype=np.float32)
                if vec.size == 0:
                    sim = 0.0
                else:
                    sim = float(np.dot(q, vec) / ((np.linalg.norm(vec) or 1.0) * q_norm))

            bm = bm25_scores[idx]
            # Hardcode: 70% vector + 30% BM25 (tốt hơn 65/35 cho tiếng Việt)
            combined = VECTOR_WEIGHT * sim + BM25_WEIGHT * bm

            scored.append(
                {
                    "chunk_id": chunk.get("id", chunk.get("chunk_id", "")),
                    "document_id": chunk["document_id"],
                    "filename": chunk["filename"],
                    "page": chunk.get("page"),
                    "text": chunk["text"],
                    "embedding_score": round(sim, 6),
                    "bm25_score": round(bm, 6),
                    "combined_score": round(combined, 6),
                }
            )

        # Sắp xếp theo combined_score, lấy pre-rerank top K
        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        pre_rerank = scored[:RETRIEVAL_PRE_RERANK_TOP_K]

        logger.debug(
            "🔍 Hybrid retrieval: %d chunks → pre-rerank top %d (threshold=%.2f)",
            len(chunks), len(pre_rerank), threshold,
        )

        # ── Bước 4: Cross-Encoder Reranking ───────────────────────────────
        final_top_k = min(top_k, RETRIEVAL_FINAL_TOP_K)
        reranked = self._reranker.rerank(
            query=query,
            chunks=pre_rerank,
            top_k=final_top_k,
            threshold=threshold,
        )

        # Nếu rerank trả về rỗng (tất cả dưới threshold), nới lỏng và thử lại
        if not reranked and pre_rerank:
            logger.warning(
                "⚠️  Reranker trả về rỗng (threshold=%.2f quá cao). "
                "Nới lỏng threshold về 0.2 để trả về ít nhất 1 kết quả.",
                threshold,
            )
            reranked = self._reranker.rerank(
                query=query,
                chunks=pre_rerank[:3],
                top_k=1,
                threshold=0.2,
            )

        logger.info(
            "✅ Retrieval hoàn tất: %d→%d chunks (reranker=%s)",
            len(pre_rerank),
            len(reranked),
            "CrossEncoder" if self._reranker.is_available() else "fallback",
        )
        return reranked

    # ─────────────────────── BM25 Implementation ────────────────────────────

    def _bm25_scores(self, query: str, chunks: list[dict[str, Any]]) -> list[float]:
        """
        BM25 (Okapi BM25) với k1=1.5, b=0.75.
        Kết quả được normalize về [0, 1] để dễ kết hợp với cosine similarity.
        """
        docs_tokens = [_tokens(chunk["text"]) for chunk in chunks]
        query_tokens = _tokens(query)

        if not query_tokens:
            return [0.0] * len(chunks)

        n_docs = len(docs_tokens)
        avgdl = sum(len(t) for t in docs_tokens) / max(n_docs, 1)

        df: dict[str, int] = defaultdict(int)
        for tokens in docs_tokens:
            for term in set(tokens):
                df[term] += 1

        k1 = 1.5
        b = 0.75
        scores: list[float] = []

        for tokens in docs_tokens:
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for term in query_tokens:
                if term not in tf:
                    continue
                idf = math.log(
                    (n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1.0
                )
                numer = tf[term] * (k1 + 1)
                denom = tf[term] + k1 * (1 - b + b * (dl / (avgdl or 1.0)))
                score += idf * (numer / (denom or 1.0))
            scores.append(float(score))

        # Normalize về [0, 1]
        max_score = max(scores) if scores else 0.0
        if max_score > 0:
            scores = [s / max_score for s in scores]
        return scores
