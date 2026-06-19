"""
retriever.py — Hybrid Retriever thế hệ mới sử dụng bộ trộn Reciprocal Rank Fusion (RRF)
kết hợp giữa Dense semantic search (BGE-M3) và Sparse keyword search (Okapi BM25),
sau đó đưa qua Cross-Encoder Reranker (BGE-Reranker-Large).
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
    Hybrid Retriever sử dụng giải thuật Reciprocal Rank Fusion (RRF)
    kết hợp với Cross-Encoder Reranker chuẩn công nghiệp.
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
        retrieval_mode: str = "hybrid",      # Giữ nguyên tương thích ngược
        use_reranking: bool = True,           # Luôn bật reranking
    ) -> list[dict[str, Any]]:
        """
        Thực hiện truy xuất thông tin đa tầng:
          Bước 1: Chấm điểm BM25 Okapi cho từng chunk.
          Bước 2: Tính Cosine Similarity của Vector Embedding.
          Bước 3: Thực hiện Reciprocal Rank Fusion (RRF) trộn 2 bảng xếp hạng độc lập.
          Bước 4: Sắp xếp theo RRF Score và lấy Top Candidates.
          Bước 5: Đưa qua Cross-Encoder Reranker để lấy kết quả có độ liên quan sâu sắc nhất.
        """
        import time
        self.last_latency = {
            "bm25": 0.0,
            "vector_rrf": 0.0,
            "rerank": 0.0
        }

        if not chunks:
            return []

        # ── Bước 1: Tính BM25 scores ──────────────────────────────────────
        t_bm25_start = time.perf_counter()
        bm25_scores = self._bm25_scores(query, chunks)
        t_bm25 = time.perf_counter() - t_bm25_start

        # ── Bước 2: Tính Dense Vector similarity (cosine) ──────────────────
        t_dense_start = time.perf_counter()
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q) or 1.0

        dense_scored: list[tuple[int, float]] = []
        for idx, chunk in enumerate(chunks):
            if "embedding_score" in chunk:
                sim = float(chunk["embedding_score"])
            else:
                vec = np.array(chunk.get("vector", []), dtype=np.float32)
                if vec.size == 0:
                    sim = 0.0
                else:
                    sim = float(np.dot(q, vec) / ((np.linalg.norm(vec) or 1.0) * q_norm))
            dense_scored.append((idx, sim))
        t_dense = time.perf_counter() - t_dense_start

        # ── Bước 3: Reciprocal Rank Fusion (RRF) ──────────────────────────
        t_rrf_start = time.perf_counter()
        # Sắp xếp Dense để lấy hạng (rank)
        dense_scored.sort(key=lambda x: x[1], reverse=True)
        dense_ranks = {item[0]: rank for rank, item in enumerate(dense_scored, start=1)}

        # Sắp xếp BM25 để lấy hạng (rank)
        bm25_scored = list(enumerate(bm25_scores))
        bm25_scored.sort(key=lambda x: x[1], reverse=True)
        bm25_ranks = {item[0]: rank for rank, item in enumerate(bm25_scored, start=1)}

        # Tính điểm RRF kết hợp
        # RRF Score formula: RRF(d) = sum(1 / (k + rank_i(d)))
        # Mặc định hằng số k = 60 theo tiêu chuẩn của Elasticsearch & IR Research
        k = 60.0
        rrf_scored: list[dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            dense_rank = dense_ranks[idx]
            bm25_rank = bm25_ranks[idx]
            
            # Tính RRF Score
            rrf_score = (1.0 / (k + dense_rank)) + (1.0 / (k + bm25_rank))
            
            # Giữ lại các chỉ số điểm số thô để phân tích học thuật / XAI
            sim = next(item[1] for item in dense_scored if item[0] == idx)
            bm = bm25_scores[idx]
            
            rrf_scored.append({
                "chunk_id": chunk.get("id", chunk.get("chunk_id", "")),
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "page": chunk.get("page"),
                "text": chunk["text"],
                "embedding_score": round(sim, 6),
                "bm25_score": round(bm, 6),
                "combined_score": round(rrf_score * 100, 6),  # Nhân 100 để scale trực quan
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "metadata": chunk.get("metadata", {}),
            })

        # Sắp xếp các ứng viên theo RRF Score giảm dần
        rrf_scored.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Lấy pre-rerank top K để làm đầu vào cho Reranker
        pre_rerank = rrf_scored[:RETRIEVAL_PRE_RERANK_TOP_K]
        t_rrf = time.perf_counter() - t_rrf_start

        logger.debug(
            "🔍 Hybrid RRF retrieval complete: %d chunks → pre-rerank top %d",
            len(chunks), len(pre_rerank),
        )

        # ── Bước 4: Cross-Encoder Reranking ───────────────────────────────
        t_rerank_start = time.perf_counter()
        final_top_k = min(top_k, RETRIEVAL_FINAL_TOP_K)
        reranked = self._reranker.rerank(
            query=query,
            chunks=pre_rerank,
            top_k=final_top_k,
            threshold=threshold,
        )

        # Trình fallback nếu threshold rerank quá gắt gây rỗng
        if not reranked and pre_rerank:
            logger.warning(
                "⚠️ Reranker returned empty (threshold=%.2f too high). "
                "Relaxing threshold to 0.15 for at least 1 result.",
                threshold,
            )
            reranked = self._reranker.rerank(
                query=query,
                chunks=pre_rerank[:3],
                top_k=1,
                threshold=0.15,
            )
        t_rerank = time.perf_counter() - t_rerank_start

        # Lưu lại nhật ký thời gian đo đạc
        self.last_latency = {
            "bm25": round(t_bm25, 6),
            "vector_rrf": round(t_dense + t_rrf, 6),
            "rerank": round(t_rerank, 6)
        }

        logger.info(
            "✅ Retrieval hoàn tất: %d → %d chunks (Reranker=%s, bm25=%.4fs, vector_rrf=%.4fs, rerank=%.4fs)",
            len(pre_rerank),
            len(reranked),
            "CrossEncoder" if self._reranker.is_available() else "fallback",
            t_bm25,
            t_dense + t_rrf,
            t_rerank,
        )
        return reranked

    # ─────────────────────── BM25 Implementation ────────────────────────────

    def _bm25_scores(self, query: str, chunks: list[dict[str, Any]]) -> list[float]:
        """
        BM25 (Okapi BM25) với k1=1.5, b=0.75.
        Kết quả được normalize về [0, 1] để dễ kết hợp.
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
