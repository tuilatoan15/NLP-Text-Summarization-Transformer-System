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
    RAG_PARALLEL_RETRIEVAL,
    RAG_RRF_K,
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
        document_ids: list[str] | None = None,
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

        # ── Bước 1 & 2: BM25 + Dense (song song nếu bật) ───────────────────
        if RAG_PARALLEL_RETRIEVAL and len(chunks) >= 8:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag_ret") as pool:
                f_bm25 = pool.submit(self._bm25_scores, query, chunks)
                f_dense = pool.submit(self._dense_scores, query_vector, chunks)
                bm25_scores = f_bm25.result()
                dense_scored = f_dense.result()
            t_bm25 = self.last_latency.get("bm25", 0.0)
            t_dense = self.last_latency.get("dense", 0.0)
        else:
            t_bm25_start = time.perf_counter()
            bm25_scores = self._bm25_scores(query, chunks)
            t_bm25 = time.perf_counter() - t_bm25_start
            self.last_latency["bm25"] = round(t_bm25, 6)
            dense_scored = self._dense_scores(query_vector, chunks)
            t_dense = self.last_latency.get("dense", 0.0)

        # ── Bước 3: Weighted Reciprocal Rank Fusion (RRF) ───────────────────
        t_rrf_start = time.perf_counter()
        # Sắp xếp Dense để lấy hạng (rank)
        dense_scored.sort(key=lambda x: x[1], reverse=True)
        dense_ranks = {item[0]: rank for rank, item in enumerate(dense_scored, start=1)}
        dense_sim = {item[0]: item[1] for item in dense_scored}

        # Sắp xếp BM25 để lấy hạng (rank)
        bm25_scored = list(enumerate(bm25_scores))
        bm25_scored.sort(key=lambda x: x[1], reverse=True)
        bm25_ranks = {item[0]: rank for rank, item in enumerate(bm25_scored, start=1)}

        # Weighted RRF: vector 70% + BM25 30% (theo rag_config)
        k = RAG_RRF_K
        rrf_scored: list[dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            dense_rank = dense_ranks[idx]
            bm25_rank = bm25_ranks[idx]

            rrf_score = (VECTOR_WEIGHT / (k + dense_rank)) + (BM25_WEIGHT / (k + bm25_rank))
            
            # Giữ lại các chỉ số điểm số thô để phân tích học thuật / XAI
            sim = dense_sim[idx]
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
        effective_threshold = threshold
        if document_ids and len(document_ids) > 1:
            effective_threshold = min(threshold, 0.25)
        reranked = self._reranker.rerank(
            query=query,
            chunks=pre_rerank,
            top_k=final_top_k,
            threshold=effective_threshold,
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

        if document_ids and len(document_ids) > 1:
            if reranked:
                reranked = self._apply_multi_doc_diversity(
                    reranked, top_k=final_top_k, document_ids=document_ids,
                )
            reranked = self._ensure_per_document_coverage(
                reranked,
                document_ids=document_ids,
                backfill_pool=rrf_scored,
                top_k=final_top_k,
            )

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

    @staticmethod
    def _apply_multi_doc_diversity(
        chunks: list[dict[str, Any]],
        *,
        top_k: int,
        document_ids: list[str],
    ) -> list[dict[str, Any]]:
        """
        Đảm bảo top-k chunks đến từ nhiều tài liệu (max per doc + global fill).
        """
        n_docs = len(document_ids)
        top_k_per_doc = max(3, top_k // n_docs) if n_docs >= 3 else max(2, top_k // n_docs)

        by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            by_doc[str(chunk.get("document_id", ""))].append(chunk)

        selected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for doc_id in document_ids:
            for chunk in by_doc.get(doc_id, [])[:top_k_per_doc]:
                cid = str(chunk.get("chunk_id") or chunk.get("id") or "")
                if cid and cid not in seen_ids:
                    selected.append(chunk)
                    seen_ids.add(cid)

        for chunk in chunks:
            if len(selected) >= top_k:
                break
            cid = str(chunk.get("chunk_id") or chunk.get("id") or "")
            if cid and cid not in seen_ids:
                selected.append(chunk)
                seen_ids.add(cid)

        unique_docs = {str(c.get("document_id", "")) for c in selected}
        logger.debug(
            "Multi-doc diversity: %d docs → %d chunks from %d documents",
            n_docs, len(selected), len(unique_docs),
        )
        return selected[:top_k]

    @staticmethod
    def _ensure_per_document_coverage(
        chunks: list[dict[str, Any]],
        *,
        document_ids: list[str],
        backfill_pool: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Đảm bảo mỗi document_id đã chọn có đủ chunk trong kết quả.
        Backfill từ pool pre-rerank (RRF) khi reranker loại hết.
        Với ≥3 docs: backfill/giữ tối đa 2 chunk/doc.
        """
        if not document_ids or len(document_ids) <= 1:
            return chunks[:top_k]

        n_docs = len(document_ids)
        min_per_doc = 2 if n_docs >= 3 else 1

        def _chunk_key(chunk: dict[str, Any]) -> str:
            return str(chunk.get("chunk_id") or chunk.get("id") or "")

        def _score(chunk: dict[str, Any]) -> float:
            return float(chunk.get("rerank_score") or chunk.get("combined_score") or 0.0)

        selected = list(chunks)
        seen = {_chunk_key(c) for c in selected if _chunk_key(c)}

        by_doc_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in backfill_pool:
            by_doc_pool[str(chunk.get("document_id", ""))].append(chunk)
        for doc_id in by_doc_pool:
            by_doc_pool[doc_id].sort(key=_score, reverse=True)

        by_doc_selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in selected:
            by_doc_selected[str(chunk.get("document_id", ""))].append(chunk)

        for doc_id in document_ids:
            have = len(by_doc_selected.get(doc_id, []))
            need = max(0, min_per_doc - have)
            if need == 0:
                continue
            added = 0
            for candidate in by_doc_pool.get(doc_id, []):
                if added >= need:
                    break
                ck = _chunk_key(candidate)
                if ck and ck not in seen:
                    entry = dict(candidate)
                    entry["coverage_backfill"] = True
                    selected.append(entry)
                    by_doc_selected[doc_id].append(entry)
                    seen.add(ck)
                    added += 1
                    logger.info(
                        "📎 Backfill coverage: thêm chunk từ doc %s (score=%.3f)",
                        doc_id,
                        _score(entry),
                    )

        if len(selected) <= top_k:
            return selected

        by_doc_selected = defaultdict(list)
        for chunk in selected:
            by_doc_selected[str(chunk.get("document_id", ""))].append(chunk)

        keep_per_doc = 2 if n_docs >= 3 else 1
        final: list[dict[str, Any]] = []
        final_seen: set[str] = set()
        for doc_id in document_ids:
            doc_chunks = sorted(by_doc_selected.get(doc_id, []), key=_score, reverse=True)
            for chunk in doc_chunks[:keep_per_doc]:
                ck = _chunk_key(chunk)
                if ck and ck not in final_seen:
                    final.append(chunk)
                    final_seen.add(ck)

        for chunk in sorted(selected, key=_score, reverse=True):
            if len(final) >= top_k:
                break
            ck = _chunk_key(chunk)
            if ck and ck not in final_seen:
                final.append(chunk)
                final_seen.add(ck)

        return final[:top_k]

    def _dense_scores(
        self, query_vector: list[float], chunks: list[dict[str, Any]]
    ) -> list[tuple[int, float]]:
        import time

        t0 = time.perf_counter()
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
        self.last_latency["dense"] = round(time.perf_counter() - t0, 6)
        return dense_scored

    # ─────────────────────── BM25 Implementation ────────────────────────────

    def _bm25_scores(self, query: str, chunks: list[dict[str, Any]]) -> list[float]:
        """
        BM25 (Okapi BM25) với k1=1.5, b=0.75.
        Kết quả được normalize về [0, 1] để dễ kết hợp.
        """
        import time

        t0 = time.perf_counter()
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
        self.last_latency["bm25"] = round(time.perf_counter() - t0, 6)
        return scores
