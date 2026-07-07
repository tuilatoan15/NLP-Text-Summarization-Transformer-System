"""
tests/test_rag_faithfulness_multidoc.py — Faithfulness metrics & multi-doc retrieval.
"""
import unittest

from backend.services.rag.faithfulness import (
    compute_chat_faithfulness,
    compute_retrieval_confidence,
    is_comparison_query,
)
from backend.services.rag.retriever import HybridRetriever


def _chunk(doc_id: str, filename: str, text: str, score: float, rerank: float | None = None) -> dict:
    return {
        "id": f"{doc_id}_c",
        "chunk_id": f"{doc_id}_c",
        "document_id": doc_id,
        "filename": filename,
        "page": 1,
        "text": text,
        "combined_score": score,
        "rerank_score": rerank,
        "embedding_score": score,
        "bm25_score": score,
        "metadata": {},
    }


class TestFaithfulnessMetrics(unittest.TestCase):
    def test_retrieval_confidence_from_rerank(self):
        retrieved = [_chunk("d1", "a.txt", "nội dung", 0.5, rerank=0.87)]
        self.assertAlmostEqual(compute_retrieval_confidence(retrieved), 0.87, places=4)

    def test_retrieval_confidence_empty(self):
        self.assertEqual(compute_retrieval_confidence([]), 0.0)

    def test_faithfulness_heuristic_grounded_answer(self):
        answer = "Học máy là lĩnh vực trí tuệ nhân tạo."
        source = "Học máy là một lĩnh vực của trí tuệ nhân tạo cho phép máy tính học từ dữ liệu."
        chunks = [_chunk("d1", "a.txt", source, 0.9, rerank=0.9)]
        metrics = compute_chat_faithfulness(answer, source, chunks)
        self.assertIn("faithfulness", metrics)
        self.assertGreater(metrics["faithfulness"], 0.2)
        self.assertIn("hallucination_risk", metrics)

    def test_comparison_query_detection(self):
        self.assertTrue(is_comparison_query("So sánh hai tài liệu về doanh thu"))
        self.assertFalse(is_comparison_query("Doanh thu quý 1 là bao nhiêu?"))


class TestMultiDocRetrievalDiversity(unittest.TestCase):
    def test_apply_multi_doc_diversity_spreads_across_docs(self):
        chunks = [
            _chunk("doc_a", "a.pdf", "Nội dung A cao điểm", 0.99, rerank=0.99),
            {**_chunk("doc_a", "a.pdf", "Nội dung A phụ", 0.95, rerank=0.95), "id": "doc_a_c2", "chunk_id": "doc_a_c2"},
            {**_chunk("doc_a", "a.pdf", "Nội dung A thứ ba", 0.93, rerank=0.93), "id": "doc_a_c3", "chunk_id": "doc_a_c3"},
            _chunk("doc_b", "b.pdf", "Nội dung B", 0.80, rerank=0.80),
            _chunk("doc_c", "c.pdf", "Nội dung C", 0.75, rerank=0.75),
        ]
        result = HybridRetriever._apply_multi_doc_diversity(
            chunks, top_k=5, document_ids=["doc_a", "doc_b", "doc_c"],
        )
        doc_ids = {c["document_id"] for c in result}
        self.assertIn("doc_a", doc_ids)
        self.assertIn("doc_b", doc_ids)
        self.assertIn("doc_c", doc_ids)
        self.assertLessEqual(len(result), 5)

    def test_single_doc_unchanged_behavior(self):
        chunks = [
            _chunk("doc_a", "a.pdf", "Một", 0.9, rerank=0.9),
            {**_chunk("doc_a", "a.pdf", "Hai", 0.8, rerank=0.8), "id": "doc_a_c2", "chunk_id": "doc_a_c2"},
        ]
        result = HybridRetriever._apply_multi_doc_diversity(
            chunks, top_k=2, document_ids=["doc_a"],
        )
        self.assertEqual(len(result), 2)

    def test_ensure_per_document_coverage_backfills_missing_doc(self):
        """Reranker chỉ giữ doc_a, doc_b — doc_c được backfill từ RRF pool."""
        reranked = [
            _chunk("doc_a", "001_a.pdf", "Nội dung A", 0.99, rerank=0.99),
            {**_chunk("doc_a", "001_a.pdf", "A phụ", 0.95, rerank=0.95), "id": "doc_a_c2", "chunk_id": "doc_a_c2"},
            _chunk("doc_b", "003_b.pdf", "Nội dung B", 0.80, rerank=0.80),
        ]
        backfill_pool = reranked + [
            _chunk("doc_c", "002_c.pdf", "Nội dung C thấp điểm", 0.20, rerank=None),
        ]
        document_ids = ["doc_a", "doc_b", "doc_c"]
        result = HybridRetriever._ensure_per_document_coverage(
            reranked,
            document_ids=document_ids,
            backfill_pool=backfill_pool,
            top_k=5,
        )
        doc_ids = {c["document_id"] for c in result}
        self.assertEqual(doc_ids, set(document_ids))
        backfilled = [c for c in result if c.get("coverage_backfill")]
        self.assertEqual(len(backfilled), 1)
        self.assertEqual(backfilled[0]["document_id"], "doc_c")

    def test_three_docs_final_chunks_cover_all_document_ids(self):
        """Pipeline diversity + coverage: 3 docs đã chọn → 3 document_ids trong kết quả."""
        reranked = [
            _chunk("doc_a", "001_a.pdf", "A cao", 0.99, rerank=0.99),
            {**_chunk("doc_a", "001_a.pdf", "A phụ", 0.97, rerank=0.97), "id": "doc_a_c2", "chunk_id": "doc_a_c2"},
            {**_chunk("doc_a", "001_a.pdf", "A thứ ba", 0.95, rerank=0.95), "id": "doc_a_c3", "chunk_id": "doc_a_c3"},
            _chunk("doc_b", "003_b.pdf", "B", 0.80, rerank=0.80),
        ]
        document_ids = ["doc_a", "doc_b", "doc_c"]
        diverse = HybridRetriever._apply_multi_doc_diversity(
            reranked, top_k=5, document_ids=document_ids,
        )
        pool = diverse + [_chunk("doc_c", "002_c.pdf", "C từ RRF", 0.15, rerank=None)]
        result = HybridRetriever._ensure_per_document_coverage(
            diverse,
            document_ids=document_ids,
            backfill_pool=pool,
            top_k=5,
        )
        self.assertEqual(
            {c["document_id"] for c in result},
            {"doc_a", "doc_b", "doc_c"},
        )


if __name__ == "__main__":
    unittest.main()
