"""
tests/test_adaptive_context_builder.py — Unit tests cho Adaptive Context Builder.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from backend.services.rag.adaptive_context_builder import (
    analyze_query_focus,
    attach_citations_to_summary,
    extract_facts_from_chunks,
    merge_fact_chunks,
    resolve_compression_tier,
    select_dynamic_chunks,
    build_adaptive_context,
)
from backend.services.rag.context_compression import CompressedContext, build_retrieved_context


def _chunk(cid: str, text: str, rerank: float) -> dict:
    return {
        "id": cid,
        "chunk_id": cid,
        "text": text,
        "filename": "test.docx",
        "page": 1,
        "document_id": "doc1",
        "rerank_score": rerank,
        "combined_score": rerank * 0.9,
    }


class TestAdaptiveContextBuilder(unittest.TestCase):
    def test_resolve_compression_tier(self):
        with mock.patch.dict(os.environ, {
            "RAG_LIGHT_COMPRESSION": "1500",
            "RAG_MEDIUM_COMPRESSION": "3000",
            "RAG_HEAVY_COMPRESSION": "6000",
        }):
            import importlib
            import backend.services.rag.rag_config as rc
            importlib.reload(rc)
            import backend.services.rag.adaptive_context_builder as acb
            importlib.reload(acb)

            self.assertEqual(acb.resolve_compression_tier(800), "none")
            self.assertEqual(acb.resolve_compression_tier(2000), "light")
            self.assertEqual(acb.resolve_compression_tier(4500), "medium")
            self.assertEqual(acb.resolve_compression_tier(8000), "aggressive")

    def test_select_dynamic_chunks_by_ratio(self):
        chunks = [
            _chunk("a", "low", 0.5),
            _chunk("b", "high", 0.95),
            _chunk("c", "mid", 0.88),
            _chunk("d", "low2", 0.4),
        ]
        with mock.patch.dict(os.environ, {
            "RAG_DYNAMIC_CHUNK_RATIO": "0.9",
            "RAG_MIN_RERANK_SCORE": "0.85",
        }):
            import importlib
            import backend.services.rag.rag_config as rc
            importlib.reload(rc)
            import backend.services.rag.adaptive_context_builder as acb
            importlib.reload(acb)

            selected = acb.select_dynamic_chunks(chunks)
            ids = [c["id"] for c in selected]
            self.assertIn("b", ids)
            self.assertIn("c", ids)
            self.assertNotIn("a", ids)

    def test_extract_facts_numbers_and_dates(self):
        chunks = [
            _chunk("1", "Doanh thu năm 2024 đạt 15,5 tỷ đồng, tăng 12%.", 0.9),
        ]
        facts = extract_facts_from_chunks(chunks)
        self.assertGreater(len(facts), 0)
        fact_texts = " ".join(f["text"] for f in facts)
        self.assertIn("2024", fact_texts)

    def test_merge_fact_chunks_adds_missing(self):
        selected = [_chunk("b", "text b", 0.95)]
        all_chunks = [
            _chunk("a", "Giá 100 USD ngày 01/01/2024", 0.5),
            _chunk("b", "text b", 0.95),
        ]
        facts = extract_facts_from_chunks(all_chunks)
        merged = merge_fact_chunks(selected, all_chunks, facts)
        ids = {c["id"] for c in merged}
        self.assertIn("a", ids)
        self.assertIn("b", ids)

    def test_attach_citations_to_summary(self):
        summary = "Doanh thu tăng mạnh. Kết quả thí nghiệm tốt."
        chunks = [
            _chunk("1", "Doanh thu tăng 20% trong quý.", 0.9),
            _chunk("2", "Kết quả thí nghiệm đạt F1=0.92.", 0.8),
        ]
        annotated, citations = attach_citations_to_summary(summary, chunks)
        self.assertIn("[chunk:", annotated)
        self.assertGreater(len(citations), 0)

    def test_analyze_query_focus_no_llm(self):
        with mock.patch.dict(os.environ, {"RAG_USE_LLM_INTENT": "0"}):
            intent, focus = analyze_query_focus("Kết quả thí nghiệm bao nhiêu phần trăm?")
            self.assertIn(intent, ("DOCUMENT_QA", "GENERAL", "SUMMARIZE"))
            self.assertTrue(len(focus) > 0)

    def test_build_adaptive_short_context_no_summary(self):
        with mock.patch.dict(os.environ, {
            "RAG_ADAPTIVE_CONTEXT": "1",
            "RAG_LIGHT_COMPRESSION": "1500",
            "RAG_DYNAMIC_CHUNK_RATIO": "0.9",
            "RAG_MIN_RERANK_SCORE": "0.5",
        }):
            import importlib
            import backend.services.rag.rag_config as rc
            importlib.reload(rc)
            import backend.services.rag.adaptive_context_builder as acb
            importlib.reload(acb)

            chunks = [_chunk("1", "Ngắn.", 0.9)]
            result = acb.build_adaptive_context(chunks, "Câu hỏi?")
            self.assertEqual(result.mode, "adaptive")
            self.assertEqual(result.compression_tier, "none")
            self.assertTrue(result.enabled)

    def test_build_adaptive_long_context_with_mock_summary(self):
        with mock.patch.dict(os.environ, {
            "RAG_ADAPTIVE_CONTEXT": "1",
            "RAG_LIGHT_COMPRESSION": "100",
            "RAG_MEDIUM_COMPRESSION": "500",
            "RAG_HEAVY_COMPRESSION": "2000",
            "RAG_DYNAMIC_CHUNK_RATIO": "0.8",
            "RAG_MIN_RERANK_SCORE": "0.5",
            "RAG_ADAPTIVE_CONTEXT_CACHE": "0",
        }):
            import importlib
            import backend.services.rag.rag_config as rc
            importlib.reload(rc)
            import backend.services.rag.adaptive_context_builder as acb
            importlib.reload(acb)

            chunks = [
                _chunk("1", "A" * 600, 0.95),
                _chunk("2", "B" * 600, 0.85),
                _chunk("3", "C" * 600, 0.75),
            ]
            with mock.patch.object(
                acb,
                "_generate_query_aware_summary",
                return_value=("Tóm tắt query-aware đủ dài cho kiểm thử adaptive builder.", "bartpho", "textrank-bartpho"),
            ):
                result = acb.build_adaptive_context(chunks, "Phương pháp nào được dùng?")
            self.assertEqual(result.mode, "adaptive")
            self.assertIn(result.compression_tier, ("medium", "aggressive", "light"))
            self.assertIn("Tóm tắt query-aware", result.hybrid_summary)

    def test_build_retrieved_context_routes_adaptive(self):
        with mock.patch.dict(os.environ, {
            "RAG_ADAPTIVE_CONTEXT": "1",
            "RAG_LIGHT_COMPRESSION": "1500",
            "RAG_ADAPTIVE_CONTEXT_CACHE": "0",
        }):
            import importlib
            import backend.services.rag.rag_config as rc
            importlib.reload(rc)
            import backend.services.rag.context_compression as cc
            importlib.reload(cc)

            chunks = [_chunk("1", "Test context ngắn.", 0.9)]
            result = cc.build_retrieved_context(chunks, "Test?")
            self.assertEqual(result.mode, "adaptive")

    def test_adaptive_prompt_template(self):
        from backend.services.rag.generator import GroundedGenerator

        ctx = CompressedContext(
            enabled=True,
            mode="adaptive",
            query_focus="số liệu và thống kê",
            hybrid_summary="Tóm tắt có trích dẫn [chunk:1, p.1].",
            top_original_chunks=[_chunk("1", "Đoạn gốc 100%.", 0.9)],
            all_retrieved=[_chunk("1", "Đoạn gốc 100%.", 0.9)],
        )
        prompt = GroundedGenerator().compose_prompt("Câu hỏi?", [], compressed_context=ctx)
        self.assertIn("QUERY FOCUS", prompt)
        self.assertIn("VERIFIED ORIGINAL PASSAGES", prompt)
        self.assertIn("RULES", prompt)
        self.assertIn("số liệu", prompt)


if __name__ == "__main__":
    unittest.main()
