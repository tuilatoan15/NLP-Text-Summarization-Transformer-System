"""
tests/test_context_compression.py — Unit tests cho Hybrid Context Compression.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from backend.services.rag.context_compression import (
    CompressedContext,
    compress_retrieved_context,
    compute_total_context_chars,
    select_top_original_chunks,
    should_compress_context,
)


def _chunk(cid: str, text: str, rerank: float) -> dict:
    return {
        "id": cid,
        "chunk_id": cid,
        "text": text,
        "filename": "test.docx",
        "rerank_score": rerank,
        "combined_score": rerank * 0.9,
    }


class TestContextCompression(unittest.TestCase):
    def test_select_top_original_by_rerank(self):
        chunks = [
            _chunk("a", "low", 0.2),
            _chunk("b", "high", 0.95),
            _chunk("c", "mid", 0.6),
        ]
        top = select_top_original_chunks(chunks, 2)
        self.assertEqual([c["id"] for c in top], ["b", "c"])

    def test_should_compress_short_context_skips(self):
        with mock.patch.dict(os.environ, {
            "RAG_CONTEXT_COMPRESSION": "1",
            "RAG_SUMMARY_FOR_LONG_CONTEXT_ONLY": "1",
            "RAG_CONTEXT_COMPRESSION_THRESHOLD": "2500",
        }):
            import importlib
            import backend.services.rag.rag_config as rc
            importlib.reload(rc)
            from backend.services.rag import context_compression as cc_mod
            importlib.reload(cc_mod)
            self.assertFalse(cc_mod.should_compress_context(500))

    def test_compression_disabled_passthrough(self):
        with mock.patch.dict(os.environ, {"RAG_CONTEXT_COMPRESSION": "0"}):
            import importlib
            import backend.services.rag.rag_config as rc
            importlib.reload(rc)
            from backend.services.rag import context_compression as cc_mod
            importlib.reload(cc_mod)

            chunks = [_chunk("x", "x" * 100, 0.8)]
            result = cc_mod.compress_retrieved_context(chunks, "test query")
            self.assertFalse(result.enabled)
            self.assertEqual(result.skipped_reason, "disabled")

    def test_long_context_triggers_compression(self):
        with mock.patch.dict(os.environ, {
            "RAG_CONTEXT_COMPRESSION": "1",
            "RAG_SUMMARY_FOR_LONG_CONTEXT_ONLY": "1",
            "RAG_CONTEXT_COMPRESSION_THRESHOLD": "100",
            "RAG_TOP_ORIGINAL_CHUNKS": "2",
        }):
            import importlib
            import backend.services.rag.rag_config as rc
            importlib.reload(rc)
            from backend.services.rag import context_compression as cc_mod
            importlib.reload(cc_mod)

            chunks = [
                _chunk("1", "A" * 800, 0.9),
                _chunk("2", "B" * 800, 0.7),
                _chunk("3", "C" * 800, 0.5),
            ]
            with mock.patch.object(
                cc_mod,
                "_generate_hybrid_summary",
                return_value=("Tóm tắt hybrid kiểm thử với đủ độ dài từ.", "bartpho", "textrank-bartpho"),
            ):
                result = cc_mod.compress_retrieved_context(chunks, "Câu hỏi?")
            self.assertTrue(result.enabled)
            self.assertEqual(len(result.top_original_chunks), 2)
            self.assertIn("Tóm tắt hybrid", result.hybrid_summary)

    def test_generator_compose_compressed_prompt(self):
        from backend.services.rag.generator import GroundedGenerator

        compressed = CompressedContext(
            enabled=True,
            hybrid_summary="Tóm tắt tài liệu.",
            top_original_chunks=[_chunk("1", "Đoạn gốc quan trọng.", 0.9)],
            all_retrieved=[_chunk("1", "Đoạn gốc quan trọng.", 0.9)],
        )
        prompt = GroundedGenerator().compose_prompt(
            "Câu hỏi?",
            [],
            compressed_context=compressed,
        )
        self.assertIn("DOCUMENT SUMMARY", prompt)
        self.assertIn("IMPORTANT ORIGINAL PASSAGES", prompt)
        self.assertIn("Đoạn gốc quan trọng", prompt)
        self.assertIn("ưu tiên", prompt.lower())


if __name__ == "__main__":
    unittest.main()
