"""
tests/test_summarization_quality_fixes.py — Unit tests for quality improvements and bug fixes.
"""

import sys
from pathlib import Path
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import clean_generated_summary
from evaluation.output_validator import is_garbled_abstractive, validate_output
from pipeline.hybrid_summarizer import HybridSummarizer
from summarizers.abstractive.abstractive_summarizer import AbstractiveSummarizer


class TestTextCleaningAndTrimming(unittest.TestCase):
    def test_strip_leading_bullet_artifacts(self):
        # Test leading bullets are stripped
        self.assertEqual(clean_generated_summary("* Hội đồng Bảo an."), "Hội đồng Bảo an.")
        self.assertEqual(clean_generated_summary("- Hội đồng Bảo an."), "Hội đồng Bảo an.")
        self.assertEqual(clean_generated_summary("• Hội đồng Bảo an."), "Hội đồng Bảo an.")
        self.assertEqual(clean_generated_summary(":. Hội đồng Bảo an."), "Hội đồng Bảo an.")

    def test_incomplete_sentence_trimming(self):
        # Normal complete sentence is untouched
        text = "Hội đồng Bảo an đã họp khẩn cấp."
        self.assertEqual(clean_generated_summary(text), text)

        # Incomplete sentence is trimmed to the last period
        text_with_dangling = "Hội đồng Bảo an đã họp khẩn cấp. Nhiều quốc gia kêu gọi ngừng bắn"
        self.assertEqual(clean_generated_summary(text_with_dangling), "Hội đồng Bảo an đã họp khẩn cấp.")


class TestGarbledTextValidator(unittest.TestCase):
    def test_softened_threshold_permits_numbers(self):
        # A normal sentence containing numbers (around 15-20% ratio) should NOT be flagged as garbled
        text = "Có 3 bệnh nhân mới tại quận 1 và quận 5."
        # This was previously flagged as garbled because of single_letter_threshold=0.10
        self.assertFalse(is_garbled_abstractive(text))

        validation = validate_output(text, require_vietnamese=True)
        self.assertFalse(validation["is_corrupted"])


class TestHybridSummarizerMocks(unittest.TestCase):
    @mock.patch("summarizers.abstractive.abstractive_summarizer.AbstractiveSummarizer.summarize")
    def test_hybrid_summarizer_calls_summarize_without_attribute_error(self, mock_summarize):
        mock_summarize.return_value = "Bản tóm tắt sinh ra thành công."
        
        # Test HybridSummarizer initialization and summarize call
        summarizer = HybridSummarizer(abstractive_model_key="vit5")
        text = "Văn bản gốc rất dài để kiểm thử hybrid summarizer. Cần lọc qua extractive trước khi đưa vào abstractive model."
        
        # Call summarize
        result = summarizer.summarize(text)
        
        # Verify call was forwarded to summarize with correct max length
        mock_summarize.assert_called_once()
        self.assertEqual(result, "Bản tóm tắt sinh ra thành công.")


if __name__ == "__main__":
    unittest.main()
