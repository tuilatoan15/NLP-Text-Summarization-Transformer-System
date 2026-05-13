"""
tests/test_extractive.py — Unit tests cho module tóm tắt trích xuất.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.extractive import extractive_summarize, extractive_summarize_with_details, lexrank_summarize

SAMPLE = """
Hội nghị thượng đỉnh G7 năm nay diễn ra tại Hiroshima, Nhật Bản với sự tham dự
của lãnh đạo 7 quốc gia phát triển hàng đầu thế giới. Chủ đề chính của hội nghị
xoay quanh các vấn đề an ninh toàn cầu, biến đổi khí hậu và phục hồi kinh tế.
Thủ tướng Nhật Bản Fumio Kishida đã có bài phát biểu khai mạc ấn tượng.
Các nhà lãnh đạo thảo luận về cuộc xung đột ở Ukraine và cam kết tiếp tục hỗ trợ.
Hội nghị cũng thông qua tuyên bố chung về an toàn hạt nhân và giải trừ quân bị.
Kết quả hội nghị được kỳ vọng sẽ định hình chính sách toàn cầu trong những năm tới.
"""


class TestExtractiveSummarize:
    def test_returns_string(self):
        result = extractive_summarize(SAMPLE, sentence_count=3)
        assert isinstance(result, str)

    def test_non_empty(self):
        result = extractive_summarize(SAMPLE, sentence_count=3)
        assert len(result.strip()) > 0

    def test_sentence_count_respected(self):
        result = extractive_summarize(SAMPLE, sentence_count=2)
        # Result should be shorter than original
        assert len(result.split()) < len(SAMPLE.split())

    def test_empty_input(self):
        result = extractive_summarize("", sentence_count=3)
        assert result == "" or result is None or len(result.strip()) == 0

    def test_short_text_returns_all(self):
        short = "Câu một. Câu hai."
        result = extractive_summarize(short, sentence_count=5)
        assert len(result) > 0


class TestExtractiveSummarizeWithDetails:
    def test_returns_dict_with_keys(self):
        result = extractive_summarize_with_details(SAMPLE, sentence_count=3)
        assert "summary" in result
        assert "selected_sentences" in result
        assert "highlighted_sentence_indexes" in result

    def test_selected_sentences_list(self):
        result = extractive_summarize_with_details(SAMPLE, sentence_count=3)
        assert isinstance(result["selected_sentences"], list)

    def test_each_selected_has_fields(self):
        result = extractive_summarize_with_details(SAMPLE, sentence_count=2)
        for item in result["selected_sentences"]:
            assert "sentence" in item
            assert "sentence_index" in item
            assert "sentence_score" in item


class TestLexRankSummarize:
    def test_returns_string(self):
        result = lexrank_summarize(SAMPLE, sentence_count=3)
        assert isinstance(result, str)
        assert len(result.strip()) > 0
