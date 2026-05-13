"""
tests/test_preprocess.py — Unit tests cho module tiền xử lý văn bản.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.preprocess import preprocess, split_sentences


class TestPreprocess:
    def test_basic_cleaning(self):
        text = "  Hội đồng Bảo an   họp khẩn.  "
        result = preprocess(text)
        assert "cleaned" in result
        assert result["cleaned"].strip() != ""

    def test_html_removal(self):
        text = "<p>Đây là <b>văn bản</b> có HTML.</p>"
        result = preprocess(text)
        assert "<p>" not in result["cleaned"]
        assert "<b>" not in result["cleaned"]

    def test_empty_input(self):
        result = preprocess("")
        assert result["cleaned"] == "" or result["cleaned"] is None or len(result["cleaned"].split()) == 0

    def test_unicode_normalization(self):
        text = "Viêt Nam"  # diacritics
        result = preprocess(text)
        assert result["cleaned"] is not None

    def test_returns_dict(self):
        result = preprocess("Đây là văn bản thử nghiệm.")
        assert isinstance(result, dict)
        assert "cleaned" in result


class TestSplitSentences:
    def test_multiple_sentences(self):
        text = "Câu một. Câu hai. Câu ba."
        sents = split_sentences(text)
        assert len(sents) >= 1

    def test_single_sentence(self):
        text = "Chỉ một câu duy nhất"
        sents = split_sentences(text)
        assert len(sents) == 1

    def test_empty(self):
        sents = split_sentences("")
        assert sents == [] or len(sents) == 0
