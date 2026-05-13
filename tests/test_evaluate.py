"""
tests/test_evaluate.py — Unit tests cho module đánh giá ROUGE / BLEU / BERTScore.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.evaluate import compute_rouge, compute_rouge_batch, compute_bleu, compute_bertscore


REF  = "Hội đồng Bảo an Liên Hợp Quốc họp khẩn về Trung Đông, kêu gọi ngừng bắn."
PRED = "Liên Hợp Quốc họp về Trung Đông và kêu gọi ngừng bắn ngay lập tức."


class TestComputeRouge:
    def test_returns_four_keys(self):
        scores = compute_rouge(PRED, REF)
        assert {"rouge1", "rouge2", "rougeL", "rougeLsum"} <= set(scores.keys())

    def test_scores_in_range(self):
        scores = compute_rouge(PRED, REF)
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_identical_strings(self):
        scores = compute_rouge(REF, REF)
        assert scores["rouge1"] >= 0.99

    def test_empty_prediction(self):
        scores = compute_rouge("", REF)
        assert scores["rouge1"] == 0.0

    def test_empty_reference(self):
        scores = compute_rouge(PRED, "")
        assert scores["rouge1"] == 0.0


class TestComputeRougeBatch:
    def test_basic_batch(self):
        preds = [PRED, "Kết quả họp bàn quốc tế."]
        refs  = [REF,  "Quốc tế bàn về hòa bình."]
        scores = compute_rouge_batch(preds, refs)
        assert "rouge1" in scores

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            compute_rouge_batch(["a", "b"], ["c"])

    def test_empty_batch(self):
        scores = compute_rouge_batch([], [])
        assert scores["rouge1"] == 0.0


class TestComputeBleu:
    def test_returns_float(self):
        score = compute_bleu(PRED, REF)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_identical_strings(self):
        score = compute_bleu(REF, REF)
        assert score > 0.0

    def test_empty_returns_zero(self):
        score = compute_bleu("", REF)
        assert score == 0.0


class TestComputeBertscore:
    def test_returns_dict(self):
        result = compute_bertscore(PRED, REF)
        assert {"precision", "recall", "f1"} <= set(result.keys())

    def test_scores_in_range(self):
        result = compute_bertscore(PRED, REF)
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_similar_text_high_score(self):
        result = compute_bertscore(REF, REF)
        assert result["f1"] >= 0.9

    def test_empty_returns_zero(self):
        result = compute_bertscore("", REF)
        assert result["f1"] == 0.0
