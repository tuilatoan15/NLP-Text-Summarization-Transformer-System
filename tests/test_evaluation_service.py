"""
tests/test_evaluation_service.py — Unit tests for shared Compare/Summarize evaluation pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from backend.services.evaluation_service import (
    COMPARE_ALGORITHM_KEYS,
    COMPARE_METRIC_KEYS,
    EvaluationService,
    evaluate_compare_metrics,
    format_compare_metrics,
)
from src.evaluate import compute_rouge, compute_bertscore

REF = "Hội đồng Bảo an Liên Hợp Quốc họp khẩn về Trung Đông, kêu gọi ngừng bắn."
PRED = "Liên Hợp Quốc họp về Trung Đông và kêu gọi ngừng bắn ngay lập tức."
SRC = "Hôm nay Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn về tình hình Trung Đông."


class TestCompareAlgorithmKeys:
    def test_fifteen_algorithms(self):
        assert len(COMPARE_ALGORITHM_KEYS) == 15

    def test_includes_all_groups(self):
        extractive = {"textrank", "lexrank", "lsa"}
        abstractive = {"vit5", "mt5", "bartpho"}
        hybrid = {k for k in COMPARE_ALGORITHM_KEYS if "-" in k}
        assert extractive <= set(COMPARE_ALGORITHM_KEYS)
        assert abstractive <= set(COMPARE_ALGORITHM_KEYS)
        assert len(hybrid) == 9


class TestFormatCompareMetrics:
    def test_maps_full_metrics_to_compare_schema(self):
        full = {
            "rouge1": 0.5,
            "rouge2": 0.3,
            "rougeL": 0.4,
            "rougeLsum": 0.41,
            "bertscore": {"precision": 0.9, "recall": 0.85, "f1": 0.87},
            "bertscore_f1": 0.87,
            "processing_time": 1.234,
        }
        out = format_compare_metrics(full)
        assert set(COMPARE_METRIC_KEYS) <= set(out.keys())
        assert out["bert_p"] == 0.9
        assert out["bert_r"] == 0.85
        assert out["bertscore"] == 0.87
        assert isinstance(out["bertscore"], float)
        assert out["latency"] == 1.234

    def test_bertscore_status_propagates(self):
        full = {
            "rouge1": 0.0,
            "bertscore": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "bertscore_f1": 0.0,
            "bertscore_status": "timeout",
            "bertscore_error": "BERTScore timed out after 30s",
            "processing_time": 0.5,
        }
        out = format_compare_metrics(full)
        assert out["bertscore_status"] == "timeout"
        assert "BERTScore timed out" in (out.get("warning") or "")


class TestEvaluationService:
    def test_evaluate_returns_compare_keys(self):
        result = EvaluationService.evaluate(PRED, REF, SRC, 0.5, timeout=60.0)
        for key in COMPARE_METRIC_KEYS:
            assert key in result
            assert isinstance(result[key], (int, float))

    def test_evaluate_alias(self):
        result = evaluate_compare_metrics(PRED, REF, SRC, 0.25, timeout=60.0)
        assert result["rouge1"] >= 0.0
        assert result["bertscore"] >= 0.0

    def test_uses_same_rouge_as_metrics_module(self):
        expected = compute_rouge(PRED, REF)
        result = EvaluationService.evaluate(PRED, REF, SRC, 0.1, timeout=60.0)
        assert result["rouge1"] == expected["rouge1"]
        assert result["rougeLsum"] == expected["rougeLsum"]

    def test_bertscore_consistent_with_metrics_module(self):
        expected = compute_bertscore(PRED, REF)
        result = EvaluationService.evaluate(PRED, REF, SRC, 0.1, timeout=60.0)
        assert result["bert_p"] == expected["precision"]
        assert result["bert_r"] == expected["recall"]
        assert result["bertscore"] == expected["f1"]

    def test_no_reference_zeros_overlap_when_biased(self):
        result = EvaluationService.evaluate(PRED, SRC, SRC, 0.1, timeout=60.0)
        assert result.get("is_biased") is True
        assert result["rouge1"] == 0.0
