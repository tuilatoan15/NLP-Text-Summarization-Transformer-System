"""
Tests for SummaryLengthManager.
"""

from __future__ import annotations

import pytest
from summarizers.length_manager import SummaryLengthManager


def test_analyze_input_short():
    text = " ".join(["từ"] * 100)
    analysis = SummaryLengthManager.analyze_input(text)
    assert analysis["word_count"] == 100
    assert analysis["suggested_mode"] == "short"
    assert not analysis["is_extremely_long"]


def test_analyze_input_standard():
    text = " ".join(["từ"] * 1000)
    analysis = SummaryLengthManager.analyze_input(text)
    assert analysis["word_count"] == 1000
    assert analysis["suggested_mode"] == "standard"
    assert not analysis["is_extremely_long"]


def test_analyze_input_detailed():
    text = " ".join(["từ"] * 4000)
    analysis = SummaryLengthManager.analyze_input(text)
    assert analysis["word_count"] == 4000
    assert analysis["suggested_mode"] == "detailed"
    assert not analysis["is_extremely_long"]


def test_analyze_input_extremely_long():
    text = " ".join(["từ"] * 11000)
    analysis = SummaryLengthManager.analyze_input(text)
    assert analysis["word_count"] == 11000
    assert analysis["is_extremely_long"]


def test_get_extractive_sentences():
    analysis = {"suggested_mode": "short"}
    assert SummaryLengthManager.get_extractive_sentences("auto", analysis) == 3
    assert SummaryLengthManager.get_extractive_sentences("short", analysis) == 3
    assert SummaryLengthManager.get_extractive_sentences("standard", analysis) == 5
    assert SummaryLengthManager.get_extractive_sentences("detailed", analysis) == 8


def test_get_abstractive_limits():
    analysis = {"suggested_mode": "standard"}
    assert SummaryLengthManager.get_abstractive_limits("vit5", "auto", analysis) == (60, 200)
    assert SummaryLengthManager.get_abstractive_limits("vit5", "short", analysis) == (30, 100)
    assert SummaryLengthManager.get_abstractive_limits("vit5", "detailed", analysis) == (120, 400)


def test_hierarchical_summarize_pipeline_extractive(monkeypatch):
    # Mock extractive summarizer
    mocked_calls = []

    def mock_summarize_extractive_algorithm(text, algorithm, sentence_count):
        mocked_calls.append((algorithm, sentence_count))
        return {"summary": "Tóm tắt trích xuất giả lập."}

    monkeypatch.setattr(
        "src.extractive.summarize_extractive_algorithm",
        mock_summarize_extractive_algorithm,
    )

    # 11,000 words input text
    text = " ".join(["Đây là câu thử nghiệm dài."] * 2200)
    summary = SummaryLengthManager.hierarchical_summarize_pipeline(
        text, "textrank", "auto", "extractive"
    )

    assert len(mocked_calls) > 1
    assert summary == "Tóm tắt trích xuất giả lập."
