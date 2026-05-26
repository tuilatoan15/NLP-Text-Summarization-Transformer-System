from __future__ import annotations

from src.extractive import summarize_extractive_algorithm


def test_tfidf_extractive_returns_summary() -> None:
    text = (
        "Nhu cầu điện tăng cao trong mùa nắng nóng. "
        "Thủy điện miền Bắc vận hành thận trọng. "
        "Cần theo dõi phụ tải giờ cao điểm."
    )
    result = summarize_extractive_algorithm(text, "tfidf", sentence_count=2)
    assert result["summary"]
    assert len(result["selected_sentences"]) <= 2
