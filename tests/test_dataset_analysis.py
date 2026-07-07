"""Unit tests for dataset analysis module."""

from __future__ import annotations

from backend.services.dataset_analysis.cleaner import clean_article, clean_summary, tokenize_words
from backend.services.dataset_analysis.loader import DatasetRecord, LoadedDataset, resolve_limit
from backend.services.dataset_analysis.statistics import compute_all_statistics


FIXTURE_RECORDS = [
    DatasetRecord(
        article="<p>Bài viết thứ nhất về kinh tế Việt Nam năm 2024 với nhiều thông tin quan trọng.</p>",
        abstract="Kinh tế Việt Nam tăng trưởng mạnh.",
        title="Kinh tế 2024",
        split="train",
    ),
    DatasetRecord(
        article="Công nghệ AI đang phát triển nhanh chóng tại các trường đại học và doanh nghiệp công nghệ.",
        abstract="AI phát triển mạnh tại Việt Nam.",
        title="Công nghệ AI",
        split="train",
    ),
    DatasetRecord(
        article="Thể thao: đội tuyển bóng đá giành chiến thắng trong trận đấu quan trọng tối qua.",
        abstract="Đội tuyển thắng trận quan trọng.",
        title="Bóng đá",
        split="validation",
    ),
]


def test_resolve_limit_zero_means_full():
    assert resolve_limit(0) is None
    assert resolve_limit(None) is None
    assert resolve_limit(-1) is None
    assert resolve_limit(500) == 500
    assert resolve_limit(500, full=True) is None


def test_clean_article_strips_html():
    raw = "<b>Xin chào</b> &amp; thế giới"
    cleaned = clean_article(raw)
    assert "<" not in cleaned
    assert "Xin chào" in cleaned


def test_clean_summary_preserves_content():
    text = "Tóm tắt ngắn gọn về sự kiện."
    assert len(clean_summary(text)) > 5


def test_tokenize_vietnamese():
    tokens = tokenize_words("Việt Nam có 64 tỉnh thành")
    assert "việt" in tokens or "nam" in tokens


def test_compute_statistics_on_fixture():
    loaded = LoadedDataset(
        dataset_name="test/fixture",
        records=FIXTURE_RECORDS,
        splits={"train": 2, "validation": 1},
        columns=["article", "abstract", "title"],
        source="test",
        limit_per_split=None,
    )
    stats = compute_all_statistics(loaded, rouge_sample_size=3, ngram_sample_size=3)

    assert stats["overview"]["total_documents"] == 3
    assert stats["overview"]["full_dataset"] is True
    assert stats["document_stats"]["words"]["count"] == 3
    assert stats["document_stats"]["words"]["p99"] is not None
    assert stats["summary_stats"]["words"]["count"] == 3
    assert stats["vocabulary"]["unique_words"] >= 3
    assert "compression" in stats["summary_stats"]
    assert stats["quality"]["valid_pairs"] == 3
    assert "extractive_metrics" in stats
    assert "lead_words_proportional" in stats["rouge_baseline"]


def test_compression_ratio_bounded():
    loaded = LoadedDataset(dataset_name="test", records=FIXTURE_RECORDS, splits={"train": 3})
    stats = compute_all_statistics(loaded, rouge_sample_size=2, ngram_sample_size=3)
    ratio = stats["overview"]["avg_compression_ratio"]
    assert 0 < ratio < 1
