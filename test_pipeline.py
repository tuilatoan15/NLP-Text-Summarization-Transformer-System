"""
test_pipeline.py — Kiểm tra toàn bộ pipeline (không cần GPU/model ViT5).
Chạy: python test_pipeline.py
"""
import sys
import io
import warnings

# Set UTF-8 encoding cho console Windows (tránh lỗi encode tiếng Việt)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_preprocess():
    from src.preprocess import preprocess
    text = """
    <p>Hội nghị thượng đỉnh G7 năm nay diễn ra tại Hiroshima, Nhật Bản với sự tham dự
    của lãnh đạo 7 quốc gia phát triển hàng đầu thế giới.</p> Chủ đề chính của hội nghị
    xoay quanh các vấn đề an ninh toàn cầu, biến đổi khí hậu và phục hồi kinh tế
    sau đại dịch COVID-19. Thủ tướng Nhật Bản Fumio Kishida đã có bài phát biểu khai
    mạc ấn tượng, nhấn mạnh tầm quan trọng của hợp tác quốc tế. Các nhà lãnh đạo
    thảo luận về cuộc xung đột ở Ukraine và cam kết tiếp tục hỗ trợ Kyiv.
    Hội nghị cũng thông qua tuyên bố chung về an toàn hạt nhân và giải trừ quân bị.
    Đây được xem là một trong những hội nghị G7 quan trọng nhất trong thập kỷ qua.
    """
    result = preprocess(text, aggressive=True)
    assert result["cleaned"], "cleaned text is empty"
    assert len(result["sentences"]) > 0, "no sentences found"
    print(f"  [PASS] Preprocess: {len(result['cleaned'].split())} words, {len(result['sentences'])} sentences")
    return result["cleaned"]


def test_extractive(text):
    from src.extractive import extractive_summarize
    summary = extractive_summarize(text, sentence_count=3)
    assert summary and len(summary.strip()) > 0, "extractive summary is empty"
    words = len(summary.split())
    preview = summary[:80].encode('ascii', errors='replace').decode('ascii')
    print(f"  [PASS] Extractive: {words} words | preview: '{preview}...'")
    return summary


def test_rouge(prediction, reference):
    from src.evaluate import compute_rouge
    scores = compute_rouge(prediction, reference)
    assert "rouge1" in scores, "rouge1 missing"
    assert "rougeL" in scores, "rougeL missing"
    assert 0.0 <= scores["rouge1"] <= 1.0, "rouge1 out of range"
    print(f"  [PASS] ROUGE: rouge1={scores['rouge1']:.4f}, rouge2={scores['rouge2']:.4f}, rougeL={scores['rougeL']:.4f}")
    return scores


def test_selector(ext, reference):
    from src.selector import select_best_summary, compute_length_score

    # Test length score
    ls = compute_length_score("từ " * 50)  # 50 từ (trong khoảng lý tưởng)
    assert ls == 1.0, f"length_score should be 1.0 for 50 words, got {ls}"

    ls_short = compute_length_score("ngắn quá")  # 2 từ
    assert ls_short < 1.0, "short text should be penalized"

    fake_abs = "Hội nghị G7 tại Nhật Bản thảo luận về an ninh, khí hậu và kinh tế sau COVID-19."
    result = select_best_summary(ext, fake_abs, reference)

    assert "best_summary" in result, "best_summary missing"
    assert result["best_type"] in ("extractive", "abstractive"), "invalid best_type"
    assert "scores" in result, "scores missing"

    ext_score = result["scores"]["extractive"]["combined_score"]
    abs_score = result["scores"]["abstractive"]["combined_score"]
    print(f"  [PASS] Selector: winner={result['best_type'].upper()}, ext={ext_score:.4f}, abs={abs_score:.4f}")
    return result


def test_utils():
    from src.utils import count_words, count_sentences, truncate_text, format_scores
    assert count_words("xin chào thế giới") == 4
    assert count_words("") == 0
    assert count_sentences("Câu một. Câu hai! Câu ba?") == 3
    truncated = truncate_text("từ " * 1000, max_words=50)
    assert len(truncated.split()) == 50, "truncation failed"
    formatted = format_scores({"rouge1": 0.5, "rougeL": 0.4})
    assert "rouge1" in formatted
    print("  [PASS] Utils: count_words, count_sentences, truncate_text, format_scores")


if __name__ == "__main__":
    print("=" * 55)
    print("  INTEGRATION TEST — Vietnamese Summarization System")
    print("=" * 55)

    print("\n[1/4] Testing utils...")
    test_utils()

    print("\n[2/4] Testing preprocess...")
    clean = test_preprocess()

    print("\n[3/4] Testing extractive summarization...")
    ext_summary = test_extractive(clean)

    print("\n[3b/4] Testing ROUGE evaluation...")
    test_rouge(ext_summary, clean)

    print("\n[4/4] Testing selector...")
    test_selector(ext_summary, clean)

    print()
    print("=" * 55)
    print("  ALL TESTS PASSED! System is ready.")
    print("=" * 55)
    print()
    print("Next steps:")
    print("  1. Train model : python -m train.train_vit5 --max_samples 100 --epochs 1")
    print("  2. Run API     : uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
