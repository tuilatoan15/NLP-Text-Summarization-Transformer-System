"""
selector.py — Chọn bản tóm tắt tốt nhất dựa trên điểm số tổng hợp.

Công thức điểm:
    Score = 0.4 * ROUGE-L + 0.3 * ROUGE-1 + 0.2 * ROUGE-2 + 0.1 * length_score

Trong đó length_score phạt các bản tóm tắt quá ngắn hoặc quá dài so với
khoảng lý tưởng được cấu hình (mặc định: 30 - 150 từ).
"""

from typing import Optional
from src.evaluate import compute_rouge
from src.utils import logger, count_words


# ==============================================================================
# CẤU HÌNH ĐIỂM
# ==============================================================================

# Trọng số điểm ROUGE trong công thức tổng hợp
WEIGHT_ROUGE_L = 0.4
WEIGHT_ROUGE_1 = 0.3
WEIGHT_ROUGE_2 = 0.2
WEIGHT_LENGTH  = 0.1

# Khoảng độ dài lý tưởng (số từ) của bản tóm tắt
IDEAL_MIN_WORDS = 30
IDEAL_MAX_WORDS = 150


# ==============================================================================
# TÍNH LENGTH SCORE
# ==============================================================================

def compute_length_score(
    summary: str,
    ideal_min: int = IDEAL_MIN_WORDS,
    ideal_max: int = IDEAL_MAX_WORDS,
) -> float:
    """
    Tính điểm dựa trên độ dài bản tóm tắt.

    - Nếu nằm trong khoảng [ideal_min, ideal_max]: điểm 1.0
    - Nếu quá ngắn (< ideal_min): phạt tuyến tính về 0 khi về 0 từ
    - Nếu quá dài  (> ideal_max): phạt tuyến tính, giảm dần khi dài thêm

    Args:
        summary:   Bản tóm tắt
        ideal_min: Số từ tối thiểu lý tưởng
        ideal_max: Số từ tối đa lý tưởng

    Returns:
        Điểm từ 0.0 đến 1.0
    """
    word_count = count_words(summary)

    if word_count == 0:
        return 0.0

    if word_count < ideal_min:
        # Phạt khi quá ngắn: điểm tỷ lệ thuận với số từ/ideal_min
        score = word_count / ideal_min
    elif word_count > ideal_max:
        # Phạt khi quá dài: giảm 5% cho mỗi 10 từ vượt quá
        excess = word_count - ideal_max
        penalty = excess / (ideal_max * 2)  # Penalty tăng dần
        score = max(0.0, 1.0 - penalty)
    else:
        score = 1.0

    logger.debug(f"Length score: {score:.3f} ({word_count} từ, range [{ideal_min}-{ideal_max}])")
    return round(score, 4)


# ==============================================================================
# TÍNH ĐIỂM TỔNG HỢP
# ==============================================================================

def compute_combined_score(rouge_scores: dict, length_score: float) -> float:
    """
    Tính điểm tổng hợp theo công thức có trọng số.

    Score = 0.4 * rougeL + 0.3 * rouge1 + 0.2 * rouge2 + 0.1 * length_score

    Args:
        rouge_scores: Dict chứa rouge1, rouge2, rougeL (F1 scores)
        length_score: Điểm độ dài (0.0 - 1.0)

    Returns:
        Điểm tổng hợp (0.0 - 1.0)
    """
    rouge1 = rouge_scores.get("rouge1", 0.0)
    rouge2 = rouge_scores.get("rouge2", 0.0)
    rougeL = rouge_scores.get("rougeL", 0.0)

    score = (
        WEIGHT_ROUGE_L * rougeL +
        WEIGHT_ROUGE_1 * rouge1 +
        WEIGHT_ROUGE_2 * rouge2 +
        WEIGHT_LENGTH  * length_score
    )

    return round(score, 4)


# ==============================================================================
# HÀM CHÍNH: CHỌN BẢN TÓM TẮT TỐT NHẤT
# ==============================================================================

def select_best_summary(
    extractive_summary: str,
    abstractive_summary: str,
    reference: str,
    ideal_min_words: int = IDEAL_MIN_WORDS,
    ideal_max_words: int = IDEAL_MAX_WORDS,
) -> dict:
    """
    So sánh và chọn bản tóm tắt tốt nhất giữa extractive và abstractive.

    Args:
        extractive_summary:  Bản tóm tắt trích xuất
        abstractive_summary: Bản tóm tắt diễn giải
        reference: Văn bản tham chiếu để tính ROUGE
                   (nếu không có reference thực, dùng văn bản gốc)
        ideal_min_words: Số từ tối thiểu lý tưởng
        ideal_max_words: Số từ tối đa lý tưởng

    Returns:
        Dict chứa:
          - best_summary: Bản tóm tắt được chọn
          - best_type: 'extractive' hoặc 'abstractive'
          - scores: Dict điểm chi tiết cho cả hai bản
    """
    logger.info("Đang đánh giá và chọn bản tóm tắt tốt nhất...")

    # --- Đánh giá bản trích xuất ---
    ext_rouge = compute_rouge(extractive_summary, reference)
    ext_length = compute_length_score(extractive_summary, ideal_min_words, ideal_max_words)
    ext_combined = compute_combined_score(ext_rouge, ext_length)

    # --- Đánh giá bản diễn giải ---
    abs_rouge = compute_rouge(abstractive_summary, reference)
    abs_length = compute_length_score(abstractive_summary, ideal_min_words, ideal_max_words)
    abs_combined = compute_combined_score(abs_rouge, abs_length)

    # --- Chọn bản tốt hơn ---
    if ext_combined >= abs_combined:
        best_type = "extractive"
        best_summary = extractive_summary
    else:
        best_type = "abstractive"
        best_summary = abstractive_summary

    logger.info(
        f"Kết quả: {best_type} thắng "
        f"(extractive={ext_combined:.4f} vs abstractive={abs_combined:.4f})"
    )

    result = {
        "best_summary": best_summary,
        "best_type": best_type,
        "scores": {
            "extractive": {
                **ext_rouge,
                "length_score": ext_length,
                "combined_score": ext_combined,
            },
            "abstractive": {
                **abs_rouge,
                "length_score": abs_length,
                "combined_score": abs_combined,
            },
        },
    }

    return result


# ==============================================================================
# CHẠY THỬ TRỰC TIẾP
# ==============================================================================

if __name__ == "__main__":
    reference_text = """
    Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp về tình hình leo thang căng thẳng
    ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay lập tức và mở hành lang nhân đạo.
    Cuộc khủng hoảng nhân đạo ngày càng nghiêm trọng với hàng nghìn thường dân phải di tản.
    """

    ext = "Hội đồng Bảo an họp khẩn về Trung Đông. Nhiều nước kêu gọi ngừng bắn và hành lang nhân đạo."
    abs_ = "Liên Hợp Quốc tổ chức cuộc họp khẩn để giải quyết khủng hoảng nhân đạo tại Trung Đông."

    result = select_best_summary(ext, abs_, reference_text)

    print(f"\n🏆 Bản tốt nhất: [{result['best_type'].upper()}]")
    print(f"   {result['best_summary']}")
    print("\n📊 Điểm chi tiết:")
    for stype, s in result["scores"].items():
        print(f"  {stype}:")
        for k, v in s.items():
            print(f"    {k}: {v:.4f}")
