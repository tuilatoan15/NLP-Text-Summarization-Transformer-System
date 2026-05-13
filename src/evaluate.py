"""
evaluate.py — Đánh giá chất lượng tóm tắt bằng chỉ số ROUGE.

Dùng thư viện `evaluate` của Hugging Face để tính:
  - ROUGE-1: Precision/Recall/F1 ở cấp unigram
  - ROUGE-2: Precision/Recall/F1 ở cấp bigram
  - ROUGE-L: F1 dựa trên Longest Common Subsequence

Hỗ trợ đánh giá đơn lẻ (1 cặp) hoặc hàng loạt (batch).
"""

from typing import Optional
import evaluate as hf_evaluate

from src.utils import logger
from src import config


# ==============================================================================
# LOAD ROUGE METRIC (Singleton để tránh load nhiều lần)
# ==============================================================================

_rouge_metric = None


def _get_rouge():
    """Lazy load ROUGE metric từ Hugging Face evaluate."""
    global _rouge_metric
    if _rouge_metric is None:
        logger.info("Đang load ROUGE metric...")
        _rouge_metric = hf_evaluate.load("rouge")
        logger.info("✅ ROUGE metric đã sẵn sàng.")
    return _rouge_metric


# ==============================================================================
# ĐÁNH GIÁ ĐƠN LẺ
# ==============================================================================

def compute_rouge(
    prediction: str,
    reference: str,
    use_stemmer: bool = False,
) -> dict[str, float]:
    """
    Tính điểm ROUGE giữa một bản tóm tắt và văn bản tham chiếu.

    Args:
        prediction: Bản tóm tắt cần đánh giá
        reference:  Văn bản tham chiếu (reference summary hoặc văn bản gốc)
        use_stemmer: Dùng stemmer (Tiếng Việt không có stemmer chuẩn, tắt mặc định)

    Returns:
        Dict chứa:
          - rouge1: F1 score ROUGE-1
          - rouge2: F1 score ROUGE-2
          - rougeL: F1 score ROUGE-L
          - rougeLsum: F1 score ROUGE-Lsum (tính trên từng câu)
    """
    if not prediction or not prediction.strip():
        logger.warning("Prediction rỗng, trả về ROUGE = 0.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    if not reference or not reference.strip():
        logger.warning("Reference rỗng, trả về ROUGE = 0.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    rouge = _get_rouge()

    results = rouge.compute(
        predictions=[prediction],
        references=[reference],
        use_stemmer=use_stemmer,
        use_aggregator=True,
    )

    # Làm tròn 4 chữ số thập phân
    scores = {k: round(float(v), 4) for k, v in results.items()}
    logger.debug(f"ROUGE scores: {scores}")
    return scores


def compute_bleu(prediction: str, reference: str) -> float:
    """
    Compute BLEU score (corpus-level style for a single pair using HF evaluate wrapper).
    Returns a float BLEU score (0-100 scale) as returned by the metric.
    """
    try:
        bleu = hf_evaluate.load("bleu")
        results = bleu.compute(predictions=[prediction], references=[[reference]])
        # sacrebleu returns 'bleu' key
        score = float(results.get("bleu", 0.0))
        return round(score, 4)
    except Exception as e:
        logger.warning(f"Không thể tính BLEU: {e}")
        return 0.0


def compute_bertscore(
    prediction: str,
    reference: str,
    lang: str = config.BERTSCORE_LANG,
    model_type: str = config.BERTSCORE_MODEL,
) -> dict:
    """
    Tính BERTScore — metric hiện đại đánh giá chất lượng tóm tắt dựa trên
    contextual embeddings của BERT thay vì n-gram matching như ROUGE.

    BERTScore đã được chứng minh tương quan tốt hơn với đánh giá của con người
    so với ROUGE (Zhang et al., 2020).

    Args:
        prediction: Bản tóm tắt cần đánh giá
        reference:  Văn bản tham chiếu
        lang:       Ngôn ngữ ("vi" cho tiếng Việt)
        model_type: Model BERT dùng để tính (mặc định: multilingual)

    Returns:
        Dict chứa precision, recall, f1 (trung bình trên token level)
    """
    if not prediction or not prediction.strip():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not reference or not reference.strip():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        bertscore_metric = hf_evaluate.load("bertscore")
        results = bertscore_metric.compute(
            predictions=[prediction],
            references=[reference],
            lang=lang,
            model_type=model_type,
        )
        return {
            "precision": round(float(results["precision"][0]), 4),
            "recall":    round(float(results["recall"][0]), 4),
            "f1":        round(float(results["f1"][0]), 4),
        }
    except Exception as e:
        logger.warning(f"Không thể tính BERTScore: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def compute_semantic_similarity(prediction: str, reference: str, model_name: str = config.SBERT_MODEL) -> float:
    """
    Tính similarity cosine giữa embedding của prediction và reference.
    Trả về giá trị [-1, 1] (chuẩn hóa sang 0..1 nếu cần).
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer(model_name)
        emb1 = model.encode(prediction, convert_to_tensor=True)
        emb2 = model.encode(reference, convert_to_tensor=True)
        sim = util.cos_sim(emb1, emb2).item()
        # Giá trị cosine similarity trong [-1,1], chuyển về [0,1]
        sim01 = (sim + 1.0) / 2.0
        return round(float(sim01), 4)
    except Exception as e:
        logger.warning(f"Không thể tính semantic similarity: {e}")
        return 0.0


# ==============================================================================
# ĐÁNH GIÁ HÀNG LOẠT (BATCH)
# ==============================================================================

def compute_rouge_batch(
    predictions: list[str],
    references: list[str],
    use_stemmer: bool = False,
) -> dict[str, float]:
    """
    Tính điểm ROUGE trung bình cho một tập dữ liệu lớn.

    Args:
        predictions: Danh sách bản tóm tắt sinh ra
        references:  Danh sách văn bản tham chiếu tương ứng
        use_stemmer: Dùng stemmer hay không

    Returns:
        Dict điểm ROUGE trung bình trên toàn bộ batch
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Số lượng predictions ({len(predictions)}) "
            f"≠ references ({len(references)})"
        )

    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    # Lọc các cặp rỗng
    valid_pairs = [
        (p, r) for p, r in zip(predictions, references)
        if p and p.strip() and r and r.strip()
    ]

    if not valid_pairs:
        logger.warning("Không có cặp hợp lệ để tính ROUGE.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    valid_preds, valid_refs = zip(*valid_pairs)
    rouge = _get_rouge()

    results = rouge.compute(
        predictions=list(valid_preds),
        references=list(valid_refs),
        use_stemmer=use_stemmer,
        use_aggregator=True,
    )

    scores = {k: round(float(v), 4) for k, v in results.items()}
    logger.info(f"Batch ROUGE ({len(valid_pairs)} mẫu): {scores}")
    return scores


# ==============================================================================
# ĐÁNH GIÁ SO SÁNH (Extractive vs Abstractive)
# ==============================================================================

def evaluate_both(
    extractive_summary: str,
    abstractive_summary: str,
    reference: str,
) -> dict:
    """
    Đánh giá cả hai bản tóm tắt (extractive và abstractive) so với reference.

    Args:
        extractive_summary:  Bản tóm tắt trích xuất
        abstractive_summary: Bản tóm tắt diễn giải
        reference: Văn bản tham chiếu

    Returns:
        Dict chứa điểm ROUGE cho cả hai bản tóm tắt:
        {
            "extractive": {"rouge1": ..., "rouge2": ..., "rougeL": ...},
            "abstractive": {"rouge1": ..., "rouge2": ..., "rougeL": ...}
        }
    """
    logger.info("Đang đánh giá bản tóm tắt trích xuất...")
    ext_scores = compute_rouge(extractive_summary, reference)

    logger.info("Đang đánh giá bản tóm tắt diễn giải...")
    abs_scores = compute_rouge(abstractive_summary, reference)

    return {
        "extractive": ext_scores,
        "abstractive": abs_scores,
    }


# ==============================================================================
# CHẠY THỬ TRỰC TIẾP
# ==============================================================================

if __name__ == "__main__":
    reference = (
        "Hội đồng Bảo an Liên Hợp Quốc họp khẩn về tình hình Trung Đông, "
        "kêu gọi ngừng bắn và mở hành lang nhân đạo."
    )
    prediction = (
        "Liên Hợp Quốc họp khẩn về Trung Đông, nhiều nước kêu gọi ngừng bắn "
        "và hỗ trợ nhân đạo cho người dân vùng chiến sự."
    )

    scores = compute_rouge(prediction, reference)
    print("=== ROUGE SCORES ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
