"""
extractive.py — Tóm tắt trích xuất (Extractive Summarization) dùng TextRank.

Thuật toán TextRank (từ thư viện sumy) xây dựng đồ thị câu-câu và chọn ra
những câu quan trọng nhất dựa trên độ tương đồng với các câu khác.
Không cần GPU, chạy nhanh, phù hợp với máy yếu.
"""

import re
from typing import Optional

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from src.utils import logger
from src.preprocess import split_sentences


def _get_tokenizer(language: str) -> Tokenizer:
    """
    Tạo Tokenizer cho sumy.
    sumy/NLTK không có punkt_tab cho tiếng Việt nên fallback sang 'english'.
    Điều này không ảnh hưởng chất lượng vì TextRank chỉ cần tách câu đơn giản.
    """
    try:
        tok = Tokenizer(language)
        # Thử gọi để kích hoạt lỗi ngay nếu language không hỗ trợ
        _ = list(tok.to_sentences("Thử nghiệm."))
        return tok
    except Exception:
        logger.warning(
            f"Tokenizer cho '{language}' không khả dụng, dùng 'english' fallback."
        )
        return Tokenizer("english")


# ==============================================================================
# CẤU HÌNH
# ==============================================================================

# sumy dùng tên ngôn ngữ theo chuẩn NLTK/ISO
LANGUAGE = "vietnamese"

# Số câu mặc định trong bản tóm tắt trích xuất
DEFAULT_SENTENCE_COUNT = 5


# ==============================================================================
# TÓM TẮT TRÍCH XUẤT
# ==============================================================================

def extractive_summarize(
    text: str,
    sentence_count: int = DEFAULT_SENTENCE_COUNT,
    language: str = LANGUAGE,
    remove_duplicates: bool = True,
) -> str:
    """
    Thực hiện tóm tắt trích xuất bằng thuật toán TextRank.

    Quy trình:
      1. Parse văn bản thành cấu trúc câu
      2. Tính điểm tầm quan trọng mỗi câu (TextRank)
      3. Chọn `sentence_count` câu quan trọng nhất
      4. Ghép lại theo thứ tự xuất hiện trong văn bản gốc
      5. Loại bỏ câu trùng lặp (nếu bật)

    Args:
        text: Văn bản tiếng Việt đã được tiền xử lý
        sentence_count: Số câu mong muốn trong bản tóm tắt
        language: Ngôn ngữ để chọn stop words (mặc định: 'vietnamese')
        remove_duplicates: Loại bỏ câu có nội dung tương tự nhau

    Returns:
        Bản tóm tắt trích xuất dạng chuỗi
    """
    if not text or not text.strip():
        logger.warning("Văn bản đầu vào rỗng, không thể tóm tắt.")
        return ""

    # Kiểm tra văn bản có đủ câu không
    sentences_raw = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences_raw) <= sentence_count:
        logger.info(
            f"Văn bản chỉ có {len(sentences_raw)} câu, trả về toàn bộ."
        )
        return text.strip()

    try:
        # --- Parse văn bản ---
        tokenizer = _get_tokenizer(language)
        parser = PlaintextParser.from_string(text, tokenizer)

        # --- Cấu hình TextRank ---
        # Dùng stemmer 'english' vì tiếng Việt không có stemmer chuẩn trong sumy
        try:
            stemmer = Stemmer(language)
        except Exception:
            stemmer = Stemmer("english")
        summarizer = TextRankSummarizer(stemmer)

        # Cấu hình stop words
        try:
            summarizer.stop_words = get_stop_words(language)
        except LookupError:
            logger.warning("Không tìm thấy stop words tiếng Việt cho sumy, bỏ qua.")

        # --- Chạy tóm tắt ---
        summary_sentences = summarizer(parser.document, sentence_count)
        sentences = [str(s) for s in summary_sentences]

        if not sentences:
            logger.warning("TextRank không trả về câu nào, dùng fallback.")
            return _fallback_extractive(text, sentence_count)

        # --- Loại bỏ câu trùng lặp ---
        if remove_duplicates:
            sentences = _remove_duplicate_sentences(sentences)

        summary = " ".join(sentences)
        logger.info(f"Trích xuất xong: {len(sentences)} câu, {len(summary.split())} từ.")
        return summary

    except Exception as e:
        logger.error(f"Lỗi TextRank: {e}. Dùng fallback.")
        return _fallback_extractive(text, sentence_count)


def extractive_summarize_with_details(
    text: str,
    sentence_count: int = DEFAULT_SENTENCE_COUNT,
    language: str = LANGUAGE,
) -> dict:
    """Return TextRank summary with selected sentence indexes and scores.

    Use embedding-based TextRank scoring when possible for more interpretable
    sentence_score values. Falls back to centroid-based scoring if embedding
    model is unavailable.
    """
    summary = extractive_summarize(text, sentence_count=sentence_count, language=language)
    source_sentences = split_sentences(text)
    selected_sentences = split_sentences(summary)
    # Try embedding-based TextRank scoring; fallback to centroid
    ranked = _textrank_scores(source_sentences) if source_sentences else {}
    selected = []
    used_indexes = set()

    for selected_sentence in selected_sentences:
        index, source_sentence, similarity = _best_sentence_match(
            selected_sentence,
            source_sentences,
            used_indexes,
        )
        if index >= 0:
            used_indexes.add(index)
        selected.append({
            "sentence": source_sentence or selected_sentence,
            "sentence_index": index,
            "sentence_score": round(ranked.get(index, 0.0), 4),
            "match_similarity": round(similarity, 4),
        })

    return {
        "summary": summary,
        "selected_sentences": selected,
        "highlighted_sentence_indexes": [
            item["sentence_index"] for item in selected if item["sentence_index"] >= 0
        ],
    }


def _textrank_scores(sentences: list[str], model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> dict[int, float]:
    """
    Compute TextRank-like sentence importance scores using sentence embeddings.

    Steps:
      - Encode sentences to dense vectors (SentenceTransformer)
      - Compute cosine similarity matrix (embedding vectors are L2-normalized)
      - Remove self-similarity and run PageRank via power iteration
      - Normalize scores to [0, 1] by dividing by max

    Falls back to _rank_by_centroid if sentence-transformers is unavailable.
    """
    if not sentences:
        return {}
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        logger.warning(f"Không thể tải sentence-transformers cho TextRank scores: {e}. Dùng centroid fallback.")
        return _rank_by_centroid(sentences)

    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(sentences, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        sim = embeddings @ embeddings.T
        # remove self similarity
        np.fill_diagonal(sim, 0.0)

        # Build column-stochastic matrix for PageRank
        A = np.where(sim < 0, 0.0, sim)  # ensure non-negative
        col_sum = A.sum(axis=0)
        # avoid division by zero
        col_sum[col_sum == 0] = 1.0
        M = A / col_sum

        # power iteration
        n = A.shape[0]
        p = np.ones(n) / n
        d = 0.85
        tol = 1e-6
        for _ in range(1000):
            p_new = d * (M @ p) + (1.0 - d) / n
            if np.linalg.norm(p_new - p, ord=1) < tol:
                p = p_new
                break
            p = p_new

        scores = {i: float(p[i]) for i in range(n)}
        max_score = max(scores.values()) if scores else 1.0
        if max_score:
            scores = {i: scores[i] / max_score for i in scores}
        return scores
    except Exception as e:
        logger.warning(f"Lỗi khi tính TextRank scores: {e}. Dùng centroid fallback.")
        return _rank_by_centroid(sentences)


def _remove_duplicate_sentences(sentences: list[str], threshold: float = 0.8) -> list[str]:
    """
    Loại bỏ các câu quá giống nhau (trùng lặp thông tin).

    Tính Jaccard similarity giữa các câu, nếu >= threshold thì loại câu sau.

    Args:
        sentences: Danh sách câu
        threshold: Ngưỡng tương đồng (0.0 - 1.0)

    Returns:
        Danh sách câu đã lọc trùng lặp
    """
    if not sentences:
        return sentences

    unique = [sentences[0]]

    for candidate in sentences[1:]:
        is_duplicate = False
        candidate_words = set(candidate.lower().split())

        for existing in unique:
            existing_words = set(existing.lower().split())
            if not candidate_words or not existing_words:
                continue

            # Tính Jaccard similarity
            intersection = candidate_words & existing_words
            union = candidate_words | existing_words
            similarity = len(intersection) / len(union)

            if similarity >= threshold:
                is_duplicate = True
                logger.debug(f"Loại câu trùng lặp (similarity={similarity:.2f}): {candidate[:50]}...")
                break

        if not is_duplicate:
            unique.append(candidate)

    return unique


def _fallback_extractive(text: str, sentence_count: int) -> str:
    """
    Fallback đơn giản: lấy N câu đầu tiên từ văn bản.
    Dùng khi TextRank thất bại.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    selected = sentences[:sentence_count]
    return " ".join(selected)


def _rank_by_centroid(sentences: list[str]) -> dict[int, float]:
    if not sentences:
        return {}
    token_sets = [_token_set(sentence) for sentence in sentences]
    centroid = set().union(*token_sets) if token_sets else set()
    scores = {}
    for index, tokens in enumerate(token_sets):
        scores[index] = _jaccard(tokens, centroid)
    max_score = max(scores.values()) if scores else 1.0
    if max_score:
        scores = {index: score / max_score for index, score in scores.items()}
    return scores


def _best_sentence_match(
    sentence: str,
    candidates: list[str],
    used_indexes: set[int],
) -> tuple[int, str, float]:
    sentence_tokens = _token_set(sentence)
    best = (-1, "", 0.0)
    for index, candidate in enumerate(candidates):
        if index in used_indexes:
            continue
        candidate_tokens = _token_set(candidate)
        similarity = _jaccard(sentence_tokens, candidate_tokens)
        if similarity > best[2]:
            best = (index, candidate, similarity)
    return best


def _token_set(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"\w+", text, flags=re.UNICODE)
        if len(token) > 1
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


# ==============================================================================
# CHẠY THỬ TRỰC TIẾP
# ==============================================================================

if __name__ == "__main__":
    sample_text = """
    Hội nghị thượng đỉnh G7 năm nay diễn ra tại Hiroshima, Nhật Bản với sự tham dự
    của lãnh đạo 7 quốc gia phát triển hàng đầu thế giới. Chủ đề chính của hội nghị
    xoay quanh các vấn đề an ninh toàn cầu, biến đổi khí hậu và phục hồi kinh tế
    sau đại dịch COVID-19. Thủ tướng Nhật Bản Fumio Kishida đã có bài phát biểu khai
    mạc ấn tượng, nhấn mạnh tầm quan trọng của hợp tác quốc tế. Các nhà lãnh đạo
    thảo luận về cuộc xung đột ở Ukraine và cam kết tiếp tục hỗ trợ Kyiv. Hội nghị
    cũng thông qua tuyên bố chung về an toàn hạt nhân và giải trừ quân bị. Đây được
    xem là một trong những hội nghị G7 quan trọng nhất trong thập kỷ qua. Kết quả
    hội nghị được kỳ vọng sẽ định hình chính sách toàn cầu trong những năm tới.
    Nhiều hiệp định song phương cũng được ký kết bên lề hội nghị.
    """

    print("=== TÓM TẮT TRÍCH XUẤT (TextRank) ===")
    summary = extractive_summarize(sample_text, sentence_count=3)
    print(summary)
