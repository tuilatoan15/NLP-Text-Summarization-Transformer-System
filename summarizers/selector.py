"""
selector.py - Select the best summary using quality signals beyond ROUGE.
"""

import re

from src.evaluate import compute_rouge
from src.preprocess import split_sentences
from src.utils import count_words, logger


IDEAL_MIN_WORDS = 30
IDEAL_MAX_WORDS = 150


def compute_length_score(summary: str, ideal_min: int = IDEAL_MIN_WORDS, ideal_max: int = IDEAL_MAX_WORDS) -> float:
    word_count = count_words(summary)
    if word_count == 0:
        return 0.0
    if ideal_min <= word_count <= ideal_max:
        return 1.0
    if word_count < ideal_min:
        return round(word_count / ideal_min, 4)
    excess = word_count - ideal_max
    return round(max(0.0, 1.0 - excess / (ideal_max * 2)), 4)


def compute_readability_score(summary: str) -> float:
    sentences = split_sentences(summary)
    words = count_words(summary)
    if not sentences or words == 0:
        return 0.0
    avg_sentence_len = words / len(sentences)
    if 12 <= avg_sentence_len <= 28:
        sentence_score = 1.0
    else:
        sentence_score = max(0.0, 1.0 - abs(avg_sentence_len - 20) / 40)
    noisy_chars = len(re.findall(r"[^0-9A-Za-zÀ-ỹ\s.,;:!?%()\\/-]", summary))
    noise_penalty = min(0.35, noisy_chars / max(1, len(summary)))
    return round(max(0.0, sentence_score - noise_penalty), 4)


def compute_compression_score(summary: str, reference: str) -> float:
    source_words = max(1, count_words(reference))
    summary_words = count_words(summary)
    ratio = summary_words / source_words
    if 0.08 <= ratio <= 0.35:
        return 1.0
    if ratio < 0.08:
        return round(max(0.0, ratio / 0.08), 4)
    return round(max(0.0, 1.0 - (ratio - 0.35) / 0.65), 4)


def compute_redundancy_penalty(summary: str) -> float:
    sentences = split_sentences(summary)
    if len(sentences) <= 1:
        return 0.0
    token_sets = [_token_set(sentence) for sentence in sentences]
    similarities = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            similarities.append(_jaccard(token_sets[i], token_sets[j]))
    if not similarities:
        return 0.0
    high_overlap = [score for score in similarities if score > 0.55]
    return round(min(0.35, sum(high_overlap) / max(1, len(similarities))), 4)


def compute_combined_score(rouge_scores: dict, length_score: float, readability: float, compression: float, redundancy: float) -> float:
    rouge_quality = (
        0.45 * rouge_scores.get("rougeL", 0.0)
        + 0.35 * rouge_scores.get("rouge1", 0.0)
        + 0.20 * rouge_scores.get("rouge2", 0.0)
    )
    score = (
        0.38 * rouge_quality
        + 0.24 * readability
        + 0.20 * compression
        + 0.18 * length_score
        - redundancy
    )
    return round(max(0.0, min(1.0, score)), 4)


def select_best_summary(
    extractive_summary: str,
    abstractive_summary: str,
    reference: str,
    ideal_min_words: int = IDEAL_MIN_WORDS,
    ideal_max_words: int = IDEAL_MAX_WORDS,
) -> dict:
    logger.info("Selecting best summary with ROUGE/readability/compression/redundancy signals...")

    ext_scores = _score_summary(extractive_summary, reference, ideal_min_words, ideal_max_words)
    abs_scores = _score_summary(abstractive_summary, reference, ideal_min_words, ideal_max_words)

    abstractive_bonus = 0.0
    if (
        abs_scores["readability_score"] >= ext_scores["readability_score"]
        and abs_scores["compression_score"] >= ext_scores["compression_score"]
        and abs_scores["redundancy_penalty"] <= ext_scores["redundancy_penalty"]
    ):
        abstractive_bonus = 0.035

    adjusted_abs = min(1.0, abs_scores["combined_score"] + abstractive_bonus)
    if adjusted_abs > ext_scores["combined_score"]:
        best_type = "abstractive"
        best_summary = abstractive_summary
        abs_scores["selection_bonus"] = abstractive_bonus
        abs_scores["adjusted_score"] = round(adjusted_abs, 4)
    else:
        best_type = "extractive"
        best_summary = extractive_summary
        ext_scores["adjusted_score"] = ext_scores["combined_score"]
        abs_scores["selection_bonus"] = abstractive_bonus
        abs_scores["adjusted_score"] = round(adjusted_abs, 4)

    logger.info(
        "Selector result: %s (extractive=%.4f, abstractive=%.4f, bonus=%.3f)",
        best_type,
        ext_scores.get("adjusted_score", ext_scores["combined_score"]),
        abs_scores["adjusted_score"],
        abstractive_bonus,
    )

    return {
        "best_summary": best_summary,
        "best_type": best_type,
        "scores": {
            "extractive": ext_scores,
            "abstractive": abs_scores,
        },
    }


def _score_summary(summary: str, reference: str, ideal_min_words: int, ideal_max_words: int) -> dict:
    rouge = compute_rouge(summary, reference)
    length_score = compute_length_score(summary, ideal_min_words, ideal_max_words)
    readability = compute_readability_score(summary)
    compression = compute_compression_score(summary, reference)
    redundancy = compute_redundancy_penalty(summary)
    combined = compute_combined_score(rouge, length_score, readability, compression, redundancy)
    return {
        **rouge,
        "length_score": length_score,
        "readability_score": readability,
        "compression_score": compression,
        "redundancy_penalty": redundancy,
        "combined_score": combined,
    }


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"\w+", text, flags=re.UNICODE) if len(token) > 1}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
