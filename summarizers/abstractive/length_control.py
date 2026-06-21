"""Map target summary length (% of source) to extractive/abstractive controls."""

from __future__ import annotations

from src.preprocess import split_sentences
from src.utils import count_words


def compute_length_targets(
    source_text: str,
    target_length_ratio: int = 20,
    *,
    sentence_count: int | None = None,
    max_output_length: int | None = None,
) -> dict:
    ratio = max(10, min(100, int(target_length_ratio)))
    source_words = max(1, count_words(source_text))
    source_sentences = max(1, len(split_sentences(source_text)))
    target_words = max(5, round(source_words * ratio / 100))

    resolved_sentence_count = sentence_count
    if resolved_sentence_count is None:
        estimated = max(1, round(target_words / 12))
        resolved_sentence_count = max(1, min(20, estimated, source_sentences))

    resolved_max_output = max_output_length
    if resolved_max_output is None:
        resolved_max_output = max(24, min(512, target_words))

    return {
        "target_length_ratio": ratio,
        "source_words": source_words,
        "source_sentences": source_sentences,
        "target_words": target_words,
        "sentence_count": resolved_sentence_count,
        "max_output_length": resolved_max_output,
    }


def trim_summary_to_word_budget(summary: str, target_words: int) -> str:
    if not summary or target_words <= 0:
        return (summary or "").strip()

    sentences = split_sentences(summary)
    if not sentences:
        words = summary.split()
        return " ".join(words[:target_words]).strip()

    kept: list[str] = []
    total = 0
    for sentence in sentences:
        w = count_words(sentence)
        if total + w > target_words and kept:
            break
        kept.append(sentence)
        total += w
        if total >= target_words:
            break

    result = " ".join(kept).strip()
    if count_words(result) > target_words:
        result = " ".join(result.split()[:target_words])
    return result


def length_ratio_percent(summary_words: int, source_words: int) -> float:
    if source_words <= 0:
        return 0.0
    return round(100.0 * summary_words / source_words, 1)


def words_to_max_new_tokens(word_budget: int) -> int:
    words = max(1, int(word_budget))
    return max(24, min(512, int(words * 1.45)))


def min_new_tokens_for_budget(word_budget: int) -> int:
    words = max(1, int(word_budget))
    target = max(12, int(words * 0.35))
    return min(words_to_max_new_tokens(words) - 1, target)


def allocate_chunk_word_budgets(chunks: list[str], total_word_budget: int) -> list[int]:
    weights = [max(1, count_words(chunk)) for chunk in chunks]
    total = sum(weights)
    budgets: list[int] = []
    remaining = max(5, int(total_word_budget))
    for idx, weight in enumerate(weights):
        if idx == len(weights) - 1:
            share = remaining
        else:
            share = max(20, round(total_word_budget * weight / total))
            remaining -= share
        budgets.append(share)
    return budgets
