"""
summary_control.py - Convert user-facing length controls into model settings.
"""

from src.utils import count_words


WORD_PRESETS = {
    "100_words": 100,
    "200_words": 200,
}


def resolve_summary_controls(
    source_text: str,
    length_control: str = "auto",
    extractive_sentences: int | None = None,
    max_abstractive_length: int | None = None,
) -> dict:
    word_count = count_words(source_text)
    target_words = None

    if length_control == "20_percent":
        target_words = max(30, int(word_count * 0.2))
    elif length_control == "50_percent":
        target_words = max(50, int(word_count * 0.5))
    elif length_control in WORD_PRESETS:
        target_words = WORD_PRESETS[length_control]

    if target_words is None:
        target_words = max_abstractive_length or 150

    estimated_sentences = max(1, min(20, round(target_words / 25)))

    return {
        "length_control": length_control,
        "target_words": target_words,
        "extractive_sentences": extractive_sentences or estimated_sentences,
        "max_abstractive_length": max_abstractive_length or max(30, min(512, int(target_words * 1.4))),
    }


def enforce_word_limit(text: str, target_words: int | None) -> str:
    if not text or not target_words:
        return text
    words = text.split()
    if len(words) <= target_words:
        return text
    return " ".join(words[:target_words]).rstrip(" ,;:") + "..."
