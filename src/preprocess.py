"""Facade for Vietnamese text preprocessing.

Delegates all calls to the modern implementation in preprocess/preprocessor.py
to maintain consistency across training, api, testing, and evaluation pipelines.
"""

from __future__ import annotations

from preprocess.preprocessor import (
    clean_dataset_record,
    clean_generated_summary,
    clean_text,
    dedupe_similar_sentences,
    deduplicate_records,
    detect_garbled_text,
    fix_decimal_spacing,
    fix_spaced_letters,
    is_editorial_noise_sentence,
    is_probably_bad_generation,
    normalize_punctuation,
    normalize_unicode,
    normalize_whitespace,
    post_clean_vit5_telex,
    preprocess,
    remove_html_tags,
    remove_noise_characters,
    split_sentences,
    strip_editorial_chrome,
    text_fingerprint,
    tokenize_words,
)


def tokenize_sentences(text: str) -> list[str]:
    """Vietnamese sentence tokenizer (alias kept for backward-compat)."""
    return split_sentences(text)