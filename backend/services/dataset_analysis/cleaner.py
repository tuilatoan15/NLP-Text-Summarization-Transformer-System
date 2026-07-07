"""Text cleaning utilities for dataset analytics (wraps project preprocessor)."""

from __future__ import annotations

from preprocess.preprocessor import (
    clean_text,
    split_sentences,
    tokenize_words,
    text_fingerprint,
    VN_STOPWORDS,
)

__all__ = [
    "clean_text",
    "split_sentences",
    "tokenize_words",
    "text_fingerprint",
    "VN_STOPWORDS",
    "clean_article",
    "clean_summary",
]


def clean_article(text: str) -> str:
    return clean_text(text or "", aggressive=True)


def clean_summary(text: str) -> str:
    return clean_text(text or "", aggressive=False)
