"""Lightweight readability and redundancy heuristics for Vietnamese summaries."""

from __future__ import annotations

from src.preprocess import split_sentences, tokenize_words
from src.utils import count_words


def readability_scores(text: str) -> dict[str, float]:
    """Approximate readability without external corpora."""
    sentences = split_sentences(text)
    words = tokenize_words(text, remove_stopwords=False)
    if not sentences or not words:
        return {
            "avg_sentence_length": 0.0,
            "avg_word_length": 0.0,
            "sentence_count": 0,
            "redundancy_ratio": 0.0,
        }

    sent_lens = [len(tokenize_words(s, remove_stopwords=False)) for s in sentences]
    avg_sent = sum(sent_lens) / len(sent_lens)
    avg_word = sum(len(w) for w in words) / len(words)

    unique = len(set(w.lower() for w in words))
    redundancy = 1.0 - (unique / max(1, len(words)))

    return {
        "avg_sentence_length": round(avg_sent, 2),
        "avg_word_length": round(avg_word, 2),
        "sentence_count": len(sentences),
        "redundancy_ratio": round(redundancy, 4),
        "word_count": count_words(text),
    }


def repetition_ngrams(text: str, n: int = 3) -> float:
    tokens = tokenize_words(text, remove_stopwords=False)
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    unique = len(set(grams))
    return round(1.0 - unique / len(grams), 4)
