"""
extractive_summarizer.py — Parallel extractive summarizers: TextRank, LexRank, LSA.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np

from src import config
from src.preprocess import clean_text, split_sentences, tokenize_words
from src.utils import logger


DEFAULT_SENTENCE_COUNT = 5

_EXTRACTIVE_POOL = ThreadPoolExecutor(
    max_workers=config.EXTRACTIVE_WORKERS,
    thread_name_prefix="extractive",
)


# ─────────────────────────── TF-IDF sentence matrix ────────────────────────

def _sentence_matrix(sentences: list[str]) -> tuple[np.ndarray, list[str]]:
    tokenized = [tokenize_words(s, remove_stopwords=True) for s in sentences]
    vocab = sorted({token for tokens in tokenized for token in tokens})
    if not vocab:
        return np.zeros((len(sentences), 0), dtype=np.float32), []

    index = {token: i for i, token in enumerate(vocab)}
    doc_freq = Counter(token for tokens in tokenized for token in set(tokens))
    n = max(1, len(sentences))
    matrix = np.zeros((len(sentences), len(vocab)), dtype=np.float32)

    for row, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        total = max(1, sum(counts.values()))
        for token, count in counts.items():
            tf = count / total
            idf = math.log((1 + n) / (1 + doc_freq[token])) + 1
            matrix[row, index[token]] = tf * idf

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)
    return matrix, vocab


def _cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float32)
    sim = matrix @ matrix.T
    np.fill_diagonal(sim, 0.0)
    sim[sim < 0] = 0.0
    return sim


def _pagerank(
    similarity: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    n = similarity.shape[0]
    if n == 0:
        return np.array([], dtype=np.float32)
    if similarity.sum() == 0:
        return np.ones(n, dtype=np.float32) / n

    col_sum = similarity.sum(axis=0)
    col_sum[col_sum == 0] = 1.0
    transition = similarity / col_sum
    scores = np.ones(n, dtype=np.float32) / n

    for _ in range(max_iter):
        new_scores = damping * (transition @ scores) + (1.0 - damping) / n
        if np.linalg.norm(new_scores - scores, ord=1) < tol:
            scores = new_scores
            break
        scores = new_scores
    return scores


# ─────────────────────────── Sentence selection ────────────────────────────

def _select_summary(
    sentences: list[str],
    scores: np.ndarray,
    sentence_count: int,
) -> tuple[str, list[dict]]:
    if not sentences:
        return "", []

    count = max(1, min(sentence_count, len(sentences)))
    if len(scores) != len(sentences):
        scores = np.ones(len(sentences), dtype=np.float32)

    # Greedy selection with light position continuity bias (helps LSA/LexRank coherence).
    norm_scores = scores.astype(np.float32).copy()
    if norm_scores.max() > 0:
        norm_scores = norm_scores / norm_scores.max()

    selected_idxs: list[int] = []
    remaining = set(range(len(sentences)))
    while len(selected_idxs) < count and remaining:
        best_idx = -1
        best_value = -1.0
        for idx in remaining:
            position_bonus = 0.0
            if selected_idxs:
                nearest = min(abs(idx - s) for s in selected_idxs)
                position_bonus = 0.12 if nearest == 1 else (0.06 if nearest <= 2 else 0.0)
            value = float(norm_scores[idx]) + position_bonus
            if value > best_value or (value == best_value and idx < best_idx):
                best_value = value
                best_idx = idx
        selected_idxs.append(best_idx)
        remaining.remove(best_idx)

    selected_idxs = sorted(selected_idxs)
    max_score = float(np.max(scores)) if len(scores) and float(np.max(scores)) > 0 else 1.0

    selected = [
        {
            "sentence": sentences[idx],
            "sentence_index": idx,
            "sentence_score": round(float(scores[idx]) / max_score, 4),
            "match_similarity": 1.0,
        }
        for idx in selected_idxs
    ]
    from src.preprocess import dedupe_similar_sentences, fix_decimal_spacing

    ordered_sentences = dedupe_similar_sentences([item["sentence"] for item in selected])
    summary = fix_decimal_spacing(" ".join(ordered_sentences))
    return summary, selected


def _prepare_sentences(text: str) -> list[str]:
    from src.preprocess import dedupe_similar_sentences, is_editorial_noise_sentence

    sentences = split_sentences(clean_text(text, aggressive=True))
    filtered = [s for s in sentences if not is_editorial_noise_sentence(s)]
    return dedupe_similar_sentences(filtered)


def _details(summary: str, selected: list[dict], source_sentences: list[str], algorithm: str) -> dict:
    return {
        "algorithm": algorithm,
        "summary": clean_text(summary),
        "selected_sentences": selected,
        "highlighted_sentence_indexes": [
            item["sentence_index"] for item in selected if item["sentence_index"] >= 0
        ],
        "source_sentences": source_sentences[:200],
    }


# ─────────────────────────── Algorithm implementations ─────────────────────

def _textrank_details(text: str, sentence_count: int = DEFAULT_SENTENCE_COUNT) -> dict:
    sentences = _prepare_sentences(text)
    if len(sentences) <= sentence_count:
        selected = [
            {"sentence": s, "sentence_index": i, "sentence_score": 1.0, "match_similarity": 1.0}
            for i, s in enumerate(sentences)
        ]
        return _details(" ".join(sentences), selected, sentences, "TextRank")

    matrix, _ = _sentence_matrix(sentences)
    scores = _pagerank(_cosine_similarity(matrix))
    summary, selected = _select_summary(sentences, scores, sentence_count)
    return _details(summary, selected, sentences, "TextRank")


def _lexrank_details(text: str, sentence_count: int = DEFAULT_SENTENCE_COUNT) -> dict:
    sentences = _prepare_sentences(text)
    if len(sentences) <= sentence_count:
        selected = [
            {"sentence": s, "sentence_index": i, "sentence_score": 1.0, "match_similarity": 1.0}
            for i, s in enumerate(sentences)
        ]
        return _details(" ".join(sentences), selected, sentences, "LexRank")

    matrix, _ = _sentence_matrix(sentences)
    similarity = _cosine_similarity(matrix)
    threshold = float(np.mean(similarity[similarity > 0])) if np.any(similarity > 0) else 0.1
    graph = np.where(similarity >= threshold, similarity, 0.0)
    scores = _pagerank(graph)
    summary, selected = _select_summary(sentences, scores, sentence_count)
    return _details(summary, selected, sentences, "LexRank")


def _tfidf_details(text: str, sentence_count: int = DEFAULT_SENTENCE_COUNT) -> dict:
    sentences = _prepare_sentences(text)
    if len(sentences) <= sentence_count:
        selected = [
            {"sentence": s, "sentence_index": i, "sentence_score": 1.0, "match_similarity": 1.0}
            for i, s in enumerate(sentences)
        ]
        return _details(" ".join(sentences), selected, sentences, "TF-IDF")

    matrix, _ = _sentence_matrix(sentences)
    if matrix.size == 0:
        scores = np.ones(len(sentences), dtype=np.float32)
    else:
        scores = matrix.sum(axis=1).astype(np.float32)

    summary, selected = _select_summary(sentences, scores, sentence_count)
    return _details(summary, selected, sentences, "TF-IDF")


def _lsa_details(text: str, sentence_count: int = DEFAULT_SENTENCE_COUNT) -> dict:
    sentences = _prepare_sentences(text)
    if len(sentences) <= sentence_count:
        selected = [
            {"sentence": s, "sentence_index": i, "sentence_score": 1.0, "match_similarity": 1.0}
            for i, s in enumerate(sentences)
        ]
        return _details(" ".join(sentences), selected, sentences, "LSA Summarizer")

    matrix, _ = _sentence_matrix(sentences)
    if matrix.size == 0:
        scores = np.ones(len(sentences), dtype=np.float32)
    else:
        try:
            _, singular_values, vt = np.linalg.svd(matrix.T, full_matrices=False)
            concepts = min(sentence_count, len(singular_values), vt.shape[0])
            weighted = (singular_values[:concepts, None] * vt[:concepts, :]) ** 2
            scores = np.sqrt(weighted.sum(axis=0))
        except Exception as exc:
            logger.warning("LSA SVD failed, fallback to centroid: %s", exc)
            centroid = matrix.mean(axis=0, keepdims=True)
            scores = (matrix @ centroid.T).reshape(-1)

    summary, selected = _select_summary(sentences, scores, sentence_count)
    return _details(summary, selected, sentences, "LSA Summarizer")


EXTRACTIVE_RUNNERS: dict[str, Callable[[str, int], dict]] = {
    "textrank": _textrank_details,
    "lexrank": _lexrank_details,
    "lsa": _lsa_details,
    "tfidf": _tfidf_details,
}


def summarize_extractive_algorithm(
    text: str,
    algorithm: str,
    sentence_count: int = DEFAULT_SENTENCE_COUNT,
) -> dict:
    key = algorithm.strip().lower()
    if key not in EXTRACTIVE_RUNNERS:
        raise KeyError(f"Unsupported extractive algorithm: {algorithm!r}")
    return EXTRACTIVE_RUNNERS[key](text, sentence_count)


def summarize_extractive_parallel(
    text: str,
    algorithms: list[str],
    sentence_count: int = DEFAULT_SENTENCE_COUNT,
) -> dict[str, dict]:
    keys = [k.strip().lower() for k in algorithms if k.strip().lower() in EXTRACTIVE_RUNNERS]
    if not keys:
        return {}

    t0 = time.perf_counter()
    futures = {
        _EXTRACTIVE_POOL.submit(EXTRACTIVE_RUNNERS[key], text, sentence_count): key
        for key in keys
    }
    results: dict[str, dict] = {}
    for future in as_completed(futures):
        key = futures[future]
        try:
            results[key] = future.result()
        except Exception as exc:
            logger.error("Extractive [%s] failed: %s", key, exc, exc_info=True)
            results[key] = {
                "algorithm": key,
                "summary": "",
                "selected_sentences": [],
                "highlighted_sentence_indexes": [],
                "source_sentences": [],
                "error": str(exc),
            }

    elapsed = time.perf_counter() - t0
    logger.info(
        "⚡ Extractive parallel [%s] done in %.3f s",
        ", ".join(keys), elapsed,
    )
    return results


def extractive_summarize(text: str, sentence_count: int = DEFAULT_SENTENCE_COUNT, **_: object) -> str:
    return _textrank_details(text, sentence_count)["summary"]


def lexrank_summarize(text: str, sentence_count: int = DEFAULT_SENTENCE_COUNT, **_: object) -> str:
    return _lexrank_details(text, sentence_count)["summary"]


def lsa_summarize(text: str, sentence_count: int = DEFAULT_SENTENCE_COUNT, **_: object) -> str:
    return _lsa_details(text, sentence_count)["summary"]


def tfidf_summarize(text: str, sentence_count: int = DEFAULT_SENTENCE_COUNT, **_: object) -> str:
    return _tfidf_details(text, sentence_count)["summary"]


def extractive_summarize_with_details(text: str, sentence_count: int = DEFAULT_SENTENCE_COUNT, **_: object) -> dict:
    return _textrank_details(text, sentence_count)
