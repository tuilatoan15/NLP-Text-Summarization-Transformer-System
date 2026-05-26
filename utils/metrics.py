"""Metrics used for document extraction, chunking, retrieval, and summarization prep."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np


VIETNAMESE_MARK_RE = re.compile(
    r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[\wÀ-ỹĐđ]+", re.UNICODE)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text or "")]


def estimate_language(text: str) -> str:
    sample = text[:8000]
    vi_marks = len(VIETNAMESE_MARK_RE.findall(sample))
    ascii_words = len(re.findall(r"\b[a-zA-Z]{3,}\b", sample))
    if vi_marks >= 8 or (vi_marks >= 3 and ascii_words >= 20):
        return "vi"
    if ascii_words >= 20:
        return "en"
    return "unknown"


def extraction_quality_score(text: str) -> dict[str, float | int | str]:
    words = tokenize_words(text)
    if not text.strip():
        return {
            "score": 0.0,
            "word_count": 0,
            "language": "unknown",
            "noise_ratio": 1.0,
            "single_char_ratio": 1.0,
            "line_break_density": 1.0,
        }

    non_text_chars = len(re.findall(r"[^\w\sÀ-ỹĐđ.,;:!?()/%+\-|]", text, flags=re.UNICODE))
    noise_ratio = safe_divide(non_text_chars, len(text))
    single_char_ratio = safe_divide(sum(1 for word in words if len(word) == 1), len(words))
    line_break_density = safe_divide(text.count("\n"), max(1, len(words)))
    avg_word_len = safe_divide(sum(len(word) for word in words), len(words))
    language = estimate_language(text)

    score = 1.0
    score -= min(0.35, noise_ratio * 3.0)
    score -= min(0.25, max(0.0, single_char_ratio - 0.18))
    score -= min(0.20, max(0.0, line_break_density - 0.12))
    if len(words) < 20:
        score -= 0.25
    if avg_word_len < 2.8:
        score -= 0.15
    if language == "unknown":
        score -= 0.10

    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "word_count": len(words),
        "language": language,
        "noise_ratio": round(noise_ratio, 4),
        "single_char_ratio": round(single_char_ratio, 4),
        "line_break_density": round(line_break_density, 4),
    }


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    return normalized @ normalized.T


def pairwise_adjacent_similarity(vectors: np.ndarray) -> list[float]:
    if vectors.shape[0] < 2:
        return []
    sims: list[float] = []
    for idx in range(vectors.shape[0] - 1):
        left = vectors[idx]
        right = vectors[idx + 1]
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        sims.append(float(np.dot(left, right) / denom) if denom else 0.0)
    return sims


def chunk_coherence_score(embeddings: np.ndarray) -> dict[str, float]:
    sims = pairwise_adjacent_similarity(embeddings)
    if not sims:
        return {"adjacent_similarity_mean": 0.0, "adjacent_similarity_min": 0.0}
    return {
        "adjacent_similarity_mean": round(float(np.mean(sims)), 4),
        "adjacent_similarity_min": round(float(np.min(sims)), 4),
    }


def lexical_overlap(reference: str, candidate: str) -> float:
    ref_tokens = Counter(tokenize_words(reference))
    cand_tokens = Counter(tokenize_words(candidate))
    if not ref_tokens or not cand_tokens:
        return 0.0
    overlap = sum((ref_tokens & cand_tokens).values())
    return round(overlap / sum(ref_tokens.values()), 4)


def retrieval_accuracy_at_k(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    relevant_indices: Iterable[int],
    k: int = 5,
) -> dict[str, float | list[int]]:
    relevant = set(relevant_indices)
    if chunk_embeddings.size == 0 or query_embedding.size == 0:
        return {"hit_at_k": 0.0, "mrr": 0.0, "top_indices": []}
    query = query_embedding.reshape(1, -1)
    sims = (query @ chunk_embeddings.T).reshape(-1)
    order = list(np.argsort(-sims)[:k])
    hit = any(idx in relevant for idx in order)
    rr = 0.0
    for rank, idx in enumerate(order, start=1):
        if idx in relevant:
            rr = 1.0 / rank
            break
    return {"hit_at_k": 1.0 if hit else 0.0, "mrr": round(rr, 4), "top_indices": order}


def compression_coverage_proxy(source: str, chunks: list[str]) -> dict[str, float]:
    source_tokens = set(tokenize_words(source))
    chunk_tokens = set(token for chunk in chunks for token in tokenize_words(chunk))
    if not source_tokens:
        return {"token_coverage": 0.0, "compression_ratio": 0.0}
    joined = " ".join(chunks)
    return {
        "token_coverage": round(len(source_tokens & chunk_tokens) / len(source_tokens), 4),
        "compression_ratio": round(len(tokenize_words(joined)) / max(1, len(tokenize_words(source))), 4),
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else math.nan
