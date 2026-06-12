"""
fact_check.py - Semantic consistency scoring with SentenceTransformer.
"""

from functools import lru_cache
import hashlib
import os
import re
from typing import Literal

import numpy as np

from src.preprocess import split_sentences
from src.utils import logger


from src import config

ConsistencyMode = Literal["fast", "full"]

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


@lru_cache(maxsize=1)
def _get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer

        model_name = getattr(config, "SBERT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        logger.info(f"Loading consistency embedding model: {model_name}")
        return SentenceTransformer(model_name)
    except Exception as exc:
        logger.warning(f"SentenceTransformer unavailable, using lexical fallback: {exc}")
        return None


import threading
from collections import OrderedDict

class LRUEmbeddingCache:
    def __init__(self, maxsize: int = 2000):
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> np.ndarray | None:
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def set(self, key: str, value: np.ndarray) -> None:
        with self.lock:
            if key in self.cache:
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                self.cache[key] = value
                if len(self.cache) > self.maxsize:
                    self.cache.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        with self.lock:
            return key in self.cache

    def __getitem__(self, key: str) -> np.ndarray:
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        self.set(key, value)

_embedding_cache = LRUEmbeddingCache(maxsize=2000)


def check_consistency(
    summary: str,
    source_text: str,
    mode: ConsistencyMode = "fast",
    max_summary_sentences: int | None = None,
    top_k: int = 3,
) -> dict:
    source_sentences = split_sentences(source_text)
    summary_sentences = split_sentences(summary)

    if mode == "fast":
        max_summary_sentences = max_summary_sentences or 5
        source_sentences = _candidate_source_sentences(summary_sentences, source_sentences, limit=80)
    else:
        max_summary_sentences = max_summary_sentences or 12

    summary_sentences = summary_sentences[:max_summary_sentences]
    if not source_sentences or not summary_sentences:
        return {
            "consistency_score": 0.0,
            "consistency_percent": 0,
            "status": "unsupported",
            "mode": mode,
            "checks": [],
            "suspicious_spans": [],
            "statistics": {"checked_sentences": 0, "suspicious_count": 0},
        }

    model = _get_embedding_model()
    if model is None:
        return _lexical_consistency(summary_sentences, source_sentences, mode, top_k)

    source_embeddings = _encode_sentences(source_sentences, model)
    summary_embeddings = _encode_sentences(summary_sentences, model)
    similarity = summary_embeddings @ source_embeddings.T

    checks = []
    for row_index, sentence in enumerate(summary_sentences):
        row = similarity[row_index]
        best_indexes = np.argsort(row)[::-1][:top_k]
        evidence = [
            {
                "index": int(index),
                "sentence": source_sentences[int(index)],
                "score": round(float(row[int(index)]), 4),
            }
            for index in best_indexes
        ]
        best = evidence[0]
        support_score = max(0.0, min(1.0, float(best["score"])))
        status = _status_from_score(support_score)
        reason = _reason_from_status(status, support_score)
        checks.append({
            "summary_sentence": sentence,
            "status": status,
            "support_score": round(support_score, 4),
            "support_percent": round(support_score * 100),
            "best_evidence": best,
            "evidence": evidence,
            "reason": reason,
        })

    return _build_result(checks, mode)


def _encode_sentences(sentences: list[str], model) -> np.ndarray:
    missing = []
    keys = []
    for sentence in sentences:
        key = _cache_key(sentence)
        keys.append(key)
        if key not in _embedding_cache:
            missing.append(sentence)

    if missing:
        embeddings = model.encode(
            missing,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for sentence, embedding in zip(missing, embeddings):
            _embedding_cache[_cache_key(sentence)] = embedding

    return np.vstack([_embedding_cache[key] for key in keys])


def _candidate_source_sentences(
    summary_sentences: list[str],
    source_sentences: list[str],
    limit: int,
) -> list[str]:
    if len(source_sentences) <= limit:
        return source_sentences

    query_tokens = set()
    for sentence in summary_sentences:
        query_tokens.update(_tokens(sentence))

    scored = []
    for sentence in source_sentences:
        tokens = set(_tokens(sentence))
        score = len(query_tokens & tokens) / max(1, len(query_tokens | tokens))
        scored.append((score, sentence))
    scored.sort(reverse=True, key=lambda item: item[0])
    return [sentence for _, sentence in scored[:limit]]


def _lexical_consistency(
    summary_sentences: list[str],
    source_sentences: list[str],
    mode: str,
    top_k: int,
) -> dict:
    checks = []
    for sentence in summary_sentences:
        scored = []
        sentence_tokens = set(_tokens(sentence))
        for index, source_sentence in enumerate(source_sentences):
            source_tokens = set(_tokens(source_sentence))
            score = len(sentence_tokens & source_tokens) / max(1, len(sentence_tokens | source_tokens))
            scored.append((score, index, source_sentence))
        scored.sort(reverse=True, key=lambda item: item[0])
        evidence = [
            {"index": index, "sentence": source_sentence, "score": round(score, 4)}
            for score, index, source_sentence in scored[:top_k]
        ]
        best = evidence[0] if evidence else {"score": 0.0, "sentence": "", "index": -1}
        support_score = float(best["score"])
        status = _status_from_score(support_score)
        checks.append({
            "summary_sentence": sentence,
            "status": status,
            "support_score": round(support_score, 4),
            "support_percent": round(support_score * 100),
            "best_evidence": best,
            "evidence": evidence,
            "reason": _reason_from_status(status, support_score),
        })
    return _build_result(checks, mode)


def _build_result(checks: list[dict], mode: str) -> dict:
    score = round(sum(item["support_score"] for item in checks) / len(checks), 4) if checks else 0.0
    suspicious = [item for item in checks if item["status"] != "supported"]
    status = "consistent"
    if score < 0.45:
        status = "high_risk"
    elif suspicious:
        status = "needs_review"

    logger.info(
        "Consistency stats: mode=%s score=%.2f suspicious=%s checked=%s",
        mode,
        score,
        len(suspicious),
        len(checks),
    )

    return {
        "consistency_score": score,
        "consistency_percent": round(score * 100),
        "status": status,
        "mode": mode,
        "checks": checks,
        "suspicious_spans": [
            {
                "text": item["summary_sentence"],
                "status": item["status"],
                "reason": item["reason"],
                "support_score": item["support_score"],
                "support_percent": item["support_percent"],
            }
            for item in suspicious
        ],
        "statistics": {
            "checked_sentences": len(checks),
            "suspicious_count": len(suspicious),
        },
    }


def _status_from_score(score: float) -> str:
    if score < 0.45:
        return "unsupported"
    if score < 0.62:
        return "suspicious"
    return "supported"


def _reason_from_status(status: str, score: float) -> str:
    if status == "supported":
        return f"Câu này có bằng chứng ngữ nghĩa gần trong văn bản gốc (similarity {score:.2f})."
    if status == "suspicious":
        return f"Similarity chỉ đạt {score:.2f}; nên kiểm tra lại bằng chứng nguồn."
    return f"Không tìm thấy bằng chứng ngữ nghĩa đủ mạnh (similarity {score:.2f})."


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"\w+", text, flags=re.UNICODE) if len(token) > 1]


def _cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
