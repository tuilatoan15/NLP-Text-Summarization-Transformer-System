"""
metrics.py — Evaluation metrics for summarization research.
"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import lru_cache
from typing import Iterable

from rouge_score import rouge_scorer

from src import config
from src.preprocess import clean_text, tokenize_words
from src.utils import compression_ratio, logger


_ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL", "rougeLsum"],
    use_stemmer=False,
)

_HEAVY_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="heavy_metrics")

_NULL_BERTSCORE = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
_NULL_HEAVY: dict = {"bertscore": _NULL_BERTSCORE, "bertscore_f1": 0.0, "semantic_similarity": 0.0}


def compute_rouge(prediction: str, reference: str, use_stemmer: bool = False) -> dict[str, float]:
    if not prediction or not prediction.strip() or not reference or not reference.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    scorer = _ROUGE_SCORER if not use_stemmer else rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True,
    )
    scores = scorer.score(clean_text(reference), clean_text(prediction))
    return {key: round(float(value.fmeasure), 4) for key, value in scores.items()}


def compute_rouge_batch(
    predictions: list[str],
    references: list[str],
    use_stemmer: bool = False,
) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    rows = [compute_rouge(p, r, use_stemmer=use_stemmer) for p, r in zip(predictions, references)]
    return {
        k: round(sum(row[k] for row in rows) / len(rows), 4)
        for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    }


def _ngram_counts(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for i in range(max(0, len(tokens) - n + 1)):
        ngram = tuple(tokens[i : i + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def compute_bleu(prediction: str, reference: str, max_order: int = 4) -> float:
    pred_tokens = tokenize_words(prediction)
    ref_tokens = tokenize_words(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    if pred_tokens == ref_tokens:
        return 1.0

    precisions = []
    for n in range(1, max_order + 1):
        pred_counts = _ngram_counts(pred_tokens, n)
        ref_counts = _ngram_counts(ref_tokens, n)
        overlap = sum(min(c, ref_counts.get(ngram, 0)) for ngram, c in pred_counts.items())
        total = max(1, sum(pred_counts.values()))
        precisions.append((overlap + 1) / (total + 1))

    log_prec = sum(math.log(p) for p in precisions) / max_order
    brevity = (
        1.0 if len(pred_tokens) > len(ref_tokens)
        else math.exp(1 - len(ref_tokens) / max(1, len(pred_tokens)))
    )
    return round(float(brevity * math.exp(log_prec)), 4)


def _lexical_f1(prediction: str, reference: str) -> float:
    pred = tokenize_words(prediction)
    ref = tokenize_words(reference)
    if not pred or not ref:
        return 0.0
    pred_counts: dict[str, int] = {}
    for t in pred:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    ref_counts: dict[str, int] = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = sum(
        min(pred_counts.get(t, 0), ref_counts.get(t, 0))
        for t in set(pred_counts) | set(ref_counts)
    )
    precision = overlap / max(1, len(pred))
    recall = overlap / max(1, len(ref))
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def compute_bertscore(
    prediction: str,
    reference: str,
    lang: str = config.BERTSCORE_LANG,
    model_type: str = config.BERTSCORE_MODEL,
) -> dict[str, float]:
    if not prediction or not prediction.strip() or not reference or not reference.strip():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if clean_text(prediction) == clean_text(reference):
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    try:
        from bert_score import score as bert_score_fn

        precision, recall, f1 = bert_score_fn(
            [prediction], [reference],
            lang=lang,
            model_type=model_type,
            verbose=False,
            rescale_with_baseline=False,
        )
        return {
            "precision": round(float(precision[0]), 4),
            "recall": round(float(recall[0]), 4),
            "f1": round(float(f1[0]), 4),
        }
    except Exception as exc:
        logger.warning("BERTScore unavailable, using lexical F1 fallback: %s", exc)
        val = _lexical_f1(prediction, reference)
        return {"precision": val, "recall": val, "f1": val}


_SBERT_LOAD_LOCK = threading.Lock()
_SBERT_CACHE: dict[str, object] = {}


def _load_sentence_transformer(model_name: str):
    """Thread-safe SentenceTransformer loader with manual cache.

    lru_cache alone is not safe when multiple threads call this simultaneously
    before the cache is populated — the concurrent calls race and encounter
    a half-initialised model in 'meta' state, triggering the
    'Cannot copy out of meta tensor' error.
    """
    if model_name in _SBERT_CACHE:
        return _SBERT_CACHE[model_name]
    with _SBERT_LOAD_LOCK:
        # Double-checked locking: another thread may have loaded while we waited.
        if model_name not in _SBERT_CACHE:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer: %s", model_name)
            _SBERT_CACHE[model_name] = SentenceTransformer(model_name)
    return _SBERT_CACHE[model_name]


def compute_semantic_similarity(
    prediction: str,
    reference: str,
    model_name: str = config.SBERT_MODEL,
) -> float:
    if not prediction or not prediction.strip() or not reference or not reference.strip():
        return 0.0
    if clean_text(prediction) == clean_text(reference):
        return 1.0
    try:
        from sentence_transformers import util

        model = _load_sentence_transformer(model_name)
        embeddings = model.encode(
            [prediction, reference],
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        cosine = float(util.cos_sim(embeddings[0], embeddings[1]).item())
        return round((cosine + 1.0) / 2.0, 4)
    except Exception as exc:
        logger.warning("Semantic similarity unavailable, lexical fallback: %s", exc)
        return _lexical_f1(prediction, reference)


def _compute_heavy(prediction: str, reference: str) -> dict:
    bertscore = compute_bertscore(prediction, reference)
    semantic = compute_semantic_similarity(prediction, reference)
    return {
        "bertscore": bertscore,
        "bertscore_f1": bertscore["f1"],
        "semantic_similarity": semantic,
    }


def evaluate_summary(
    prediction: str,
    reference: str,
    source_text: str,
    processing_time: float,
    timeout: float = config.HEAVY_METRICS_TIMEOUT,
) -> dict:
    cleaned_ref = clean_text(reference or "", aggressive=True)
    cleaned_src = clean_text(source_text or "", aggressive=True)
    
    is_real_reference = True
    if not cleaned_ref or cleaned_ref == cleaned_src:
        is_real_reference = False

    if is_real_reference or config.ALLOW_SOURCE_AS_REFERENCE:
        rouge = compute_rouge(prediction, reference)
        bleu = compute_bleu(prediction, reference)
        warning_msg = None
    else:
        rouge = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
        bleu = 0.0
        warning_msg = "Reference summary not provided — overlap metrics may be biased."

    pred_words = len(prediction.split()) if prediction else 0
    src_words = len(source_text.split()) if source_text else 1
    
    from src.preprocess import split_sentences
    pred_sents = len(split_sentences(prediction)) if prediction else 0
    src_sents = len(split_sentences(source_text)) if source_text else 1

    comp_ratio = round(pred_words / max(1, src_words), 4)
    compression_details = {
        "token_compression": round(pred_words / max(1, src_words), 4),
        "sentence_compression": round(pred_sents / max(1, src_sents), 4),
        "percent_reduction": round(100.0 * (1.0 - (pred_words / max(1, src_words))), 2),
    }

    future: Future = _HEAVY_POOL.submit(_compute_heavy, prediction, reference)
    try:
        heavy = future.result(timeout=timeout)
    except FutureTimeoutError:
        logger.warning(
            "Heavy metrics timed out after %.1f s — returning zeros for BERTScore/Semantic",
            timeout,
        )
        heavy = _NULL_HEAVY.copy()
    except Exception as exc:
        logger.warning("Heavy metrics error: %s", exc)
        heavy = _NULL_HEAVY.copy()

    from evaluation.readability import readability_scores

    readability = readability_scores(prediction)
    memory_mb: float | None = None
    try:
        import psutil

        memory_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except Exception:
        memory_mb = None

    human_eval = {
        "readability": readability.get("avg_sentence_length", 0.0),
        "fluency": max(0.0, 1.0 - readability.get("redundancy_ratio", 0.0)),
        "factual_consistency": 0.0,
        "coherence": readability.get("sentence_count", 0),
    }

    return {
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "rougeLsum": rouge["rougeLsum"],
        "bleu": bleu,
        "bertscore": heavy["bertscore"],
        "bertscore_f1": heavy["bertscore_f1"],
        "semantic_similarity": heavy["semantic_similarity"],
        "compression_ratio": comp_ratio,
        "compression_details": compression_details,
        "processing_time": round(processing_time, 4),
        "memory_usage_mb": memory_mb,
        "readability": readability,
        "warning": warning_msg,
        "is_biased": not is_real_reference,
        "human_eval_ready": human_eval,
    }


def evaluate_summary_fast(
    prediction: str,
    reference: str,
    source_text: str,
    processing_time: float,
) -> tuple[dict, Future]:
    cleaned_ref = clean_text(reference or "", aggressive=True)
    cleaned_src = clean_text(source_text or "", aggressive=True)
    
    is_real_reference = True
    if not cleaned_ref or cleaned_ref == cleaned_src:
        is_real_reference = False

    if is_real_reference or config.ALLOW_SOURCE_AS_REFERENCE:
        rouge = compute_rouge(prediction, reference)
        bleu = compute_bleu(prediction, reference)
        warning_msg = None
    else:
        rouge = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
        bleu = 0.0
        warning_msg = "Reference summary not provided — overlap metrics may be biased."

    pred_words = len(prediction.split()) if prediction else 0
    src_words = len(source_text.split()) if source_text else 1
    
    from src.preprocess import split_sentences
    pred_sents = len(split_sentences(prediction)) if prediction else 0
    src_sents = len(split_sentences(source_text)) if source_text else 1

    comp_ratio = round(pred_words / max(1, src_words), 4)
    compression_details = {
        "token_compression": round(pred_words / max(1, src_words), 4),
        "sentence_compression": round(pred_sents / max(1, src_sents), 4),
        "percent_reduction": round(100.0 * (1.0 - (pred_words / max(1, src_words))), 2),
    }

    human_eval = {
        "readability": 0.0,
        "fluency": 0.0,
        "factual_consistency": 0.0,
        "coherence": 0.0,
    }

    base = {
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "rougeLsum": rouge["rougeLsum"],
        "bleu": bleu,
        "compression_ratio": comp_ratio,
        "compression_details": compression_details,
        "processing_time": round(processing_time, 4),
        "warning": warning_msg,
        "is_biased": not is_real_reference,
        "human_eval_ready": human_eval,
        **_NULL_HEAVY,
    }
    future: Future = _HEAVY_POOL.submit(_compute_heavy, prediction, reference)
    return base, future


def aggregate_metric_rows(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        return {}
    keys = [
        "rouge1", "rouge2", "rougeL", "bleu",
        "bertscore_f1", "semantic_similarity",
        "compression_ratio", "processing_time",
    ]
    return {k: round(sum(float(r.get(k, 0.0)) for r in rows) / len(rows), 4) for k in keys}


def time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def evaluate_both(extractive_summary: str, abstractive_summary: str, reference: str) -> dict:
    return {
        "extractive": compute_rouge(extractive_summary, reference),
        "abstractive": compute_rouge(abstractive_summary, reference),
    }


# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# (evaluation/__init__.py and legacy callers import these names)
# ---------------------------------------------------------------------------

def evaluate_pair(
    prediction: str,
    reference: str,
    source_text: str = "",
    processing_time: float = 0.0,
    timeout: float = 30.0,
) -> dict:
    """Alias for :func:`evaluate_summary` kept for backward compatibility."""
    return evaluate_summary(
        prediction=prediction,
        reference=reference,
        source_text=source_text,
        processing_time=processing_time,
        timeout=timeout,
    )


def evaluate_batch(
    predictions: list[str],
    references: list[str],
    use_stemmer: bool = False,
) -> dict[str, float]:
    """Alias for :func:`compute_rouge_batch` kept for backward compatibility."""
    return compute_rouge_batch(predictions, references, use_stemmer=use_stemmer)


def aggregate_rows(rows) -> dict:
    """Alias for :func:`aggregate_metric_rows` kept for backward compatibility."""
    return aggregate_metric_rows(rows)

