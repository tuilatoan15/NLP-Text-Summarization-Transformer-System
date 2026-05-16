"""
evaluate.py — Evaluation metrics for summarization research.

Optimization vs old version
─────────────────────────────
1. Metrics are split into two tiers:
   • LIGHT  (< 1 ms/sample):  ROUGE, BLEU — always computed synchronously.
   • HEAVY  (1–5 s/sample):   BERTScore, Semantic Similarity
     – Run in a ThreadPoolExecutor so they don't block the API.
     – Timeout controlled by config.HEAVY_METRICS_TIMEOUT.

2. _load_sentence_transformer() is cached with lru_cache so the SBERT model
   is only instantiated once (was already in old code, kept here).

3. evaluate_summary_fast() returns immediately with ROUGE/BLEU only, then
   heavy metrics are merged in once the background Future resolves.

4. evaluate_summary() (old interface) remains FULLY COMPATIBLE — it still
   blocks until all metrics are ready, so dashboard.py works unchanged.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import lru_cache
from typing import Iterable

from rouge_score import rouge_scorer

from src import config
from src.preprocess import clean_text, tokenize_words
from src.utils import compression_ratio, logger


# ─────────────────────────── Shared resources ──────────────────────────────

_ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL", "rougeLsum"],
    use_stemmer=False,
)

# Thread pool for heavy metric computation
_HEAVY_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="heavy_metrics")

# Default "null" values returned when heavy metrics are unavailable
_NULL_BERTSCORE = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
_NULL_HEAVY: dict = {"bertscore": _NULL_BERTSCORE, "bertscore_f1": 0.0, "semantic_similarity": 0.0}


# ─────────────────────────── LIGHT metrics ─────────────────────────────────

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


# ─────────────────────────── HEAVY metrics ─────────────────────────────────

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


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer
    logger.info("Loading SentenceTransformer: %s", model_name)
    return SentenceTransformer(model_name)


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


# ─────────────────────────── Combined evaluator ────────────────────────────

def _compute_heavy(prediction: str, reference: str) -> dict:
    """Compute BERTScore + Semantic Similarity (slow, run in thread pool)."""
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
    """
    Compute the full metric suite (ROUGE + BLEU + BERTScore + Semantic Similarity).

    BERTScore and Semantic Similarity are offloaded to a thread pool but this
    function BLOCKS waiting for them up to *timeout* seconds.  If they exceed
    the timeout, zeros are returned and a warning is logged.

    This interface is backward-compatible with the old evaluate_summary().
    """
    rouge = compute_rouge(prediction, reference)
    bleu = compute_bleu(prediction, reference)
    comp_ratio = compression_ratio(prediction, source_text)

    # Submit heavy metrics to background thread
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
        "processing_time": round(processing_time, 4),
    }


def evaluate_summary_fast(
    prediction: str,
    reference: str,
    source_text: str,
    processing_time: float,
) -> tuple[dict, Future]:
    """
    Return ROUGE/BLEU immediately plus a Future for heavy metrics.

    Use this in streaming endpoints where you want to push ROUGE/BLEU to the
    client right away and heavy metrics later once the Future resolves.

    Usage:
        fast_metrics, heavy_future = evaluate_summary_fast(...)
        # stream fast_metrics to client ...
        try:
            heavy = heavy_future.result(timeout=30)
            fast_metrics.update(heavy)
        except TimeoutError:
            pass
    """
    rouge = compute_rouge(prediction, reference)
    bleu = compute_bleu(prediction, reference)
    comp_ratio = compression_ratio(prediction, source_text)

    base = {
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "rougeLsum": rouge["rougeLsum"],
        "bleu": bleu,
        "compression_ratio": comp_ratio,
        "processing_time": round(processing_time, 4),
        **_NULL_HEAVY,  # placeholders until future resolves
    }
    future: Future = _HEAVY_POOL.submit(_compute_heavy, prediction, reference)
    return base, future


# ─────────────────────────── Aggregate helpers ─────────────────────────────

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
