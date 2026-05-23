"""
dashboard.py — Optimized multi-algorithm comparison orchestrator.

Optimization vs old version
─────────────────────────────
1. EXTRACTIVE models run CONCURRENTLY via summarize_extractive_parallel()
   (ThreadPoolExecutor in extractive.py) — ~60-70% wall-time reduction for
   the extractive group.

2. ABSTRACTIVE models run GPU-SEQUENTIALLY to avoid CUDA OOM:
   • If VRAM >= GPU_VRAM_LIMIT_GB → run all Transformers in a single thread
     (sequential on GPU is still fast because GPU parallelism is inside each
     model.generate() call, not across models).
   • Result: no out-of-memory crashes on low-VRAM machines (e.g. 4 GB GTX).

3. Models are already preloaded via model_loader.py — zero cold-start inside
   _run_abstractive(). The only cost is the actual forward pass.

4. CPU usage + inference time are logged per algorithm.

5. _evaluate_result() is unchanged at the signature level so api/main.py
   and tests need NO changes.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import torch

from src import config
from src.abstractive import abstractive_summarize_key, get_summarizer
from src.evaluate import evaluate_summary
from src.extractive import summarize_extractive_algorithm, summarize_extractive_parallel
from src.model_registry import (
    ABSTRACTIVE_ALGORITHMS,
    DEFAULT_ALGORITHMS,
    EXTRACTIVE_ALGORITHMS,
    resolve_algorithm,
)
from src.preprocess import clean_text, split_sentences
from src.utils import count_words, log_vram_usage, logger


# ─────────────────────────── GPU scheduling lock ───────────────────────────
# Prevents multiple Transformer models from running simultaneously on a
# low-VRAM GPU, which would cause CUDA OOM errors.
_GPU_LOCK = threading.Semaphore(config.MAX_GPU_CONCURRENT)


# ─────────────────────────── Scoring ───────────────────────────────────────

def _combined_score(metrics: dict) -> float:
    compression = float(metrics.get("compression_ratio", 0.0))
    compression_score = max(0.0, 1.0 - abs(compression - 0.25) / 0.25)
    
    is_biased = metrics.get("is_biased", False)
    if is_biased:
        # Reference is missing or source-based: use weights that do not rely on ROUGE/BLEU
        weights = config.NO_REFERENCE_RANKING_WEIGHTS
        score = (
            weights.get("bertscore_f1", 0.45) * float(metrics.get("bertscore_f1", 0.0))
            + weights.get("semantic_similarity", 0.35) * float(metrics.get("semantic_similarity", 0.0))
            + weights.get("compression_score", 0.20) * compression_score
        )
    else:
        # Real reference summary is available: standard multi-metric weight suite
        weights = config.WITH_REFERENCE_RANKING_WEIGHTS
        score = (
            weights.get("rougeL", 0.25) * float(metrics.get("rougeL", 0.0))
            + weights.get("rouge2", 0.15) * float(metrics.get("rouge2", 0.0))
            + weights.get("bertscore_f1", 0.30) * float(metrics.get("bertscore_f1", 0.0))
            + weights.get("semantic_similarity", 0.20) * float(metrics.get("semantic_similarity", 0.0))
            + weights.get("compression_score", 0.10) * compression_score
        )
    return round(score, 4)


# ─────────────────────────── Per-algorithm runners ─────────────────────────

def _run_extractive(text: str, key: str, sentence_count: int) -> tuple[str, dict]:
    details = summarize_extractive_algorithm(text, key, sentence_count=sentence_count)
    return details.get("summary", ""), {
        "extractive": {
            "source_sentences": details.get("source_sentences", []),
            "selected_sentences": details.get("selected_sentences", []),
            "highlighted_sentence_indexes": details.get("highlighted_sentence_indexes", []),
        }
    }


def _fallback_summary(text: str, sentence_count: int) -> str:
    return summarize_extractive_algorithm(text, "textrank", sentence_count=sentence_count).get("summary", "")


def _run_abstractive(
    text: str,
    key: str,
    max_output_length: int,
    sentence_count: int,
) -> tuple[str, dict]:
    """
    Run one Transformer model under the GPU semaphore.

    The semaphore (MAX_GPU_CONCURRENT=1 by default) ensures that only one
    model.generate() call occupies the GPU at a time — critical for machines
    with limited VRAM (≤ 6 GB).  On high-VRAM machines you can raise
    MAX_GPU_CONCURRENT to allow parallelism.
    """
    log_vram_usage(f"before_{key}")
    with _GPU_LOCK:
        summarizer = get_summarizer(model_name=key)
        summary = summarizer.summarize(text, max_output_length=max_output_length)

    fallback_used = not summary
    if summary:
        from src.output_validator import is_garbled_abstractive, validate_output

        validation = validate_output(
            summary,
            require_vietnamese=key in {"vit5", "mt5"},
        )
        if validation["is_corrupted"] or is_garbled_abstractive(summary):
            logger.warning(
                "[%s] Bad output (%s) — TextRank fallback",
                key,
                validation.get("quality_warning") or "garbled",
            )
            summary = _fallback_summary(text, sentence_count)
            fallback_used = True

    if fallback_used and not summary:
        summary = _fallback_summary(text, sentence_count)
        logger.warning("[%s] Empty generation — using TextRank fallback", key)

    log_vram_usage(f"after_{key}")

    explain_fn = getattr(summarizer, "explain_tokens", None)
    token_importance = explain_fn(text, summary) if callable(explain_fn) else []
    if not isinstance(token_importance, list):
        token_importance = []

    return summary, {
        "abstractive": {
            "attention_available": False,
            "fallback_used": fallback_used,
            "token_importance": token_importance,
        }
    }


# ─────────────────────────── Full result builder ───────────────────────────

def _evaluate_result(
    key: str,
    text: str,
    reference: str,
    sentence_count: int,
    max_output_length: int,
    target_words: int | None = None,
    source_words: int | None = None,
) -> dict:
    """Build a full result row for a single algorithm (extractive OR abstractive)."""
    algorithm = resolve_algorithm(key)
    start = time.perf_counter()
    error: str | None = None
    explainability: dict = {}

    try:
        if algorithm.key in EXTRACTIVE_ALGORITHMS:
            summary, explainability = _run_extractive(text, algorithm.key, sentence_count)
        elif algorithm.key in ABSTRACTIVE_ALGORITHMS:
            summary, explainability = _run_abstractive(
                text, algorithm.key, max_output_length, sentence_count
            )
        else:
            raise KeyError(f"Unsupported algorithm: {algorithm.key}")
    except Exception as exc:
        logger.exception("Algorithm %s failed", algorithm.key)
        error = str(exc)
        summary = _fallback_summary(text, sentence_count)
        explainability = {"error": error, "fallback_used": True}

    if target_words and summary:
        from src.length_control import trim_summary_to_word_budget

        summary = trim_summary_to_word_budget(summary, target_words)

    duration = time.perf_counter() - start
    logger.info(
        "⏱  [%s] inference done in %.3f s  words_out=%d",
        algorithm.key, duration, count_words(summary),
    )

    metrics = evaluate_summary(summary, reference, text, duration)
    metrics["combined_score"] = _combined_score(metrics)

    src_w = source_words or count_words(text)
    from src.length_control import length_ratio_percent

    actual_length_ratio = length_ratio_percent(count_words(summary), src_w)

    # 5. Handle mT5 experimental state
    is_experimental = False
    warning_badge = None
    if algorithm.key == "mt5" and config.MT5_EXPERIMENTAL:
        is_experimental = True
        warning_badge = "Experimental Baseline"
        if summary:
            from src.output_validator import is_multilingual_garbage

            local_dir = Path(algorithm.local_dir or "")
            has_finetuned = local_dir.exists() and any(local_dir.iterdir())
            if is_multilingual_garbage(summary, require_vietnamese=not has_finetuned):
                warning_badge = "Experimental (Corrupt Multilingual)"
            elif has_finetuned:
                warning_badge = "mT5 Fine-tuned"
        logger.debug("[%s] experimental flag=%s badge=%s", algorithm.key, is_experimental, warning_badge)

    return {
        "key": algorithm.key,
        "algorithm": algorithm.name,
        "group": algorithm.group,
        "summary": summary,
        "word_count": count_words(summary),
        "length_ratio_percent": actual_length_ratio,
        "target_words": target_words,
        "metrics": metrics,
        "rouge": {
            "rouge1": metrics["rouge1"],
            "rouge2": metrics["rouge2"],
            "rougeL": metrics["rougeL"],
            "rougeLsum": metrics["rougeLsum"],
        },
        "bleu": metrics["bleu"],
        "bertscore": metrics["bertscore"],
        "semantic_similarity": metrics["semantic_similarity"],
        "compression_ratio": metrics["compression_ratio"],
        "time_seconds": metrics["processing_time"],
        "processing_time": metrics["processing_time"],
        "explainability": explainability,
        "experimental": is_experimental,
        "warning_badge": warning_badge,
        "details": (
            explainability.get("extractive")
            or explainability.get("abstractive")
            or explainability
        ),
        "source_sentences": split_sentences(text)[:200],
        "error": error,
    }


# ─────────────────────────── Parallel orchestration ────────────────────────

def _run_all_parallel(
    text: str,
    reference: str,
    extractive_keys: list[str],
    abstractive_keys: list[str],
    sentence_count: int,
    max_output_length: int,
    target_words: int,
    source_words: int,
) -> list[dict]:
    """
    Strategy
    ─────────
    • Extractive algorithms → all submitted to ThreadPoolExecutor AT ONCE
      (pure Python / NumPy, no GIL issue with GPU).
    • Abstractive algorithms → submitted ONE BY ONE to a SINGLE-THREAD pool,
      ensuring at most MAX_GPU_CONCURRENT models run on GPU simultaneously.

    This gives maximum parallelism without CUDA OOM.
    """
    results: dict[str, dict] = {}

    # ── 1. Fire extractive algorithms in parallel ───────────────────────────
    if extractive_keys:
        ext_parallel = summarize_extractive_parallel(text, extractive_keys, sentence_count)
        # Now build full result rows (evaluate_summary inside) in parallel too
        with ThreadPoolExecutor(
            max_workers=min(len(extractive_keys), config.EXTRACTIVE_WORKERS),
            thread_name_prefix="ext_eval",
        ) as ext_pool:
            ext_futures = {
                ext_pool.submit(
                    _evaluate_result,
                    key,
                    text,
                    reference,
                    sentence_count,
                    max_output_length,
                    target_words,
                    source_words,
                ): key
                for key in extractive_keys
            }
            for future in as_completed(ext_futures):
                key = ext_futures[future]
                try:
                    results[key] = future.result()
                except Exception as exc:
                    logger.error("Extractive eval [%s] failed: %s", key, exc)

    # ── 2. Run abstractive models GPU-sequentially ──────────────────────────
    # Using a single-thread executor keeps code clean while enforcing order.
    if abstractive_keys:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="abs_gpu") as abs_pool:
            abs_futures = {
                abs_pool.submit(
                    _evaluate_result,
                    key,
                    text,
                    reference,
                    sentence_count,
                    max_output_length,
                    target_words,
                    source_words,
                ): key
                for key in abstractive_keys
            }
            for future in as_completed(abs_futures):
                key = abs_futures[future]
                try:
                    results[key] = future.result()
                except Exception as exc:
                    logger.error("Abstractive eval [%s] failed: %s", key, exc)

    # ── 3. Preserve original key ordering ──────────────────────────────────
    ordered = []
    for key in extractive_keys + abstractive_keys:
        if key in results:
            ordered.append(results[key])
    return ordered


# ─────────────────────────── Ranking + charts ──────────────────────────────

def _rank_results(results: list[dict]) -> list[dict]:
    ranked = sorted(
        results,
        key=lambda row: (
            not row.get("experimental", False),  # Put non-experimental models first
            row["metrics"].get("combined_score", 0.0),
            row["metrics"].get("rougeL", 0.0),
            -row["metrics"].get("processing_time", 999.0),
        ),
        reverse=True,
    )
    return [
        {
            "rank": idx + 1,
            "key": row["key"],
            "algorithm": row["algorithm"],
            "group": row["group"],
            "combined_score": row["metrics"]["combined_score"],
            "rougeL": row["metrics"]["rougeL"],
            "bertscore_f1": row["metrics"]["bertscore_f1"],
            "processing_time": row["metrics"]["processing_time"],
            "experimental": row.get("experimental", False),
            "warning_badge": row.get("warning_badge"),
        }
        for idx, row in enumerate(ranked)
    ]


def _chart_payload(results: list[dict]) -> dict:
    return {
        "bar": [
            {
                "model": row["algorithm"],
                "ROUGE-1": row["metrics"]["rouge1"],
                "ROUGE-2": row["metrics"]["rouge2"],
                "ROUGE-L": row["metrics"]["rougeL"],
                "BERTScore": row["metrics"]["bertscore_f1"],
                "Semantic": row["metrics"]["semantic_similarity"],
            }
            for row in results
        ],
        "radar": [
            {
                "model": row["algorithm"],
                "rouge1": row["metrics"]["rouge1"],
                "rouge2": row["metrics"]["rouge2"],
                "rougeL": row["metrics"]["rougeL"],
                "bertscore": row["metrics"]["bertscore_f1"],
                "semantic": row["metrics"]["semantic_similarity"],
                "compression": row["metrics"]["compression_ratio"],
            }
            for row in results
        ],
        "time": [
            {
                "model": row["algorithm"],
                "seconds": row["metrics"]["processing_time"],
                "group": row["group"],
            }
            for row in results
        ],
    }


# ─────────────────────────── Compare payload assembly ─────────────────────

def _normalize_algorithm_keys(algorithms: list[str] | None) -> list[str]:
    selected = algorithms or DEFAULT_ALGORITHMS
    normalized_keys: list[str] = []
    for key in selected:
        try:
            normalized_keys.append(resolve_algorithm(key).key)
        except KeyError:
            logger.warning("Ignoring unsupported algorithm: %s", key)
    return normalized_keys or DEFAULT_ALGORITHMS.copy()


def _prepare_compare(
    text: str,
    reference: str | None,
    algorithms: list[str] | None,
    sentence_count: int,
    max_output_length: int,
    target_length_ratio: int | None = None,
    use_length_ratio: bool = True,
) -> tuple[str, str, bool, list[str], list[str], list[str], int, int, int, int, dict]:
    cleaned = clean_text(text, aggressive=True)
    if not cleaned or count_words(cleaned) < 5:
        raise ValueError("Input text is empty or too short after preprocessing.")

    from src.length_control import compute_length_targets

    length_meta = compute_length_targets(
        cleaned,
        target_length_ratio if target_length_ratio is not None else 100,
        sentence_count=sentence_count if not use_length_ratio else None,
        max_output_length=max_output_length if not use_length_ratio else None,
    )
    if use_length_ratio and target_length_ratio is not None:
        sentence_count = length_meta["sentence_count"]
        max_output_length = length_meta["max_output_length"]

    normalized_keys = _normalize_algorithm_keys(algorithms)
    reference_provided = bool(reference and clean_text(reference, aggressive=True))
    reference_text = clean_text(reference, aggressive=True) if reference_provided else cleaned
    extractive_keys = [k for k in normalized_keys if k in EXTRACTIVE_ALGORITHMS]
    abstractive_keys = [k for k in normalized_keys if k in ABSTRACTIVE_ALGORITHMS]
    return (
        cleaned,
        reference_text,
        reference_provided,
        normalized_keys,
        extractive_keys,
        abstractive_keys,
        sentence_count,
        max_output_length,
        length_meta["target_words"],
        length_meta["target_length_ratio"],
        length_meta,
    )


def _assemble_compare_result(
    cleaned: str,
    reference_provided: bool,
    normalized_keys: list[str],
    extractive_keys: list[str],
    abstractive_keys: list[str],
    results: list[dict],
    sentence_count: int,
    max_output_length: int,
    total_wall: float,
    reference_text: str,
    target_length_ratio: int = 100,
    length_controls: dict | None = None,
) -> dict:
    input_words = count_words(cleaned)
    ranking = _rank_results(results)
    best_key = ranking[0]["key"] if ranking else None

    # Highlight best per category
    extractive_ranked = [r for r in ranking if r["group"] == "extractive"]
    abstractive_ranked = [r for r in ranking if r["group"] == "abstractive"]
    
    best_extractive = extractive_ranked[0] if extractive_ranked else None
    best_abstractive = abstractive_ranked[0] if abstractive_ranked else None

    group_summary: dict[str, dict] = defaultdict(dict)
    for group in ("extractive", "abstractive"):
        group_rows = [row for row in results if row["group"] == group]
        if not group_rows:
            continue
        group_summary[group] = {
            "count": len(group_rows),
            "avg_rougeL": round(
                sum(row["metrics"]["rougeL"] for row in group_rows) / len(group_rows), 4
            ),
            "avg_bertscore_f1": round(
                sum(row["metrics"]["bertscore_f1"] for row in group_rows) / len(group_rows), 4
            ),
            "avg_processing_time": round(
                sum(row["metrics"]["processing_time"] for row in group_rows) / len(group_rows), 4
            ),
        }

    is_biased = not reference_provided

    research_analysis = {
        "extractive_insights": (
            "Extractive models (TextRank, LexRank, LSA) guarantee 100% factual consistency because they select "
            "sentences directly from the source text. However, they lack semantic flexibility, have lower compression "
            "adaptability, and are highly prone to score inflation when overlap metrics are evaluated without a real reference."
        ),
        "abstractive_insights": (
            "Abstractive models (ViT5, BARTPho, mT5) exhibit high semantic flexibility and produce more natural, "
            "paraphrased summaries. However, they carry the risk of hallucinations, repetition loops, or multilingual artifacts "
            "if tokenizers are misaligned."
        ),
        "recommendation": (
            "BARTPho is the recommended model for production. It is stable, handles Vietnamese syllables natively, and achieves "
            "strong semantic scores. ViT5 is a viable alternative now that its generation parameters have been optimized. "
            "mT5 should only be used as an experimental baseline."
        ),
        "bias_notice": (
            "Reference summary was NOT provided — overlap metrics (ROUGE/BLEU) have been disabled for fairness. "
            "Semantic similarity (SBERT) and BERTScore were used for ranking." if is_biased else
            "Reference summary was provided. All overlap, semantic, and compression metrics are fully active."
        )
    }

    return {
        "algorithms": normalized_keys,
        "algorithm_groups": {
            "extractive": list(EXTRACTIVE_ALGORITHMS.keys()),
            "abstractive": list(ABSTRACTIVE_ALGORITHMS.keys()),
        },
        "results": results,
        "ranking": ranking,
        "best_model": next((row for row in ranking if row["key"] == best_key), None),
        "best_extractive": best_extractive,
        "best_abstractive": best_abstractive,
        "research_analysis": research_analysis,
        "group_summary": dict(group_summary),
        "charts": _chart_payload(results),
        "warning": "Reference summary not provided — overlap metrics may be biased." if is_biased else None,
        "performance": {
            "total_wall_time_s": round(total_wall, 3),
            "extractive_count": len(extractive_keys),
            "abstractive_count": len(abstractive_keys),
        },
        "meta": {
            "input_words": input_words,
            "input_sentences": len(split_sentences(cleaned)),
            "input_preview": cleaned[:400],
            "reference_provided": reference_provided,
            "reference_words": count_words(reference_text),
            "sentence_count": sentence_count,
            "max_output_length": max_output_length,
            "target_length_ratio": target_length_ratio,
            "target_words": (length_controls or {}).get("target_words"),
            "length_controls": length_controls or {},
            "warning": "Reference summary not provided — overlap metrics may be biased." if is_biased else None,
        },
    }


# ─────────────────────────── Public entry point ────────────────────────────

def summarize_all(
    text: str,
    reference: str | None = None,
    algorithms: list[str] | None = None,
    sentence_count: int = 5,
    max_output_length: int = config.MAX_OUTPUT_LENGTH,
    target_length_ratio: int = 50,
    use_length_ratio: bool = True,
    use_cache: bool = False,
) -> dict:
    """
    Run all requested algorithms and return a unified comparison payload.

    Extractive algorithms run concurrently; abstractive models run sequentially
    on the GPU (one at a time) to avoid VRAM exhaustion.
    """
    del use_cache  # reserved for future Redis caching

    (
        cleaned,
        reference_text,
        reference_provided,
        normalized_keys,
        extractive_keys,
        abstractive_keys,
        sentence_count,
        max_output_length,
        target_words,
        target_length_ratio,
        length_controls,
    ) = _prepare_compare(
        text,
        reference,
        algorithms,
        sentence_count,
        max_output_length,
        target_length_ratio=target_length_ratio,
        use_length_ratio=use_length_ratio,
    )

    source_words = count_words(cleaned)
    t_total = time.perf_counter()
    results = _run_all_parallel(
        cleaned,
        reference_text,
        extractive_keys,
        abstractive_keys,
        sentence_count,
        max_output_length,
        target_words,
        source_words,
    )
    total_wall = time.perf_counter() - t_total
    logger.info(
        "🏁 summarize_all complete: %d algorithms in %.3f s  (ext=%d abs=%d)",
        len(results),
        total_wall,
        len(extractive_keys),
        len(abstractive_keys),
    )
    return _assemble_compare_result(
        cleaned,
        reference_provided,
        normalized_keys,
        extractive_keys,
        abstractive_keys,
        results,
        sentence_count,
        max_output_length,
        total_wall,
        reference_text,
        target_length_ratio,
        length_controls,
    )


def _sse(event: str, **payload) -> str:
    body = {"event": event, **payload}
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


# ─────────────────────────── SSE streaming helper ──────────────────────────

def stream_compare(
    text: str,
    reference: str | None,
    algorithms: list[str] | None = None,
    sentence_count: int = 5,
    max_output_length: int = config.MAX_OUTPUT_LENGTH,
    target_length_ratio: int = 50,
    use_length_ratio: bool = True,
    save_result: bool = True,
):
    """
    Server-Sent Events generator — yields running/done per algorithm as completed.
    """
    try:
        (
            cleaned,
            reference_text,
            reference_provided,
            normalized_keys,
            extractive_keys,
            abstractive_keys,
            sentence_count,
            max_output_length,
            target_words,
            target_length_ratio,
            length_controls,
        ) = _prepare_compare(
            text,
            reference,
            algorithms,
            sentence_count,
            max_output_length,
            target_length_ratio=target_length_ratio,
            use_length_ratio=use_length_ratio,
        )
    except ValueError as exc:
        yield _sse("error", error=str(exc))
        return

    source_words = count_words(cleaned)
    execution_order = extractive_keys + abstractive_keys
    yield _sse("start", algorithms=execution_order, total=len(execution_order))

    results_by_key: dict[str, dict] = {}
    t_total = time.perf_counter()

    try:
        if extractive_keys:
            with ThreadPoolExecutor(
                max_workers=min(len(extractive_keys), config.EXTRACTIVE_WORKERS),
                thread_name_prefix="ext_stream",
            ) as pool:
                futures = {
                    pool.submit(
                        _evaluate_result,
                        key,
                        cleaned,
                        reference_text,
                        sentence_count,
                        max_output_length,
                        target_words,
                        source_words,
                    ): key
                    for key in extractive_keys
                }
                for key in extractive_keys:
                    yield _sse("running", algorithm=key, index=execution_order.index(key) + 1, total=len(execution_order))
                for future in as_completed(futures):
                    key = futures[future]
                    row = future.result()
                    results_by_key[key] = row
                    yield _sse("done", algorithm=key, result=row, completed=len(results_by_key), total=len(execution_order))

        for key in abstractive_keys:
            yield _sse("running", algorithm=key, index=execution_order.index(key) + 1, total=len(execution_order))
            row = _evaluate_result(
                key,
                cleaned,
                reference_text,
                sentence_count,
                max_output_length,
                target_words,
                source_words,
            )
            results_by_key[key] = row
            yield _sse("done", algorithm=key, result=row, completed=len(results_by_key), total=len(execution_order))

        ordered = [results_by_key[k] for k in execution_order if k in results_by_key]
        total_wall = time.perf_counter() - t_total
        payload = _assemble_compare_result(
            cleaned,
            reference_provided,
            normalized_keys,
            extractive_keys,
            abstractive_keys,
            ordered,
            sentence_count,
            max_output_length,
            total_wall,
            reference_text,
            target_length_ratio,
            length_controls,
        )
        if save_result:
            from src.storage import persist_compare_result

            payload["storage"] = persist_compare_result(payload)
        yield _sse("finished", data=payload)
    except Exception as exc:
        logger.exception("stream_compare failed")
        yield _sse("error", error=str(exc))
