"""dashboard.py — Optimized multi-algorithm comparison orchestrator."""

from __future__ import annotations

import json
import os
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
from backend.services.evaluation_service import evaluate_compare_metrics, bertscore_detail
from src.extractive import summarize_extractive_algorithm, summarize_extractive_parallel
from src.model_registry import (
    ABSTRACTIVE_ALGORITHMS,
    DEFAULT_ALGORITHMS,
    EXTRACTIVE_ALGORITHMS,
    HYBRID_ALGORITHMS,
    resolve_algorithm,
)
from src.preprocess import clean_text, split_sentences
from src.utils import count_words, log_vram_usage, logger


_GPU_LOCK = threading.Semaphore(config.MAX_GPU_CONCURRENT)


def _combined_score(metrics: dict) -> float:
    compression = float(metrics.get("compression_ratio", 0.0))
    compression_score = max(0.0, 1.0 - abs(compression - 0.25) / 0.25)
    
    is_biased = metrics.get("is_biased", False)
    if is_biased:
        weights = config.NO_REFERENCE_RANKING_WEIGHTS
        score = (
            weights.get("bertscore_f1", 0.45) * float(metrics.get("bertscore_f1", 0.0))
            + weights.get("semantic_similarity", 0.35) * float(metrics.get("semantic_similarity", 0.0))
            + weights.get("compression_score", 0.20) * compression_score
        )
    else:
        weights = config.WITH_REFERENCE_RANKING_WEIGHTS
        score = (
            weights.get("rougeL", 0.25) * float(metrics.get("rougeL", 0.0))
            + weights.get("rouge2", 0.15) * float(metrics.get("rouge2", 0.0))
            + weights.get("bertscore_f1", 0.30) * float(metrics.get("bertscore_f1", 0.0))
            + weights.get("semantic_similarity", 0.20) * float(metrics.get("semantic_similarity", 0.0))
            + weights.get("compression_score", 0.10) * compression_score
        )
    return round(score, 4)


def _run_extractive(
    text: str,
    key: str,
    sentence_count: int,
    details: dict | None = None,
) -> tuple[str, dict]:
    from src.explainability import build_extractive_explanations, build_sentence_ranking_graph

    if details is None:
        details = summarize_extractive_algorithm(text, key, sentence_count=sentence_count)
    summary = details.get("summary", "")
    ranking_graph = build_sentence_ranking_graph(text, algorithm=key)
    evidence = build_extractive_explanations(text, summary)
    return summary, {
        "extractive": {
            "source_sentences": details.get("source_sentences", []),
            "selected_sentences": details.get("selected_sentences", []),
            "highlighted_sentence_indexes": details.get("highlighted_sentence_indexes", []),
            "ranking_graph": ranking_graph,
            "evidence": evidence,
        }
    }


def _fallback_summary(text: str, sentence_count: int) -> str:
    return summarize_extractive_algorithm(text, "textrank", sentence_count=sentence_count).get("summary", "")


def _clean_incomplete_sentence(text: str) -> str:
    """
    Tự động tìm kiếm dấu chấm câu cuối cùng trong văn bản tóm tắt sinh ra 
    và loại bỏ phần chữ thừa bị dở dang phía sau dấu chấm đó do chạm trần token.
    """
    text = (text or "").strip()
    if not text:
        return ""
    import re
    # Kiểm tra xem chuỗi đã kết thúc bằng một dấu chấm câu chuẩn (. ! ? … ” ") hay chưa
    if re.search(r'[.!?…]["”]?\s*$', text):
        return text
    # Tìm kiếm tất cả các vị trí kết thúc câu trong chuỗi
    ends = list(re.finditer(r'[.!?…]["”]?', text))
    if not ends:
        return text
    # Cắt đến dấu kết thúc câu cuối cùng
    return text[:ends[-1].end()].strip()
def _run_abstractive(
    text: str,
    key: str,
    max_output_length: int,
    sentence_count: int,
    min_output_length: int | None = None,
) -> tuple[str, dict]:
    log_vram_usage(f"before_{key}")
    with _GPU_LOCK:
        summarizer = get_summarizer(model_name=key)
        summary = summarizer.summarize(
            text, 
            max_output_length=max_output_length,
            min_output_length=min_output_length,
        )
        if summary:
            summary = _clean_incomplete_sentence(summary)

    fallback_used = False
    training_quality: dict = {"is_poor_training": False, "reason": None}

    if summary:
        from src.output_validator import is_garbled_abstractive, validate_output
        from evaluation.output_validator import detect_poor_training_output

        training_quality = detect_poor_training_output(summary)
        validation = validate_output(
            summary,
            require_vietnamese=key in {"vit5", "mt5"},
        )
        if validation["is_corrupted"] or is_garbled_abstractive(summary):
            logger.warning(
                "[%s] Bad/corrupted output detected (%s) — returning raw output for scientific comparison.",
                key,
                validation.get("quality_warning") or "garbled",
            )
        elif training_quality["is_poor_training"]:
            logger.warning(
                "[%s] Poor training quality detected: %s — returning raw output for scientific comparison.",
                key,
                training_quality["reason"],
            )

    if not summary:
        logger.warning("[%s] Empty abstractive generation", key)
        summary = ""

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
        },
        "training_quality": training_quality,
    }


def _metrics_warning_badge(metrics: dict, existing: str | None = None) -> str | None:
    """Surface evaluation warnings on algorithm cards."""
    if existing:
        return existing
    if metrics.get("bertscore_status") in {"timeout", "error"}:
        return metrics.get("bertscore_error") or "BERTScore không khả dụng"
    if metrics.get("is_biased"):
        return metrics.get("warning") or "Chưa có tóm tắt tham chiếu — ROUGE có thể = 0"
    return metrics.get("warning")


def _evaluate_result(
    key: str,
    text: str,
    reference: str,
    sentence_count: int,
    max_output_length: int,
    target_words: int | None = None,
    source_words: int | None = None,
    summary_length: str = "auto",
    *,
    extractive_details: dict | None = None,
    source_sentences: list[str] | None = None,
) -> dict:
    algorithm = resolve_algorithm(key)
    start = time.perf_counter()
    error: str | None = None
    explainability: dict = {}

    try:
        from src.utils import count_words
        if count_words(text) > 10000:
            from src.length_control import SummaryLengthManager
            group = "extractive" if algorithm.key in EXTRACTIVE_ALGORITHMS else "abstractive"
            summary = SummaryLengthManager.hierarchical_summarize_pipeline(
                text, algorithm.key, summary_length, group
            )
            explainability = {"hierarchical": True}
            training_quality = {"is_poor_training": False, "reason": None}
        else:
            if algorithm.group == "extractive":
                summary, explainability = _run_extractive(
                    text,
                    algorithm.key,
                    sentence_count,
                    details=extractive_details,
                )
                training_quality = {"is_poor_training": False, "reason": None}
            elif algorithm.group == "abstractive":
                from src.length_control import SummaryLengthManager
                analysis = SummaryLengthManager.analyze_input(text)
                min_tokens, _ = SummaryLengthManager.get_abstractive_limits(algorithm.key, summary_length, analysis)
                summary, explainability = _run_abstractive(
                    text, algorithm.key, max_output_length, sentence_count, min_output_length=min_tokens
                )
                training_quality = explainability.pop("training_quality", {"is_poor_training": False, "reason": None})
            elif algorithm.group == "hybrid":
                from pipeline.hybrid_summarizer import HybridSummarizer
                parts = algorithm.key.split("-")
                ext_algo = parts[0]
                abs_algo = parts[1]
                hybrid = HybridSummarizer(abstractive_model_key=abs_algo)
                summary = hybrid.summarize(
                    text,
                    extractive_algo=ext_algo,
                    max_target_tokens=max_output_length,
                    compression_ratio=0.35
                )
                explainability = {"hybrid": True, "extractive_algo": ext_algo, "abstractive_algo": abs_algo}
                training_quality = {"is_poor_training": False, "reason": None}
            else:
                raise KeyError(f"Unsupported algorithm: {algorithm.key} or group {algorithm.group}")
    except Exception as exc:
        logger.exception("Algorithm %s failed", algorithm.key)
        error = str(exc)
        summary = _fallback_summary(text, sentence_count)
        explainability = {"error": error, "fallback_used": True}
        training_quality = {"is_poor_training": False, "reason": None}

    if target_words and summary:
        from src.length_control import trim_summary_to_word_budget

        summary = trim_summary_to_word_budget(summary, target_words)

    duration = time.perf_counter() - start
    logger.info(
        "⏱  [%s] inference done in %.3f s  words_out=%d",
        algorithm.key, duration, count_words(summary),
    )

    metrics = evaluate_compare_metrics(summary, reference, text, duration)
    metrics["combined_score"] = _combined_score(metrics)

    src_w = source_words or count_words(text)
    from src.length_control import length_ratio_percent

    actual_length_ratio = length_ratio_percent(count_words(summary), src_w)

    is_experimental = False
    warning_badge = _metrics_warning_badge(metrics)
    if algorithm.key == "mt5" and config.MT5_EXPERIMENTAL:
        is_experimental = True
        warning_badge = "Experimental Baseline"
        if summary:
            from src.output_validator import is_multilingual_garbage

            local_dir = Path(algorithm.local_dir or "")
            has_finetuned = local_dir.exists() and any(local_dir.iterdir())
            if is_multilingual_garbage(summary, require_vietnamese=not has_finetuned):
                warning_badge = "Experimental (Corrupt Multilingual)"
            elif training_quality.get("is_poor_training"):
                warning_badge = "⚠️ Poor Training Quality"
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
        "bertscore": bertscore_detail(metrics),
        "semantic_similarity": metrics["semantic_similarity"],
        "compression_ratio": metrics["compression_ratio"],
        "time_seconds": metrics["processing_time"],
        "processing_time": metrics["processing_time"],
        "bertscore_status": metrics.get("bertscore_status"),
        "bertscore_error": metrics.get("bertscore_error"),
        "explainability": explainability,
        "experimental": is_experimental,
        "warning_badge": warning_badge,
        "training_quality": training_quality,
        "details": (
            explainability.get("extractive")
            or explainability.get("abstractive")
            or explainability
        ),
        "source_sentences": source_sentences if source_sentences is not None else split_sentences(text)[:200],
        "error": error,
    }


def _group_hybrid_by_backbone(hybrid_keys: list[str]) -> dict[str, list[str]]:
    """Nhóm hybrid keys theo backbone abstractive (vit5, mt5, bartpho)."""
    groups: dict[str, list[str]] = {}
    for key in hybrid_keys:
        if key not in HYBRID_ALGORITHMS:
            continue
        backbone = key.split("-", 1)[1]
        groups.setdefault(backbone, []).append(key)
    return groups


def _evaluate_hybrid_shared(
    text: str,
    reference: str,
    hybrid_keys: list[str],
    max_output_length: int,
    target_words: int | None,
    source_words: int | None,
    compression_ratio: float = 0.35,
    source_sentences: list[str] | None = None,
) -> list[dict]:
    """Chạy nhóm hybrid cùng backbone: extractive song song, 1 load model abstractive."""
    from pipeline.hybrid_summarizer import HybridSummarizer

    if not hybrid_keys:
        return []

    backbone = hybrid_keys[0].split("-", 1)[1]
    hybrid_engine = HybridSummarizer(abstractive_model_key=backbone)
    condensed_by_key: dict[str, str] = {}

    with ThreadPoolExecutor(
        max_workers=min(len(hybrid_keys), config.EXTRACTIVE_WORKERS),
        thread_name_prefix="hyb_ext",
    ) as ext_pool:
        futures = {
            ext_pool.submit(
                hybrid_engine.build_condensed_context,
                text,
                key.split("-", 1)[0],
                compression_ratio,
            ): key
            for key in hybrid_keys
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                condensed_by_key[key] = future.result()
            except Exception as exc:
                logger.error("Hybrid condensed [%s] failed: %s", key, exc)
                condensed_by_key[key] = ""

    results: list[dict] = []
    for key in hybrid_keys:
        algorithm = resolve_algorithm(key)
        ext_algo, abs_algo = key.split("-", 1)
        start = time.perf_counter()
        error: str | None = None
        try:
            with _GPU_LOCK:
                summary = hybrid_engine.summarize_from_condensed(
                    condensed_by_key.get(key, ""),
                    max_target_tokens=max_output_length,
                )
            if target_words and summary:
                from src.length_control import trim_summary_to_word_budget
                summary = trim_summary_to_word_budget(summary, target_words)
            explainability = {
                "hybrid": True,
                "extractive_algo": ext_algo,
                "abstractive_algo": abs_algo,
                "shared_backbone": True,
            }
            training_quality = {"is_poor_training": False, "reason": None}
        except Exception as exc:
            logger.exception("Hybrid shared [%s] failed", key)
            error = str(exc)
            summary = _fallback_summary(text, 5)
            explainability = {"error": error, "fallback_used": True, "shared_backbone": True}
            training_quality = {"is_poor_training": False, "reason": None}

        duration = time.perf_counter() - start
        metrics = evaluate_compare_metrics(summary, reference, text, duration)
        metrics["combined_score"] = _combined_score(metrics)
        src_w = source_words or count_words(text)
        from src.length_control import length_ratio_percent
        actual_length_ratio = length_ratio_percent(count_words(summary), src_w)
        results.append({
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
            "bertscore": bertscore_detail(metrics),
            "semantic_similarity": metrics["semantic_similarity"],
            "compression_ratio": metrics["compression_ratio"],
            "time_seconds": metrics["processing_time"],
            "processing_time": metrics["processing_time"],
            "bertscore_status": metrics.get("bertscore_status"),
            "bertscore_error": metrics.get("bertscore_error"),
            "explainability": explainability,
            "experimental": False,
            "warning_badge": _metrics_warning_badge(metrics),
            "training_quality": training_quality,
            "details": explainability,
            "source_sentences": source_sentences if source_sentences is not None else split_sentences(text)[:200],
            "error": error,
        })
    return results


def _shared_source_sentences(
    text: str,
    ext_parallel: dict[str, dict] | None = None,
) -> list[str]:
    if ext_parallel:
        for details in ext_parallel.values():
            sentences = details.get("source_sentences")
            if sentences:
                return list(sentences)[:200]
    return split_sentences(text)[:200]


def _run_all_parallel(
    text: str,
    reference: str,
    extractive_keys: list[str],
    abstractive_keys: list[str],
    sentence_count: int,
    max_output_length: int,
    target_words: int,
    source_words: int,
    summary_length: str = "auto",
) -> list[dict]:
    results: dict[str, dict] = {}
    shared_source_sentences = _shared_source_sentences(text)
    ext_parallel: dict[str, dict] = {}

    if extractive_keys:
        ext_parallel = summarize_extractive_parallel(text, extractive_keys, sentence_count)
        shared_source_sentences = _shared_source_sentences(text, ext_parallel)
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
                    summary_length,
                    extractive_details=ext_parallel.get(key),
                    source_sentences=shared_source_sentences,
                ): key
                for key in extractive_keys
            }
            for future in as_completed(ext_futures):
                key = ext_futures[future]
                try:
                    results[key] = future.result()
                except Exception as exc:
                    logger.error("Extractive eval [%s] failed: %s", key, exc)

    if abstractive_keys:
        use_shared_hybrid = os.environ.get("COMPARE_PIPELINE_V2", "1") != "0"
        pure_abs = [k for k in abstractive_keys if k in ABSTRACTIVE_ALGORITHMS]
        hybrid_keys = [k for k in abstractive_keys if k in HYBRID_ALGORITHMS]

        if pure_abs:
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
                        summary_length,
                        source_sentences=shared_source_sentences,
                    ): key
                    for key in pure_abs
                }
                for future in as_completed(abs_futures):
                    key = abs_futures[future]
                    try:
                        results[key] = future.result()
                    except Exception as exc:
                        logger.error("Abstractive eval [%s] failed: %s", key, exc)

        if hybrid_keys:
            if use_shared_hybrid:
                for backbone, group_keys in _group_hybrid_by_backbone(hybrid_keys).items():
                    logger.info("Hybrid shared backbone batch: %s (%d algos)", backbone, len(group_keys))
                    for row in _evaluate_hybrid_shared(
                        text,
                        reference,
                        group_keys,
                        max_output_length,
                        target_words,
                        source_words,
                        source_sentences=shared_source_sentences,
                    ):
                        results[row["key"]] = row
            else:
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="hyb_gpu") as hyb_pool:
                    hyb_futures = {
                        hyb_pool.submit(
                            _evaluate_result,
                            key,
                            text,
                            reference,
                            sentence_count,
                            max_output_length,
                            target_words,
                            source_words,
                            summary_length,
                        ): key
                        for key in hybrid_keys
                    }
                    for future in as_completed(hyb_futures):
                        key = hyb_futures[future]
                        try:
                            results[key] = future.result()
                        except Exception as exc:
                            logger.error("Hybrid eval [%s] failed: %s", key, exc)

    ordered = []
    for key in extractive_keys + abstractive_keys:
        if key in results:
            ordered.append(results[key])
    return ordered


def _rank_results(results: list[dict]) -> list[dict]:
    ranked = sorted(
        results,
        key=lambda row: (
            not row.get("experimental", False),
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
    summary_length: str = "auto",
) -> tuple[str, str, bool, list[str], list[str], list[str], int, int, int, int, dict]:
    cleaned = clean_text(text, aggressive=True)
    if not cleaned or count_words(cleaned) < 5:
        raise ValueError("Input text is empty or too short after preprocessing.")

    from src.length_control import SummaryLengthManager

    analysis = SummaryLengthManager.analyze_input(cleaned)
    resolved_sentence_count = SummaryLengthManager.get_extractive_sentences(summary_length, analysis)
    min_tokens, resolved_max_output = SummaryLengthManager.get_abstractive_limits("", summary_length, analysis)

    from src.length_control import compute_length_targets

    length_meta = compute_length_targets(
        cleaned,
        target_length_ratio if target_length_ratio is not None else 20,
        sentence_count=resolved_sentence_count,
        max_output_length=resolved_max_output,
    )
    
    # Override settings with SummaryLengthManager resolved values
    sentence_count = resolved_sentence_count
    max_output_length = resolved_max_output
    
    length_meta["summary_length"] = summary_length
    length_meta["analysis"] = analysis
    length_meta["min_output_length"] = min_tokens
    length_meta["is_extremely_long"] = analysis["is_extremely_long"]

    normalized_keys = _normalize_algorithm_keys(algorithms)
    reference_provided = bool(reference and clean_text(reference, aggressive=True))
    reference_text = clean_text(reference, aggressive=True) if reference_provided else cleaned
    extractive_keys = [k for k in normalized_keys if k in EXTRACTIVE_ALGORITHMS]
    abstractive_keys = [k for k in normalized_keys if k in ABSTRACTIVE_ALGORITHMS or k in HYBRID_ALGORITHMS]
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
    target_length_ratio: int = 20,
    length_controls: dict | None = None,
) -> dict:
    input_words = count_words(cleaned)
    ranking = _rank_results(results)
    best_key = ranking[0]["key"] if ranking else None

    extractive_ranked = [r for r in ranking if r["group"] == "extractive"]
    abstractive_ranked = [r for r in ranking if r["group"] == "abstractive"]
    
    best_extractive = extractive_ranked[0] if extractive_ranked else None
    best_abstractive = abstractive_ranked[0] if abstractive_ranked else None

    group_summary: dict[str, dict] = defaultdict(dict)
    for group in ("extractive", "abstractive", "hybrid"):
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
            "hybrid": list(HYBRID_ALGORITHMS.keys()),
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
            "full_text": cleaned,
            "reference_text": reference_text if reference_provided else None,
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


def summarize_all(
    text: str,
    reference: str | None = None,
    algorithms: list[str] | None = None,
    sentence_count: int = 5,
    max_output_length: int = config.MAX_OUTPUT_LENGTH,
    target_length_ratio: int = 20,
    use_length_ratio: bool = True,
    use_cache: bool = False,
    summary_length: str = "auto",
) -> dict:
    del use_cache

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
        summary_length=summary_length,
    )

    source_words = count_words(cleaned)
    from evaluation.metrics import warmup_compare_evaluation

    warmup_compare_evaluation(reference_text, cleaned)
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
        summary_length=summary_length,
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


def stream_compare(
    text: str,
    reference: str | None,
    algorithms: list[str] | None = None,
    sentence_count: int = 5,
    max_output_length: int = config.MAX_OUTPUT_LENGTH,
    target_length_ratio: int = 20,
    use_length_ratio: bool = True,
    save_result: bool = True,
    summary_length: str = "auto",
):
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
            summary_length=summary_length,
        )
    except ValueError as exc:
        yield _sse("error", error=str(exc))
        return

    source_words = count_words(cleaned)
    from evaluation.metrics import warmup_compare_evaluation

    warmup_compare_evaluation(reference_text, cleaned)
    execution_order = extractive_keys + abstractive_keys
    yield _sse("start", algorithms=execution_order, total=len(execution_order))

    results_by_key: dict[str, dict] = {}
    t_total = time.perf_counter()
    shared_source_sentences = _shared_source_sentences(cleaned)
    ext_parallel: dict[str, dict] = {}

    try:
        if extractive_keys:
            ext_parallel = summarize_extractive_parallel(cleaned, extractive_keys, sentence_count)
            shared_source_sentences = _shared_source_sentences(cleaned, ext_parallel)
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
                        summary_length,
                        extractive_details=ext_parallel.get(key),
                        source_sentences=shared_source_sentences,
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

        use_shared_hybrid = os.environ.get("COMPARE_PIPELINE_V2", "1") != "0"
        pure_abs = [k for k in abstractive_keys if k in ABSTRACTIVE_ALGORITHMS]
        hybrid_keys = [k for k in abstractive_keys if k in HYBRID_ALGORITHMS]

        for key in pure_abs:
            yield _sse("running", algorithm=key, index=execution_order.index(key) + 1, total=len(execution_order))
            row = _evaluate_result(
                key,
                cleaned,
                reference_text,
                sentence_count,
                max_output_length,
                target_words,
                source_words,
                summary_length,
                source_sentences=shared_source_sentences,
            )
            results_by_key[key] = row
            yield _sse("done", algorithm=key, result=row, completed=len(results_by_key), total=len(execution_order))

        if hybrid_keys:
            if use_shared_hybrid:
                for backbone, group_keys in _group_hybrid_by_backbone(hybrid_keys).items():
                    logger.info("Hybrid shared backbone batch (stream): %s (%d algos)", backbone, len(group_keys))
                    for key in group_keys:
                        yield _sse(
                            "running",
                            algorithm=key,
                            index=execution_order.index(key) + 1,
                            total=len(execution_order),
                        )
                    for row in _evaluate_hybrid_shared(
                        cleaned,
                        reference_text,
                        group_keys,
                        max_output_length,
                        target_words,
                        source_words,
                        source_sentences=shared_source_sentences,
                    ):
                        results_by_key[row["key"]] = row
                        yield _sse(
                            "done",
                            algorithm=row["key"],
                            result=row,
                            completed=len(results_by_key),
                            total=len(execution_order),
                        )
            else:
                for key in hybrid_keys:
                    yield _sse("running", algorithm=key, index=execution_order.index(key) + 1, total=len(execution_order))
                    row = _evaluate_result(
                        key,
                        cleaned,
                        reference_text,
                        sentence_count,
                        max_output_length,
                        target_words,
                        source_words,
                        summary_length,
                        source_sentences=shared_source_sentences,
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
