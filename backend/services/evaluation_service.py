"""
evaluation_service.py — Shared evaluation pipeline for Compare & Summarize pages.

Uses the same functions and formulas as the research benchmark
(evaluation/metrics.py: ROUGE, BERTScore, latency).
"""

from __future__ import annotations

from src import config
from evaluation.metrics import evaluate_summary

# 15 algorithms supported by both Compare leaderboard and Summarize playground.
COMPARE_ALGORITHM_KEYS: tuple[str, ...] = (
    "textrank", "lexrank", "lsa",
    "vit5", "mt5", "bartpho",
    "textrank-vit5", "lexrank-vit5", "lsa-vit5",
    "textrank-mt5", "lexrank-mt5", "lsa-mt5",
    "textrank-bartpho", "lexrank-bartpho", "lsa-bartpho",
)

COMPARE_METRIC_KEYS: tuple[str, ...] = (
    "rouge1", "rouge2", "rougeL", "rougeLsum",
    "bert_p", "bert_r", "bertscore", "latency",
)


def format_compare_metrics(full_metrics: dict) -> dict:
    """Map evaluate_summary output to Compare-page column schema."""
    bert = full_metrics.get("bertscore") or {}
    if isinstance(bert, dict):
        f1 = float(full_metrics.get("bertscore_f1", bert.get("f1", 0.0)))
        bert_p = float(bert.get("precision", 0.0))
        bert_r = float(bert.get("recall", 0.0))
    else:
        f1 = float(full_metrics.get("bertscore_f1", full_metrics.get("bert_f1", bert or 0.0)))
        bert_p = float(full_metrics.get("bert_p", 0.0))
        bert_r = float(full_metrics.get("bert_r", 0.0))
    latency = float(full_metrics.get("processing_time", 0.0))
    bertscore_status = full_metrics.get("bertscore_status", "ok")
    bertscore_error = full_metrics.get("bertscore_error")
    warning = full_metrics.get("warning")
    if bertscore_status in {"timeout", "error"} and bertscore_error:
        warning = warning or bertscore_error
    return {
        "rouge1": float(full_metrics.get("rouge1", 0.0)),
        "rouge2": float(full_metrics.get("rouge2", 0.0)),
        "rougeL": float(full_metrics.get("rougeL", 0.0)),
        "rougeLsum": float(full_metrics.get("rougeLsum", full_metrics.get("rougeL", 0.0))),
        "bert_p": bert_p,
        "bert_r": bert_r,
        "bertscore": f1,
        "bert_f1": f1,
        "latency": round(latency, 4),
        "processing_time": round(latency, 4),
        # Kept for ranking / charts — not shown on Compare-style table
        "bertscore_f1": f1,
        "bertscore_status": bertscore_status,
        "bertscore_error": bertscore_error,
        "semantic_similarity": float(full_metrics.get("semantic_similarity", 0.0)),
        "compression_ratio": float(full_metrics.get("compression_ratio", 0.0)),
        "faithfulness": float(full_metrics.get("faithfulness", 0.0)),
        "coverage": float(full_metrics.get("coverage", 0.0)),
        "composite_score": float(full_metrics.get("composite_score", 0.0)),
        "bleu": float(full_metrics.get("bleu", 0.0)),
        "is_biased": bool(full_metrics.get("is_biased", False)),
        "warning": warning,
    }


class EvaluationService:
    """Single evaluation entry point shared by Summarize and Compare pipelines."""

    @staticmethod
    def evaluate(
        prediction: str,
        reference: str,
        source_text: str,
        latency: float,
        timeout: float | None = None,
    ) -> dict:
        """
        Compute Compare-aligned metrics after each summarization.

        Reference handling matches dashboard_service._prepare_compare:
        when no human reference is supplied the caller passes source text;
        evaluate_summary disables overlap metrics unless ALLOW_SOURCE_AS_REFERENCE.
        """
        timeout = config.HEAVY_METRICS_TIMEOUT if timeout is None else timeout
        full = evaluate_summary(
            prediction=prediction,
            reference=reference,
            source_text=source_text,
            processing_time=latency,
            timeout=timeout,
        )
        return format_compare_metrics(full)

    @staticmethod
    def evaluate_batch_rows(
        rows: list[dict],
    ) -> list[dict]:
        """Evaluate multiple {prediction, reference, source_text, latency} rows."""
        return [
            EvaluationService.evaluate(
                row["prediction"],
                row["reference"],
                row.get("source_text", ""),
                row.get("latency", row.get("processing_time", 0.0)),
            )
            for row in rows
        ]


def evaluate_compare_metrics(
    prediction: str,
    reference: str,
    source_text: str,
    latency: float,
    timeout: float | None = None,
) -> dict:
    """Functional alias used by dashboard_service and tests."""
    return EvaluationService.evaluate(prediction, reference, source_text, latency, timeout)


def bertscore_detail(metrics: dict) -> dict[str, float]:
    """Backward-compatible BERTScore dict for API consumers."""
    return {
        "precision": float(metrics.get("bert_p", 0.0)),
        "recall": float(metrics.get("bert_r", 0.0)),
        "f1": float(metrics.get("bertscore", metrics.get("bertscore_f1", 0.0))),
    }
