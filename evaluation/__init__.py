"""Research evaluation metrics and hallucination checks."""

from evaluation.metrics import (
    evaluate_pair,
    evaluate_batch,
    aggregate_rows,
    compute_composite_score,
    compute_coverage_score,
    compute_info_retention,
    compute_faithfulness_score,
)
from evaluation.readability import readability_scores
from evaluation.hallucination import audit_summary

__all__ = [
    "evaluate_pair",
    "evaluate_batch",
    "aggregate_rows",
    "compute_composite_score",
    "compute_coverage_score",
    "compute_info_retention",
    "compute_faithfulness_score",
    "readability_scores",
    "audit_summary",
]

