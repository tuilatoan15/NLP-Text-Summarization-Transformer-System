"""Research evaluation metrics and hallucination checks."""

from evaluation.metrics import evaluate_pair, evaluate_batch, aggregate_rows
from evaluation.readability import readability_scores
from evaluation.hallucination import audit_summary

__all__ = [
    "evaluate_pair",
    "evaluate_batch",
    "aggregate_rows",
    "readability_scores",
    "audit_summary",
]
