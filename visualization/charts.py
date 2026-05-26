"""Chart payloads for frontend Recharts consumption."""

from __future__ import annotations

from typing import Any


def build_metric_radar(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ["rougeL", "bertscore_f1", "semantic_similarity", "compression_ratio"]
    radar = []
    for row in rows:
        m = row.get("metrics") or row
        radar.append(
            {
                "model": row.get("algorithm") or row.get("key"),
                **{k: float(m.get(k, 0.0) or 0.0) for k in metrics},
            }
        )
    return radar


def build_comparison_charts(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    time_chart = []
    score_chart = []
    for row in results:
        metrics = row.get("metrics") or {}
        label = row.get("algorithm") or row.get("key")
        time_chart.append({"model": label, "seconds": float(metrics.get("processing_time", 0.0))})
        score_chart.append(
            {
                "model": label,
                "rougeL": float(metrics.get("rougeL", 0.0)),
                "bertscore": float(metrics.get("bertscore_f1", 0.0)),
                "semantic": float(metrics.get("semantic_similarity", 0.0)),
            }
        )
    return {"time": time_chart, "scores": score_chart, "radar": build_metric_radar(results)}
