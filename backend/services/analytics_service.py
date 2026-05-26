"""analytics.py — Aggregate persisted comparison runs for the dashboard UI."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from src.storage import RESULT_DIR


def _load_all_results() -> list[dict]:
    if not RESULT_DIR.exists():
        return []
    files = sorted(RESULT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    results: list[dict] = []
    for path in files:
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return results


def _parse_created_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _filter_by_range(items: list[dict], time_range: str) -> list[dict]:
    if time_range in ("all", ""):
        return items
    days = {"7d": 7, "30d": 30, "90d": 90}.get(time_range, 30)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered = []
    for item in items:
        created = _parse_created_at(item.get("created_at"))
        if created is None or created >= cutoff:
            filtered.append(item)
    return filtered


def _algorithm_metrics_from_row(alg: dict) -> dict:
    metrics = alg.get("metrics") or {}
    rouge = alg.get("rouge") or {}
    bert = alg.get("bertscore") or {}
    return {
        "rouge1": float(metrics.get("rouge1") or rouge.get("rouge1") or 0.0),
        "rouge2": float(metrics.get("rouge2") or rouge.get("rouge2") or 0.0),
        "rougeL": float(metrics.get("rougeL") or rouge.get("rougeL") or 0.0),
        "bertscore_f1": float(
            metrics.get("bertscore_f1") or bert.get("f1") or alg.get("bertscore_f1") or 0.0
        ),
        "processing_time": float(
            metrics.get("processing_time") or alg.get("processing_time") or alg.get("time_seconds") or 0.0
        ),
        "length_ratio_percent": float(alg.get("length_ratio_percent") or 0.0),
    }


def list_recent_results(limit: int = 20) -> list[dict]:
    items = _load_all_results()
    out: list[dict] = []
    for record in items[-limit:][::-1]:
        meta = record.get("meta") or {}
        best = record.get("best_model") or {}
        if not best and record.get("ranking"):
            best = record["ranking"][0]
        out.append(
            {
                "result_id": record.get("result_id"),
                "created_at": record.get("created_at"),
                "type": record.get("type", "compare"),
                "input_words": meta.get("input_words", 0),
                "target_length_ratio": meta.get("target_length_ratio"),
                "target_words": meta.get("target_words"),
                "algorithm_count": len(record.get("results") or []),
                "best_algorithm": best.get("algorithm"),
                "best_score": best.get("combined_score"),
                "text_preview": record.get("text_preview") or meta.get("input_preview", ""),
                "processing_time_seconds": record.get("processing_time_seconds")
                or (record.get("performance") or {}).get("total_wall_time_s"),
                "storage_path": str(RESULT_DIR / f"{record.get('result_id')}.json")
                if record.get("result_id")
                else None,
            }
        )
    return out


def compute_dashboard_metrics(time_range: str = "30d") -> dict:
    items = _filter_by_range(_load_all_results(), time_range)
    if not items:
        return {
            "total_runs": 0,
            "total_algorithm_outputs": 0,
            "total_processing_time_seconds": 0.0,
            "avg_target_length_ratio": 0.0,
            "avg_actual_length_ratio": 0.0,
            "top_models": [],
            "avg_rouge": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
            "avg_bertscore_f1": 0.0,
        }

    model_counts: dict[str, int] = defaultdict(int)
    rouge1_list: list[float] = []
    rouge2_list: list[float] = []
    rougeL_list: list[float] = []
    bert_list: list[float] = []
    processing_times: list[float] = []
    target_ratios: list[float] = []
    actual_ratios: list[float] = []
    total_outputs = 0

    for record in items:
        pt = record.get("processing_time_seconds") or (record.get("performance") or {}).get(
            "total_wall_time_s"
        )
        if pt:
            processing_times.append(float(pt))

        meta = record.get("meta") or {}
        if meta.get("target_length_ratio") is not None:
            target_ratios.append(float(meta["target_length_ratio"]))

        for alg in record.get("results") or []:
            total_outputs += 1
            name = alg.get("algorithm") or alg.get("key") or "unknown"
            model_counts[name] += 1
            m = _algorithm_metrics_from_row(alg)
            rouge1_list.append(m["rouge1"])
            rouge2_list.append(m["rouge2"])
            rougeL_list.append(m["rougeL"])
            bert_list.append(m["bertscore_f1"])
            if m["length_ratio_percent"]:
                actual_ratios.append(m["length_ratio_percent"])

    avg = lambda lst: (mean(lst) if lst else 0.0)

    return {
        "total_runs": len(items),
        "total_algorithm_outputs": total_outputs,
        "total_processing_time_seconds": round(sum(processing_times), 2),
        "avg_target_length_ratio": round(avg(target_ratios), 1),
        "avg_actual_length_ratio": round(avg(actual_ratios), 1),
        "top_models": [
            {"model": k, "count": v} for k, v in sorted(model_counts.items(), key=lambda x: -x[1])[:10]
        ],
        "avg_rouge": {
            "rouge1": round(avg(rouge1_list), 4),
            "rouge2": round(avg(rouge2_list), 4),
            "rougeL": round(avg(rougeL_list), 4),
        },
        "avg_bertscore_f1": round(avg(bert_list), 4),
    }


def get_model_performance(time_range: str = "30d") -> list[dict]:
    items = _filter_by_range(_load_all_results(), time_range)
    buckets: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "bertscore_f1": [],
            "length_ratio_percent": [],
            "processing_time": [],
        }
    )

    for record in items:
        for alg in record.get("results") or []:
            name = alg.get("algorithm") or alg.get("key") or "unknown"
            m = _algorithm_metrics_from_row(alg)
            for key, value in m.items():
                buckets[name][key].append(value)

    rows = []
    for model, stats in buckets.items():
        rows.append(
            {
                "model": model,
                "count": len(stats["rougeL"]),
                "rouge1": round(mean(stats["rouge1"]), 4) if stats["rouge1"] else 0.0,
                "rouge2": round(mean(stats["rouge2"]), 4) if stats["rouge2"] else 0.0,
                "rougeL": round(mean(stats["rougeL"]), 4) if stats["rougeL"] else 0.0,
                "bertScore": round(mean(stats["bertscore_f1"]), 4) if stats["bertscore_f1"] else 0.0,
                "avgLengthRatio": round(mean(stats["length_ratio_percent"]), 1)
                if stats["length_ratio_percent"]
                else 0.0,
                "avgTime": round(mean(stats["processing_time"]), 3) if stats["processing_time"] else 0.0,
            }
        )
    return sorted(rows, key=lambda r: -r["rougeL"])


def get_runs_timeseries(time_range: str = "30d") -> list[dict]:
    items = _filter_by_range(_load_all_results(), time_range)
    by_day: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"date": "", "runs": 0, "avgRougeL": [], "avgBertScore": [], "avgLengthRatio": []}
    )

    for record in items:
        created = _parse_created_at(record.get("created_at"))
        day = (created or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
        bucket = by_day[day]
        bucket["date"] = day
        bucket["runs"] += 1
        meta = record.get("meta") or {}
        if meta.get("target_length_ratio") is not None:
            bucket["avgLengthRatio"].append(float(meta["target_length_ratio"]))

        rouge_vals = []
        bert_vals = []
        for alg in record.get("results") or []:
            m = _algorithm_metrics_from_row(alg)
            rouge_vals.append(m["rougeL"])
            bert_vals.append(m["bertscore_f1"])
        if rouge_vals:
            bucket["avgRougeL"].append(mean(rouge_vals))
        if bert_vals:
            bucket["avgBertScore"].append(mean(bert_vals))

    series = []
    for day in sorted(by_day.keys()):
        b = by_day[day]
        series.append(
            {
                "date": day,
                "count": b["runs"],
                "avgRougeL": round(mean(b["avgRougeL"]), 4) if b["avgRougeL"] else 0.0,
                "avgBertScore": round(mean(b["avgBertScore"]), 4) if b["avgBertScore"] else 0.0,
                "avgLengthRatio": round(mean(b["avgLengthRatio"]), 1) if b["avgLengthRatio"] else 0.0,
            }
        )
    return series


def get_visualization_data(time_range: str = "30d") -> dict:
    return {
        "model_performance": get_model_performance(time_range),
        "timeseries": get_runs_timeseries(time_range),
    }


def get_dashboard_payload(time_range: str = "30d", history_limit: int = 15) -> dict:
    return {
        "metrics": compute_dashboard_metrics(time_range),
        "visualization": get_visualization_data(time_range),
        "recent_runs": list_recent_results(history_limit),
        "time_range": time_range,
    }


def list_benchmark_results() -> list[dict]:
    benchmark_dir = RESULT_DIR.parent / "benchmark_results"
    if not benchmark_dir.exists():
        return []
    results = []
    for path in sorted(benchmark_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            agg = data.get("aggregate", data)
            results.append(
                {
                    "filename": path.name,
                    "dataset": agg.get("dataset_name"),
                    "model": agg.get("model_name"),
                    "samples": agg.get("samples_evaluated"),
                    "timestamp": agg.get("timestamp"),
                    "comparison": agg.get("comparison_all_algorithms", []),
                    "key_findings": agg.get("key_findings", []),
                }
            )
        except Exception:
            continue
    return results
