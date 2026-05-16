"""
analytics.py — Tập hợp các hàm tổng hợp metrics cho dashboard.

Các hàm chính:
 - _load_all_results(): đọc tất cả JSON trong `storage/results`
 - compute_dashboard_metrics(): trả về counters, trung bình ROUGE, token stats
 - get_visualization_data(): trả dữ liệu sẵn cho biểu đồ frontend
 - list_recent_results(): liệt kê metadata các kết quả đã lưu

File format: dựa trên `storage.persist_result` và payloads từ API.
"""

import json
from pathlib import Path
from statistics import mean
from typing import List

from src.utils import ensure_dir


RESULT_DIR    = Path("storage/results")
BENCHMARK_DIR = Path("storage/benchmark_results")


def _load_all_results() -> List[dict]:
    if not RESULT_DIR.exists():
        return []
    files = sorted(RESULT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    results = []
    for p in files:
        try:
            results.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return results


def list_recent_results(limit: int = 20) -> List[dict]:
    items = _load_all_results()
    out = []
    for r in items[-limit:][::-1]:
        out.append({
            "result_id": r.get("result_id"),
            "created_at": r.get("created_at"),
            "meta": r.get("meta") or {},
            "summary_count": len(r.get("documents", [])) if r.get("documents") else (len(r.get("results", [])) if r.get("results") else 1),
            "storage_path": r.get("storage", {}).get("local_path") if isinstance(r.get("storage"), dict) else None,
        })
    return out


def compute_dashboard_metrics() -> dict:
    items = _load_all_results()
    total = len(items)
    if total == 0:
        return {
            "total_summaries": 0,
            "total_processing_time_seconds": 0,
            "top_models": [],
            "avg_rouge": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
            "token_stats": {},
            "compression": {},
        }

    model_counts = {}
    rouge1_list = []
    rouge2_list = []
    rougeL_list = []
    processing_times = []
    input_lengths = []
    summary_lengths = []

    for r in items:
        # processing_time may be top-level or per-document
        pt = r.get("processing_time_seconds") or 0
        if pt:
            processing_times.append(float(pt))

        # models used
        model_name = None
        if isinstance(r.get("controls"), dict):
            model_name = r["controls"].get("model_name")
        # fallback: some payloads store meta info
        meta = r.get("meta") or {}
        if not model_name and meta.get("model_name"):
            model_name = meta.get("model_name")
        if model_name:
            model_counts[model_name] = model_counts.get(model_name, 0) + 1

        # aggregate rouge if present in different formats
        if r.get("scores") and isinstance(r.get("scores"), dict):
            # scores from single-document pipeline: scores.extractive & abstractive
            try:
                ext = r["scores"].get("extractive", {})
                rouge1_list.append(ext.get("rouge1", 0.0) or 0.0)
                rouge2_list.append(ext.get("rouge2", 0.0) or 0.0)
                rougeL_list.append(ext.get("rougeL", 0.0) or 0.0)
            except Exception:
                pass

        # from compare results: r['results'] list
        if r.get("results") and isinstance(r.get("results"), list):
            for alg in r["results"]:
                rouge = alg.get("rouge") or {}
                rouge1_list.append(rouge.get("rouge1", 0.0) or 0.0)
                rouge2_list.append(rouge.get("rouge2", 0.0) or 0.0)
                rougeL_list.append(rouge.get("rougeL", 0.0) or 0.0)
                # length
                summary_lengths.append(alg.get("length_words") or 0)

        # tokens
        wc = (r.get("word_count") or {}).get("input") or meta.get("input_words")
        if wc:
            try:
                input_lengths.append(int(wc))
            except Exception:
                pass

        # try best summary length
        if r.get("word_count"):
            try:
                summary_lengths.append(int(r["word_count"].get("best", 0) or 0))
            except Exception:
                pass

    avg = lambda lst: (mean(lst) if lst else 0.0)

    top_models = sorted(model_counts.items(), key=lambda x: -x[1])

    compression = {}
    if sum(input_lengths) > 0:
        compression["avg_compression_ratio"] = round(sum(summary_lengths) / sum(input_lengths), 3) if sum(summary_lengths) > 0 else 0.0
    else:
        compression["avg_compression_ratio"] = 0.0

    # BERTScore aggregate
    bertscore_list = []
    for r in items:
        if r.get("results") and isinstance(r.get("results"), list):
            for alg in r["results"]:
                bs = alg.get("bertscore") or {}
                if bs.get("f1"):
                    bertscore_list.append(float(bs["f1"]))

    return {
        "total_summaries": total,
        "total_processing_time_seconds": round(sum(processing_times), 2),
        "top_models": [{"model": k, "count": v} for k, v in top_models[:10]],
        "avg_rouge": {"rouge1": round(avg(rouge1_list), 4), "rouge2": round(avg(rouge2_list), 4), "rougeL": round(avg(rougeL_list), 4)},
        "avg_bertscore_f1": round(avg(bertscore_list), 4),
        "token_stats": {"min_input": min(input_lengths) if input_lengths else 0, "max_input": max(input_lengths) if input_lengths else 0, "median_input": sorted(input_lengths)[len(input_lengths)//2] if input_lengths else 0},
        "compression": compression,
    }


def get_visualization_data() -> dict:
    """Trả dữ liệu tổng hợp cho các biểu đồ frontend (ROUGE, time, length per model)."""
    items = _load_all_results()
    model_buckets = {}
    for r in items:
        # try to determine model
        model = None
        if isinstance(r.get("controls"), dict):
            model = r["controls"].get("model_name")
        if not model and r.get("results"):
            # take names from results
            for alg in r["results"]:
                m = alg.get("algorithm")
                if m:
                    model_buckets.setdefault(m, {"rougeL": [], "time": [], "length": []})
                    rouge = alg.get("rouge") or {}
                    model_buckets[m]["rougeL"].append(rouge.get("rougeL", 0.0) or 0.0)
                    model_buckets[m]["time"].append(alg.get("time_seconds", 0.0) or 0.0)
                    model_buckets[m]["length"].append(alg.get("length_words", 0) or 0)
        else:
            if model:
                # single pipeline result
                scores = r.get("scores") or {}
                if isinstance(scores, dict):
                    ext = scores.get("extractive") or {}
                    model_buckets.setdefault(model, {"rougeL": [], "time": [], "length": []})
                    model_buckets[model]["rougeL"].append(ext.get("rougeL", 0.0) or 0.0)
                    model_buckets[model]["time"].append(r.get("processing_time_seconds", 0.0) or 0.0)
                    model_buckets[model]["length"].append((r.get("word_count") or {}).get("best", 0) or 0)

    labels = list(model_buckets.keys())
    rouge_avg = [round(mean(model_buckets[m]["rougeL"]), 4) if model_buckets[m]["rougeL"] else 0.0 for m in labels]
    time_avg = [round(mean(model_buckets[m]["time"]), 3) if model_buckets[m]["time"] else 0.0 for m in labels]
    length_avg = [int(round(mean(model_buckets[m]["length"]))) if model_buckets[m]["length"] else 0 for m in labels]

    return {"labels": labels, "rougeL_avg": rouge_avg, "time_avg": time_avg, "length_avg": length_avg}


def list_benchmark_results() -> List[dict]:
    """Đọc tất cả file benchmark từ storage/benchmark_results/."""
    if not BENCHMARK_DIR.exists():
        return []
    results = []
    for p in sorted(BENCHMARK_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            agg = data.get("aggregate", data)
            results.append({
                "filename": p.name,
                "dataset": agg.get("dataset_name"),
                "model": agg.get("model_name"),
                "samples": agg.get("samples_evaluated"),
                "timestamp": agg.get("timestamp"),
                "comparison": agg.get("comparison_all_algorithms", []),
                "key_findings": agg.get("key_findings", []),
            })
        except Exception:
            continue
    return results
