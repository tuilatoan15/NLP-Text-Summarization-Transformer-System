#!/usr/bin/env python3
"""Freeze baseline đo lường trước kế hoạch tối ưu — gộp hotpath + RAG eval."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT = PROJECT_ROOT / "storage" / "results" / "baseline_before_plan.json"


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    vals_sorted = sorted(vals)
    idx = min(len(vals_sorted) - 1, int(round((p / 100.0) * (len(vals_sorted) - 1))))
    return round(vals_sorted[idx], 4)


def main() -> None:
    os.environ.setdefault("RAG_RESPONSE_CACHE", "0")
    os.environ.setdefault("RAG_RETRIEVAL_CACHE", "0")
    os.environ.setdefault("PRELOAD_RAG_MODELS", "1")

    from scripts.benchmark_rag_hotpath import run_benchmark
    from scripts.evaluate_rag_system import run_evaluation

    t0 = time.perf_counter()
    hotpath = run_benchmark("baseline_before_plan")
    rag_eval = run_evaluation()
    elapsed = time.perf_counter() - t0

    totals = [q["total_s"] for q in hotpath.get("queries", [])]
    baseline = {
        "label": "baseline_before_plan",
        "timestamp": int(time.time()),
        "freeze_duration_s": round(elapsed, 2),
        "env": {
            "RAG_RESPONSE_CACHE": os.environ.get("RAG_RESPONSE_CACHE", "0"),
            "RAG_RETRIEVAL_CACHE": os.environ.get("RAG_RETRIEVAL_CACHE", "0"),
            "RAG_GEN_NUM_BEAMS": os.environ.get("RAG_GEN_NUM_BEAMS", "5 (default)"),
        },
        "hotpath": hotpath,
        "latency": {
            "p50_total_s": _percentile(totals, 50),
            "p95_total_s": _percentile(totals, 95),
            "avg_total_s": hotpath.get("avg_total_s", 0),
            "throughput_qps": hotpath.get("throughput_qps", 0),
        },
        "rag_quality": rag_eval.get("metrics", {}),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(baseline, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
