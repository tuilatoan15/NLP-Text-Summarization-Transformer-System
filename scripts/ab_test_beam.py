#!/usr/bin/env python3
"""A/B test RAG_GEN_NUM_BEAMS=5 (default) vs 3 — latency + faithfulness."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT = PROJECT_ROOT / "storage" / "results" / "ab_test_beam_20260703.json"
QUALITY_THRESHOLD_PCT = 1.0  # rollback nếu faithfulness giảm >1%


def _run_variant(beams: int | None, limit: int = 20) -> dict:
    if beams is None:
        os.environ.pop("RAG_GEN_NUM_BEAMS", None)
        label = "beam5_default"
    else:
        os.environ["RAG_GEN_NUM_BEAMS"] = str(beams)
        label = f"beam{beams}"

    os.environ["RAG_RESPONSE_CACHE"] = "0"
    os.environ["RAG_RETRIEVAL_CACHE"] = "0"

    # Force reload rag_config profile
    import importlib
    import backend.services.rag.rag_config as rc
    importlib.reload(rc)

    from scripts.evaluate_rag_system import RAG_TEST_SUITE, run_evaluation

    t0 = time.perf_counter()
    result = run_evaluation(limit=limit)
    wall = time.perf_counter() - t0
    metrics = result.get("metrics", {})
    return {
        "label": label,
        "beams": beams if beams is not None else 5,
        "wall_s": round(wall, 2),
        "faithfulness_pct": metrics.get("faithfulness_pct", 0),
        "recall_at_5_pct": metrics.get("recall_at_5_pct", 0),
        "avg_latency_s": metrics.get("avg_latency_s", 0),
        "p50_latency_s": metrics.get("p50_latency_s", 0),
        "p95_latency_s": metrics.get("p95_latency_s", 0),
    }


def main() -> None:
    limit = int(os.environ.get("AB_TEST_LIMIT", "15"))
    baseline = _run_variant(None, limit=limit)
    variant = _run_variant(3, limit=limit)

    faith_drop = baseline["faithfulness_pct"] - variant["faithfulness_pct"]
    lat_improve = (
        (baseline["avg_latency_s"] - variant["avg_latency_s"])
        / max(baseline["avg_latency_s"], 0.001)
        * 100
    )
    accept = faith_drop <= QUALITY_THRESHOLD_PCT
    decision = "accept_beam3" if accept else "rollback_keep_beam5"

    report = {
        "timestamp": int(time.time()),
        "limit_questions": limit,
        "baseline_beam5": baseline,
        "variant_beam3": variant,
        "faithfulness_drop_pct": round(faith_drop, 3),
        "latency_improvement_pct": round(lat_improve, 2),
        "decision": decision,
        "rollback_reason": None if accept else f"Faithfulness giảm {faith_drop:.2f}% > {QUALITY_THRESHOLD_PCT}%",
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not accept:
        print(f"\n⚠️ ROLLBACK: {report['rollback_reason']}")
        sys.exit(0)
    print("\n✅ Chấp nhận beam=3 — có thể set RAG_GEN_NUM_BEAMS=3 (chưa đổi default prod)")


if __name__ == "__main__":
    main()
