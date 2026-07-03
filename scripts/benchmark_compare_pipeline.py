#!/usr/bin/env python3
"""Benchmark wall time compare 15 thuật toán — before/after shared hybrid."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT = PROJECT_ROOT / "storage" / "results" / "compare_timing_baseline.json"

SAMPLE_TEXT = """
Trường Đại học Giao thông Vận tải Phân hiệu tại TP. Hồ Chí Minh là một trong những cơ sở đào tạo
hàng đầu về ngành Công nghệ thông tin tại miền Nam. Sinh viên Nguyễn Hữu Toàn thực hiện báo cáo thực tập
tốt nghiệp tại doanh nghiệp công nghệ, tập trung vào phát triển hệ thống phần mềm quản lý.
Chương 1 giới thiệu về công ty thực tập, tên đề tài, mục đích và yêu cầu thực tập.
Mục 1.1 trình bày thông tin công ty nơi sinh viên làm việc trong suốt ba tháng.
Mục 1.2.1 ghi rõ tên đề tài thực tập liên quan đến xây dựng ứng dụng web.
Mục 1.2.2 nêu mục đích và yêu cầu của đợt thực tập tốt nghiệp.
Đề cương đồ án nghiên cứu sử dụng PhoBERT làm mô hình chính để nhận dạng thực thể tiếng Việt.
spaCy NER được chọn làm mô hình so sánh đối chứng trong nghiên cứu.
Mục 3.3.2 thảo luận hạn chế về gán nhãn dữ liệu trong quá trình huấn luyện.
Mục 3.3.3 phân tích hạn chế xử lý file đầu vào với định dạng phức tạp.
Kết quả thực nghiệm cho thấy PhoBERT vượt trội hơn spaCy NER trên bộ dữ liệu tiếng Việt.
""" * 8  # ~2K từ


def _run(label: str, use_v2: bool) -> dict:
    os.environ["COMPARE_PIPELINE_V2"] = "1" if use_v2 else "0"
    from ai_models.model_registry import DEFAULT_ALGORITHMS
    from backend.services import dashboard_service
    import importlib
    importlib.reload(dashboard_service)
    from backend.services.dashboard_service import summarize_all

    t0 = time.perf_counter()
    result = summarize_all(SAMPLE_TEXT, algorithms=DEFAULT_ALGORITHMS[:15])
    wall = time.perf_counter() - t0
    return {
        "label": label,
        "compare_pipeline_v2": use_v2,
        "wall_time_s": round(wall, 3),
        "algorithm_count": len(result.get("results", [])),
        "performance": result.get("performance", {}),
    }


def main() -> None:
    # Chỉ benchmark 3 hybrid vit5 để tránh GPU timeout dài
    quick_algos = ["textrank", "lexrank", "lsa", "textrank-vit5", "lexrank-vit5", "lsa-vit5"]
    os.environ.setdefault("PRELOAD_MODELS_LIST", "vit5")

    from backend.services.dashboard_service import summarize_all
    results: dict = {"timestamp": int(time.time()), "variants": []}

    for use_v2, label in [(False, "legacy_sequential"), (True, "shared_hybrid_v2")]:
        os.environ["COMPARE_PIPELINE_V2"] = "1" if use_v2 else "0"
        t0 = time.perf_counter()
        out = summarize_all(SAMPLE_TEXT, algorithms=quick_algos)
        wall = time.perf_counter() - t0
        results["variants"].append({
            "label": label,
            "compare_pipeline_v2": use_v2,
            "algorithms": quick_algos,
            "wall_time_s": round(wall, 3),
            "per_algo_avg_s": round(
                sum(r["metrics"]["processing_time"] for r in out["results"]) / max(len(out["results"]), 1),
                3,
            ),
        })

    legacy = results["variants"][0]["wall_time_s"]
    v2 = results["variants"][1]["wall_time_s"]
    results["improvement_pct"] = round((legacy - v2) / max(legacy, 0.001) * 100, 1)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
