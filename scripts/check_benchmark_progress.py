#!/usr/bin/env python3
"""
scripts/check_benchmark_progress.py
Phân tích checkpoint benchmark 10K mẫu, tính toán tiến độ và ETA dựa trên latency thực tế.
"""

import json
from pathlib import Path
import sys

import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Danh sách 15 mô hình/thuật toán tóm tắt
ALL_CONFIGS = [
    # Extractive
    "textrank", "lexrank", "lsa",
    # Abstractive
    "vit5", "mt5", "bartpho",
    # Hybrid
    "textrank_vit5", "lexrank_vit5", "lsa_vit5",
    "textrank_mt5", "lexrank_mt5", "lsa_mt5",
    "textrank_bartpho", "lexrank_bartpho", "lsa_bartpho"
]

# Latency giả định ban đầu (sử dụng khi chưa có mẫu thực tế nào chạy để làm mốc)
DEFAULT_LATENCIES = {
    "textrank": 0.04, "lexrank": 0.04, "lsa": 0.06,
    "vit5": 7.5, "mt5": 8.0, "bartpho": 9.5,
    "textrank_vit5": 4.2, "lexrank_vit5": 4.3, "lsa_vit5": 4.4,
    "textrank_mt5": 4.5, "lexrank_mt5": 4.6, "lsa_mt5": 4.7,
    "textrank_bartpho": 4.9, "lexrank_bartpho": 5.0, "lsa_bartpho": 5.1
}

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} giây"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} phút"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f} giờ"
    days = hours / 24
    return f"{days:.2f} ngày ({hours:.1f} giờ)"

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Check benchmark progress.")
    parser.add_argument("--samples", type=int, default=10000, help="Total number of samples")
    args = parser.parse_args()
    
    total_samples = args.samples
    checkpoint_path = PROJECT_ROOT / "storage" / "results" / f"benchmark_checkpoint_{total_samples}.json"
    final_results_path = PROJECT_ROOT / "storage" / "results" / f"benchmark_{total_samples}_real.json"
    
    # 1. Kiểm tra xem benchmark đã hoàn thành chưa
    if final_results_path.exists():
        print(json.dumps({
            "status": "COMPLETED",
            "progress_percent": 100.0,
            "message": f"Benchmark {total_samples} mẫu đã hoàn thành hoàn toàn! Kết quả đã được lưu.",
            "details": {}
        }, ensure_ascii=False, indent=2))
        return

    # 2. Kiểm tra file checkpoint
    if not checkpoint_path.exists():
        print(json.dumps({
            "status": "NOT_STARTED",
            "progress_percent": 0.0,
            "message": f"Không tìm thấy file checkpoint {checkpoint_path.name}. Tiến trình benchmark {total_samples} chưa được khởi chạy hoặc chưa hoàn thành mẫu nào.",
            "details": {}
        }, ensure_ascii=False, indent=2))
        return

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
    except Exception as e:
        print(json.dumps({
            "status": "ERROR",
            "progress_percent": 0.0,
            "message": f"Lỗi đọc file checkpoint: {str(e)}",
            "details": {}
        }, ensure_ascii=False, indent=2))
        return

    summaries_db = checkpoint_data.get("summaries_db", {})
    if not summaries_db:
        print(json.dumps({
            "status": "RUNNING",
            "progress_percent": 0.0,
            "message": "Đã khởi tạo checkpoint nhưng chưa có mẫu nào được ghi nhận.",
            "details": {}
        }, ensure_ascii=False, indent=2))
        return

    # Đếm số mẫu hoàn thành cho từng giải thuật và tính latency thực tế trung bình
    completed_counts = {cfg: 0 for cfg in ALL_CONFIGS}
    total_latencies = {cfg: 0.0 for cfg in ALL_CONFIGS}

    for s_id, models in summaries_db.items():
        for cfg, result in models.items():
            if cfg in completed_counts and result.get("summary"):
                completed_counts[cfg] += 1
                total_latencies[cfg] += result.get("latency", 0.0)

    # Tính toán tiến độ từng giải thuật và ước lượng thời gian còn lại
    total_completed_tasks = 0
    total_tasks = len(ALL_CONFIGS) * total_samples
    remaining_time_seconds = 0.0
    
    details = {}
    
    for cfg in ALL_CONFIGS:
        completed = completed_counts[cfg]
        total_completed_tasks += completed
        remaining = max(0, total_samples - completed)
        
        # Tính latency trung bình thực tế, nếu chưa có thì dùng giả định
        if completed > 0:
            avg_latency = total_latencies[cfg] / completed
        else:
            avg_latency = DEFAULT_LATENCIES[cfg]
            
        remaining_cfg_time = remaining * avg_latency
        remaining_time_seconds += remaining_cfg_time
        
        details[cfg] = {
            "completed": completed,
            "remaining": remaining,
            "progress_percent": round((completed / total_samples) * 100, 2),
            "avg_latency": round(avg_latency, 4),
            "estimated_remaining_time": format_time(remaining_cfg_time)
        }

    progress_percent = (total_completed_tasks / total_tasks) * 100

    print(json.dumps({
        "status": "RUNNING",
        "progress_percent": round(progress_percent, 2),
        "completed_tasks": total_completed_tasks,
        "total_tasks": total_tasks,
        "estimated_remaining_time": format_time(remaining_time_seconds),
        "estimated_remaining_seconds": round(remaining_time_seconds, 1),
        "message": f"Tiến trình tổng thể: {progress_percent:.2f}% ({total_completed_tasks}/{total_tasks} tác vụ). Thời gian còn lại ước tính: {format_time(remaining_time_seconds)}.",
        "details": details
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
