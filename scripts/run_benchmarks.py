#!/usr/bin/env python3
"""Automated research benchmarking suite for comparing summarization algorithms."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard import summarize_all
from src.preprocess import clean_text
from src.utils import count_words, logger

# Premium mock articles to guarantee the benchmark script runs even without internet/dataset files
MOCK_VAL_SAMPLES = [
    {
        "article": (
            "Thủ tướng Chính phủ vừa ban hành quyết định phê duyệt chiến lược quốc gia về phát triển trí tuệ nhân tạo (AI) "
            "đến năm 2030. Chiến lược đặt mục tiêu đưa Việt Nam vào nhóm 4 nước dẫn đầu khu vực ASEAN và nhóm 50 nước dẫn đầu "
            "thế giới về nghiên cứu, phát triển và ứng dụng AI. Để đạt được mục tiêu này, Việt Nam sẽ thành lập 3 trung tâm "
            "quốc gia về lưu trữ dữ liệu lớn và tính toán hiệu năng cao. Ngoài ra, việc đào tạo nguồn nhân lực chất lượng cao "
            "cũng được nhấn mạnh với mục tiêu đào tạo 50.000 kỹ sư, chuyên gia về AI và bán dẫn trong vòng 5 năm tới. "
            "Chiến lược cũng khuyến khích các doanh nghiệp công nghệ lớn đầu tư xây dựng các phòng thí nghiệm AI xuất sắc."
        ),
        "summary": (
            "Việt Nam phê duyệt chiến lược quốc gia về phát triển trí tuệ nhân tạo (AI) đến năm 2030 với mục tiêu vào top 4 ASEAN "
            "và đào tạo 50.000 kỹ sư AI, bán dẫn trong vòng 5 năm tới."
        )
    },
    {
        "article": (
            "Tập đoàn Điện lực Việt Nam (EVN) cảnh báo tình hình cung cấp điện trong mùa khô năm nay sẽ gặp nhiều khó khăn. "
            "Nhu cầu phụ tải điện toàn hệ thống dự báo tăng trưởng từ 11% đến 13% so với cùng kỳ năm trước, đạt mức kỷ lục mới. "
            "Để bảo đảm cung cấp điện an toàn, EVN đang đẩy nhanh tiến độ thi công đường dây 500kV mạch 3 từ Quảng Trạch đến Phố Nối. "
            "Đồng thời, EVN huy động tối đa các nguồn điện than, khí và năng lượng tái tạo, kết hợp điều tiết nước các hồ thủy điện "
            "hợp lý để giữ nước cho giai đoạn cao điểm tháng 5 và tháng 6. Người dân và doanh nghiệp được khuyến khích tiết kiệm điện."
        ),
        "summary": (
            "EVN cảnh báo khó khăn cung cấp điện mùa khô do nhu cầu tăng cao và đang đẩy nhanh tiến độ đường dây 500kV mạch 3 "
            "để bảo đảm cung cấp điện an toàn."
        )
    }
]


def load_validation_data(path_str: str, limit: int = 5) -> list[dict]:
    path = Path(path_str)
    if not path.exists():
        logger.info(f"Validation dataset not found at {path_str}. Falling back to preloaded mock research samples.")
        return MOCK_VAL_SAMPLES[:limit]

    samples = []
    try:
        with path.open("r", encoding="utf-8") as reader:
            for line in reader:
                if line.strip():
                    samples.append(json.loads(line))
                if len(samples) >= limit:
                    break
        logger.info(f"Loaded {len(samples)} validation samples from {path_str}")
        return samples
    except Exception as exc:
        logger.error(f"Error loading validation file: {exc}. Falling back to mock samples.")
        return MOCK_VAL_SAMPLES[:limit]


def print_comparison_table(results: dict[str, dict]) -> None:
    """Print a beautifully formatted ASCII table comparing summarization models."""
    header = f"{'Model Key':<15} | {'ROUGE-1':<8} | {'ROUGE-2':<8} | {'ROUGE-L':<8} | {'BLEU':<8} | {'BERTScore':<10} | {'Semantic':<8} | {'Latency (s)':<12}"
    divider = "-" * len(header)
    
    print("\n" + divider)
    print("📈 RESEARCH ALGORITHM BENCHMARK COMPARISON MATRIX")
    print(divider)
    print(header)
    print(divider)
    
    for key, metrics in results.items():
        print(
            f"{key:<15} | "
            f"{metrics.get('rouge1', 0.0):<8.4f} | "
            f"{metrics.get('rouge2', 0.0):<8.4f} | "
            f"{metrics.get('rougeL', 0.0):<8.4f} | "
            f"{metrics.get('bleu', 0.0):<8.4f} | "
            f"{metrics.get('bertscore_f1', 0.0):<10.4f} | "
            f"{metrics.get('semantic_similarity', 0.0):<8.4f} | "
            f"{metrics.get('processing_time', 0.0):<12.3f}"
        )
    print(divider + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Summarization Benchmarking Automation.")
    parser.add_argument("--val-data", default="data/processed/vnexpress/validation.jsonl", help="Path to validation data file (.jsonl)")
    parser.add_argument("--limit", type=int, default=3, help="Number of samples to evaluate on.")
    parser.add_argument("--algorithms", nargs="+", default=["textrank", "lexrank", "lsa", "tfidf"], help="List of algorithms to benchmark.")
    parser.add_argument("--target-ratio", type=int, default=40, help="Target summary ratio in %.")
    parser.add_argument("--output-dir", default="storage/results", help="Directory to save benchmark run outputs.")
    args = parser.parse_args()

    logger.info("Initializing benchmarking suite...")
    samples = load_validation_data(args.val_data, limit=args.limit)
    if not samples:
        logger.error("No samples loaded. Exiting.")
        sys.exit(1)

    logger.info(f"Running benchmarks on {len(samples)} samples for algorithms: {args.algorithms}")
    
    # Store aggregated scores per model
    model_scores: dict[str, dict[str, list[float]]] = {
        alg: {
            "rouge1": [], "rouge2": [], "rougeL": [], "bleu": [],
            "bertscore_f1": [], "semantic_similarity": [], "processing_time": []
        }
        for alg in args.algorithms
    }

    t_start_suite = time.perf_counter()

    for idx, sample in enumerate(samples, start=1):
        logger.info(f"--- Benchmarking Sample [{idx}/{len(samples)}] ({count_words(sample['article'])} words) ---")
        article = sample["article"]
        reference = sample.get("summary") or sample.get("title")

        # Run compare using summarize_all pipeline
        try:
            compare_result = summarize_all(
                text=article,
                reference=reference,
                algorithms=args.algorithms,
                target_length_ratio=args.target_ratio,
                use_length_ratio=True
            )
            
            for row in compare_result.get("results", []):
                key = row.get("key")
                if key not in model_scores:
                    continue
                metrics = row.get("metrics") or {}
                model_scores[key]["rouge1"].append(metrics.get("rouge1", 0.0))
                model_scores[key]["rouge2"].append(metrics.get("rouge2", 0.0))
                model_scores[key]["rougeL"].append(metrics.get("rougeL", 0.0))
                model_scores[key]["bleu"].append(metrics.get("bleu", 0.0))
                model_scores[key]["bertscore_f1"].append(metrics.get("bertscore_f1", 0.0))
                model_scores[key]["semantic_similarity"].append(metrics.get("semantic_similarity", 0.0))
                model_scores[key]["processing_time"].append(row.get("processing_time", 0.0))
        except Exception as exc:
            logger.error(f"Sample {idx} benchmark failed: {exc}", exc_info=True)

    # Compute averages
    aggregated_results = {}
    for key, scores in model_scores.items():
        if not scores["rouge1"]:
            continue
        aggregated_results[key] = {
            metric: round(mean(score_list), 4)
            for metric, score_list in scores.items()
        }

    # Print summary table
    print_comparison_table(aggregated_results)

    # Save benchmark run log
    elapsed_total = time.perf_counter() - t_start_suite
    run_log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "samples_evaluated": len(samples),
        "total_wall_time_seconds": round(elapsed_total, 3),
        "algorithms": args.algorithms,
        "target_length_ratio": args.target_ratio,
        "results": aggregated_results
    }
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"benchmark_run_{int(time.time())}.json"
    out_file.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved benchmark log to {out_file}")


if __name__ == "__main__":
    main()
