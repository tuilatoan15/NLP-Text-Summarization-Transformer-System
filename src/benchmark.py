"""
benchmark.py — Script đánh giá hệ thống: so sánh Extractive vs Abstractive.

Tính toán cho mỗi mẫu:
  - Extractive summary (TextRank)
  - Abstractive summary (ViT5 local hoặc Hub)
  - ROUGE cho cả hai
  - Consistency score cho abstractive (SentenceTransformer)
  - Thời gian xử lý và compression ratio

Lưu kết quả chi tiết và tổng hợp vào `storage/results/`.

Chạy ví dụ:
  python -m src.benchmark --max_samples 200 --samples 100 --model vit5
"""

from __future__ import annotations

import argparse
import time
import json
from pathlib import Path
from statistics import mean
from datetime import datetime

from train.dataset_loader import load_vnexpress_dataset
from src.extractive import extractive_summarize_with_details
from src.abstractive import get_summarizer, resolve_model_name
from src.evaluate import compute_rouge
from src.fact_check import check_consistency
from src.utils import ensure_dir, save_json, logger


def run_benchmark(
    dataset_name: str = "thanhnew2001/vnexpress",
    max_samples: int = 500,
    samples: int | None = None,
    model_name: str | None = None,
    local_model_dir: str | None = None,
    output_dir: str = "storage/results",
):
    ensure_dir(output_dir)

    logger.info("Loading dataset for benchmark...")
    dataset = load_vnexpress_dataset(max_samples=max_samples, dataset_name=dataset_name)
    validation = dataset["validation"]
    total = len(validation)
    n = samples if samples and samples > 0 else total
    n = min(n, total)
    logger.info(f"Validation samples available: {total}. Evaluating: {n}")

    model_name_resolved = resolve_model_name(model_name)
    logger.info(f"Preparing abstractive summarizer: {model_name_resolved}")
    summarizer = get_summarizer(model_name=model_name_resolved, local_model_dir=local_model_dir)

    per_sample = []
    ext_times = []
    abs_times = []
    ext_rouge1 = []
    ext_rouge2 = []
    ext_rougeL = []
    abs_rouge1 = []
    abs_rouge2 = []
    abs_rougeL = []
    consistency_scores = []

    for i in range(n):
        item = validation[i]
        article = item.get("article", "")
        reference = item.get("title", "")

        # Extractive (with details)
        t0 = time.time()
        try:
            ext_details = extractive_summarize_with_details(article)
            ext_summary = ext_details.get("summary", "")
        except Exception as e:
            logger.warning(f"Extractive failed at index {i}: {e}")
            ext_summary = ""
            ext_details = {}
        ext_t = time.time() - t0
        ext_times.append(ext_t)

        # Abstractive
        t0 = time.time()
        try:
            abs_summary = summarizer.summarize(article)
        except Exception as e:
            logger.warning(f"Abstractive generation failed at index {i}: {e}")
            abs_summary = ""
        abs_t = time.time() - t0
        abs_times.append(abs_t)

        # Metrics: ROUGE
        ext_scores = compute_rouge(ext_summary, reference)
        abs_scores = compute_rouge(abs_summary, reference)

        ext_rouge1.append(ext_scores.get("rouge1", 0.0))
        ext_rouge2.append(ext_scores.get("rouge2", 0.0))
        ext_rougeL.append(ext_scores.get("rougeL", 0.0))

        abs_rouge1.append(abs_scores.get("rouge1", 0.0))
        abs_rouge2.append(abs_scores.get("rouge2", 0.0))
        abs_rougeL.append(abs_scores.get("rougeL", 0.0))

        # Consistency (abstractive vs source)
        try:
            consistency = check_consistency(abs_summary, article, mode="fast")
            consistency_scores.append(consistency.get("consistency_score", 0.0))
        except Exception as e:
            logger.warning(f"Consistency check failed at index {i}: {e}")
            consistency = {"consistency_score": 0.0}
            consistency_scores.append(0.0)

        # Compression ratio (abstractive length / article length)
        art_len = len(article.split()) if article else 0
        abs_len = len(abs_summary.split()) if abs_summary else 0
        compression = round(abs_len / max(1, art_len), 4) if art_len else 0.0

        per_sample.append({
            "index": i,
            "article_len": art_len,
            "extractive": {
                "summary": ext_summary,
                "details": ext_details,
                "rouge": ext_scores,
                "time": round(ext_t, 4),
            },
            "abstractive": {
                "summary": abs_summary,
                "rouge": abs_scores,
                "time": round(abs_t, 4),
                "compression": compression,
            },
            "consistency": consistency,
        })

        if (i + 1) % 10 == 0 or i == n - 1:
            logger.info(f"Processed {i+1}/{n} samples")

    # Aggregate results
    agg = {
        "dataset_name": dataset_name,
        "model_name": model_name_resolved,
        "samples_evaluated": n,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "extractive": {
            "avg_rouge1": round(mean(ext_rouge1) if ext_rouge1 else 0.0, 4),
            "avg_rouge2": round(mean(ext_rouge2) if ext_rouge2 else 0.0, 4),
            "avg_rougeL": round(mean(ext_rougeL) if ext_rougeL else 0.0, 4),
            "avg_time": round(mean(ext_times) if ext_times else 0.0, 4),
        },
        "abstractive": {
            "avg_rouge1": round(mean(abs_rouge1) if abs_rouge1 else 0.0, 4),
            "avg_rouge2": round(mean(abs_rouge2) if abs_rouge2 else 0.0, 4),
            "avg_rougeL": round(mean(abs_rougeL) if abs_rougeL else 0.0, 4),
            "avg_time": round(mean(abs_times) if abs_times else 0.0, 4),
        },
        "consistency": {
            "avg_score": round(mean(consistency_scores) if consistency_scores else 0.0, 4),
        },
    }

    out_path = Path(output_dir) / f"benchmark_{model_name_resolved.replace('/','_')}_{n}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    save_json({"aggregate": agg, "samples": per_sample}, str(out_path))
    logger.info(f"Benchmark saved to: {out_path}")
    return agg, per_sample


def _parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark: Extractive vs Abstractive vs Consistency")
    parser.add_argument("--dataset_name", type=str, default="thanhnew2001/vnexpress")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--samples", type=int, default=100, help="Số mẫu đánh giá từ tập validation (0 = all)")
    parser.add_argument("--model", type=str, default="vit5", help="Model abstractive: vit5, bart hoặc HF path")
    parser.add_argument("--local_model_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="storage/results")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    samples = args.samples if args.samples > 0 else None
    run_benchmark(
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        samples=samples,
        model_name=args.model,
        local_model_dir=args.local_model_dir,
        output_dir=args.output_dir,
    )
