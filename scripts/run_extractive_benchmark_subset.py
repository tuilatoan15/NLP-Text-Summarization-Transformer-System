#!/usr/bin/env python3
"""Extractive-only ROUGE benchmark — workaround khi full run_research_benchmark bị WinError pyarrow."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean

# Pre-import trước torch/sklearn để tránh race Windows KTM
import pyarrow  # noqa: F401
import pandas  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from evaluation.metrics import compute_rouge
from summarizers.extractive.extractive_summarizer import summarize_extractive_algorithm
from src.utils import logger

EXTRACTIVE_KEYS = ["textrank", "lexrank", "lsa"]


def load_samples(limit: int) -> list[dict]:
    dataset = load_dataset("nam194/vietnews", split="test")
    valid = []
    for item in dataset:
        article = (item.get("article") or "").strip()
        summary = (item.get("abstract") or item.get("title") or "").strip()
        if article and summary and len(article.split()) >= 30:
            valid.append({"article": article, "summary": summary, "title": item.get("title", "")})
    random.seed(42)
    picked = random.sample(valid, min(limit, len(valid)))
    for i, s in enumerate(picked):
        s["id"] = f"extractive_sample_{i+1:04d}"
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output", default="storage/results/benchmark_baseline_200.json")
    args = parser.parse_args()

    samples = load_samples(args.samples)
    logger.info("Loaded %d samples for extractive-only benchmark", len(samples))

    leaderboard: dict[str, dict] = {}
    t0 = time.perf_counter()

    for key in EXTRACTIVE_KEYS:
        rouge_scores: list[dict] = []
        latencies: list[float] = []
        for s in samples:
            t1 = time.perf_counter()
            res = summarize_extractive_algorithm(s["article"], algorithm=key, sentence_count=3)
            summary = res.get("summary", "") if isinstance(res, dict) else str(res)
            latencies.append(time.perf_counter() - t1)
            rouge = compute_rouge(summary, s["summary"])
            rouge_scores.append(rouge)

        leaderboard[key] = {
            "key": key,
            "name": key.upper(),
            "group": "extractive",
            "rouge1": round(mean(r["rouge1"] for r in rouge_scores), 4),
            "rouge2": round(mean(r["rouge2"] for r in rouge_scores), 4),
            "rougeL": round(mean(r["rougeL"] for r in rouge_scores), 4),
            "latency": round(mean(latencies), 4),
            "mode": "extractive_only_subset",
        }
        logger.info("%s ROUGE-L=%.4f", key, leaderboard[key]["rougeL"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset_name": "nam194/vietnews",
            "total_samples": len(samples),
            "mode": "extractive_only_subset",
            "note": "Subset extractive — full 15-model benchmark blocked by pyarrow WinError khi chạy song song",
            "wall_s": round(time.perf_counter() - t0, 2),
        },
        "leaderboard": leaderboard,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
