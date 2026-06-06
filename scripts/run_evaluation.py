"""
scripts/run_evaluation.py — Batch evaluation for thesis.

Runs all 6 algorithms (3 extractive + 3 abstractive) on the VietNews test set
and produces a comparison table (JSON + CSV) for the thesis results chapter.

Usage:
    python scripts/run_evaluation.py --samples 200 --output storage/results/eval_report.json
    python scripts/run_evaluation.py --samples 500 --model vit5 --output storage/results/vit5_eval.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.utils import logger
from evaluation.metrics import compute_rouge, compute_bertscore


# ─────────────────────── Extractive algorithms ───────────────────
def _run_extractive(text: str, algo: str, num_sentences: int = 3) -> str:
    """Run a single extractive algorithm and return the summary string."""
    from summarizers.extractive.extractive_summarizer import summarize_extractive_algorithm
    result = summarize_extractive_algorithm(text, algorithm=algo, sentence_count=num_sentences)
    return result.get("summary", "") if isinstance(result, dict) else str(result)


# ─────────────────────── Abstractive algorithms ──────────────────
def _run_abstractive(text: str, algo: str) -> str:
    """Run a single abstractive algorithm and return the summary string."""
    from summarizers.abstractive.abstractive_summarizer import abstractive_summarize_key
    return abstractive_summarize_key(algo, text)


# ─────────────────── Core evaluation loop ────────────────────
def evaluate_algorithm(
    algorithm: str,
    algo_type: str,
    samples: list[dict],
    article_col: str,
    reference_col: str,
) -> dict:
    """Evaluate one algorithm on all samples. Returns aggregated metrics."""
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    times = []
    errors = 0

    logger.info("[%s | %s] Evaluating on %d samples...", algo_type, algorithm, len(samples))

    for i, sample in enumerate(samples):
        article = str(sample.get(article_col, "")).strip()
        reference = str(sample.get(reference_col, "")).strip()

        if not article or not reference:
            errors += 1
            continue

        t0 = time.perf_counter()
        try:
            if algo_type == "extractive":
                summary = _run_extractive(article, algorithm)
            else:
                summary = _run_abstractive(article, algorithm)
        except Exception as exc:
            logger.warning("  [%s] Sample %d failed: %s", algorithm, i, exc)
            errors += 1
            continue
        elapsed = time.perf_counter() - t0

        if not summary or not summary.strip():
            errors += 1
            continue

        rouge = compute_rouge(summary, reference)
        rouge1_scores.append(rouge["rouge1"])
        rouge2_scores.append(rouge["rouge2"])
        rougeL_scores.append(rouge["rougeL"])
        times.append(elapsed)

        if (i + 1) % 50 == 0:
            logger.info(
                "  [%s] Progress: %d/%d | avg ROUGE-1=%.4f",
                algorithm, i + 1, len(samples),
                sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0,
            )

    n = max(1, len(rouge1_scores))
    result = {
        "algorithm": algorithm,
        "type": algo_type,
        "num_samples": len(samples),
        "num_valid": len(rouge1_scores),
        "num_errors": errors,
        "rouge1": round(sum(rouge1_scores) / n, 4),
        "rouge2": round(sum(rouge2_scores) / n, 4),
        "rougeL": round(rougeL_scores and sum(rougeL_scores) / n or 0.0, 4),
        "avg_time_sec": round(sum(times) / max(1, len(times)), 4),
    }
    logger.info(
        "  [%s] Done: ROUGE-1=%.4f | ROUGE-2=%.4f | ROUGE-L=%.4f | avg_time=%.2fs",
        algorithm, result["rouge1"], result["rouge2"], result["rougeL"], result["avg_time_sec"],
    )
    return result


def load_test_samples(dataset_name: str, n: int, article_col: str, reference_col: str) -> list[dict]:
    """Load N samples from HuggingFace dataset."""
    from datasets import load_dataset

    logger.info("Loading dataset: %s (streaming, %d samples)", dataset_name, n)
    try:
        raw = load_dataset(dataset_name)
        # Pick test or validation split
        for split_name in ("test", "validation", "train"):
            if split_name in raw:
                split = raw[split_name]
                break
        else:
            split = raw[list(raw.keys())[0]]
    except Exception:
        raw = load_dataset(dataset_name, streaming=True)
        first_split = list(raw.keys())[0]
        split = list(raw[first_split].take(n))
        return split

    if len(split) > n:
        split = split.select(range(n))

    # Filter valid samples
    samples = [
        s for s in split
        if s.get(article_col) and s.get(reference_col)
        and len(str(s[article_col]).split()) >= 30
    ]
    logger.info("Loaded %d valid samples (from %d total)", len(samples), len(split))
    return samples


def save_results(results: list[dict], output_path: Path) -> None:
    """Save results as JSON and CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "meta": {"dataset": config.DATASET_NAME}}, f, ensure_ascii=False, indent=2)
    logger.info("Saved JSON: %s", output_path)

    # CSV
    csv_path = output_path.with_suffix(".csv")
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info("Saved CSV: %s", csv_path)


def print_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    header = f"{'Algorithm':<22} {'Type':<12} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'Time(s)':>8}"
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("  EVALUATION RESULTS — VietNews Summarization")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['algorithm']:<22} {r['type']:<12} "
            f"{r['rouge1']:>8.4f} {r['rouge2']:>8.4f} {r['rougeL']:>8.4f} {r['avg_time_sec']:>8.2f}"
        )
    print(sep)
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluation for thesis — 6 algorithms on VietNews.")
    parser.add_argument("--samples", type=int, default=200, help="Number of test samples per algorithm")
    parser.add_argument("--dataset", default=config.DATASET_NAME, help="HuggingFace dataset name")
    parser.add_argument("--article_col", default="article", help="Column name for article text")
    parser.add_argument("--reference_col", default="abstract", help="Column name for reference summary")
    parser.add_argument(
        "--model", default=None,
        choices=["vit5", "mt5", "bartpho", "textrank", "lexrank", "tfidf"],
        help="Run only one algorithm (default: all 6)",
    )
    parser.add_argument(
        "--output", default="storage/results/eval_report.json",
        help="Output path for JSON report",
    )
    parser.add_argument("--skip_abstractive", action="store_true", help="Only run extractive algorithms")
    parser.add_argument("--skip_extractive", action="store_true", help="Only run abstractive algorithms")
    return parser.parse_args()


# ─────────────────── Main ─────────────────────────────────────
EXTRACTIVE_ALGOS = ["textrank", "lexrank", "tfidf"]
ABSTRACTIVE_ALGOS = ["vit5", "bartpho", "mt5"]


def main():
    args = parse_args()

    samples = load_test_samples(
        dataset_name=args.dataset,
        n=args.samples,
        article_col=args.article_col,
        reference_col=args.reference_col,
    )

    if not samples:
        logger.error("No valid samples loaded. Exiting.")
        sys.exit(1)

    results = []

    # Determine which algorithms to run
    if args.model:
        # Single model mode
        if args.model in EXTRACTIVE_ALGOS:
            results.append(evaluate_algorithm(args.model, "extractive", samples, args.article_col, args.reference_col))
        else:
            results.append(evaluate_algorithm(args.model, "abstractive", samples, args.article_col, args.reference_col))
    else:
        # Run all 6
        if not args.skip_extractive:
            for algo in EXTRACTIVE_ALGOS:
                results.append(evaluate_algorithm(algo, "extractive", samples, args.article_col, args.reference_col))

        if not args.skip_abstractive:
            for algo in ABSTRACTIVE_ALGOS:
                results.append(evaluate_algorithm(algo, "abstractive", samples, args.article_col, args.reference_col))

    print_table(results)
    save_results(results, Path(args.output))
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
