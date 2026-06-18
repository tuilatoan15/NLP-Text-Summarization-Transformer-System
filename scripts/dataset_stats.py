"""
scripts/dataset_stats.py — Dataset statistics for thesis Chapter 4.

Produces an academic-standard statistical analysis of the VietNews dataset
including sample counts, text length distributions, and data splits.

Usage:
    python scripts/dataset_stats.py --samples 5000
    python scripts/dataset_stats.py --full  (use all data)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.utils import logger


def compute_text_stats(texts: list[str]) -> dict:
    """Compute word-level and character-level statistics."""
    if not texts:
        return {}

    lengths_words = [len(t.split()) for t in texts]
    lengths_chars = [len(t) for t in texts]

    def percentile(data, pct):
        data_sorted = sorted(data)
        k = (len(data_sorted) - 1) * pct / 100
        f = int(k)
        c = f + 1
        if c >= len(data_sorted):
            return data_sorted[-1]
        return round(data_sorted[f] + (k - f) * (data_sorted[c] - data_sorted[f]), 1)

    return {
        "count": len(texts),
        "avg_words": round(sum(lengths_words) / len(lengths_words), 1),
        "min_words": min(lengths_words),
        "max_words": max(lengths_words),
        "p25_words": percentile(lengths_words, 25),
        "p50_words": percentile(lengths_words, 50),
        "p75_words": percentile(lengths_words, 75),
        "p95_words": percentile(lengths_words, 95),
        "avg_chars": round(sum(lengths_chars) / len(lengths_chars), 1),
    }


def compute_compression_ratio(articles: list[str], abstracts: list[str]) -> dict:
    """Compute compression ratio statistics (abstract words / article words)."""
    ratios = []
    for a, s in zip(articles, abstracts):
        a_words = len(a.split())
        s_words = len(s.split())
        if a_words > 0:
            ratios.append(s_words / a_words)

    if not ratios:
        return {}

    return {
        "avg_ratio": round(sum(ratios) / len(ratios), 4),
        "avg_reduction_pct": round(100 * (1 - sum(ratios) / len(ratios)), 2),
        "min_ratio": round(min(ratios), 4),
        "max_ratio": round(max(ratios), 4),
    }


def analyze_dataset(dataset_name: str, max_samples: int | None) -> dict:
    """Load dataset and compute all statistics."""
    from datasets import load_dataset

    logger.info("Loading dataset: %s", dataset_name)
    raw = load_dataset(dataset_name)

    split_info = {}
    all_articles = []
    all_abstracts = []

    for split_name in ("train", "validation", "test"):
        if split_name not in raw:
            continue
        split = raw[split_name]
        if max_samples and len(split) > max_samples:
            split = split.select(range(max_samples))

        articles = [str(s.get("article", "")).strip() for s in split]
        abstracts = [str(s.get("abstract", "")).strip() for s in split]
        titles = [str(s.get("title", "")).strip() for s in split]

        # Filter empty
        valid = [(a, ab, t) for a, ab, t in zip(articles, abstracts, titles) if a and ab]
        articles = [v[0] for v in valid]
        abstracts = [v[1] for v in valid]

        split_info[split_name] = {
            "total_samples": len(split),
            "valid_samples": len(valid),
            "article_stats": compute_text_stats(articles),
            "abstract_stats": compute_text_stats(abstracts),
            "compression": compute_compression_ratio(articles, abstracts),
        }

        all_articles.extend(articles)
        all_abstracts.extend(abstracts)

    # Overall stats
    overall = {
        "total_valid_samples": len(all_articles),
        "dataset_name": dataset_name,
        "columns": list(raw[list(raw.keys())[0]].column_names),
        "splits": list(raw.keys()),
        "article_stats_overall": compute_text_stats(all_articles),
        "abstract_stats_overall": compute_text_stats(all_abstracts),
        "compression_overall": compute_compression_ratio(all_articles, all_abstracts),
        "split_details": split_info,
    }

    return overall


def print_thesis_table(stats: dict) -> None:
    """Print a LaTeX-ready statistics table for the thesis."""
    print()
    print("=" * 60)
    print("  DATASET STATISTICS — VietNews (nam194/vietnews)")
    print("=" * 60)
    print(f"  Dataset    : {stats['dataset_name']}")
    print(f"  Columns    : {', '.join(stats['columns'])}")
    print(f"  Total Valid: {stats['total_valid_samples']:,} samples")
    print()

    for split_name, info in stats["split_details"].items():
        art = info["article_stats"]
        abst = info["abstract_stats"]
        comp = info["compression"]
        print(f"  [{split_name.upper()}] {info['valid_samples']:,} samples")
        print(f"    Article  — avg: {art.get('avg_words', '?'):>7.1f} words | "
              f"p50: {art.get('p50_words', '?')} | p95: {art.get('p95_words', '?')}")
        print(f"    Abstract — avg: {abst.get('avg_words', '?'):>7.1f} words | "
              f"p50: {abst.get('p50_words', '?')} | p95: {abst.get('p95_words', '?')}")
        print(f"    Compression ratio: {comp.get('avg_ratio', '?'):.4f} "
              f"(avg {comp.get('avg_reduction_pct', '?'):.1f}% reduction)")
        print()

    overall_art = stats["article_stats_overall"]
    overall_abst = stats["abstract_stats_overall"]
    overall_comp = stats["compression_overall"]
    print("  [OVERALL STATISTICS]")
    print(f"    Article  — avg words: {overall_art.get('avg_words'):.1f}")
    print(f"    Abstract — avg words: {overall_abst.get('avg_words'):.1f}")
    print(f"    Compression ratio   : {overall_comp.get('avg_ratio'):.4f}")
    print(f"    Avg reduction       : {overall_comp.get('avg_reduction_pct'):.1f}%")
    print("=" * 60)
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Academic dataset statistics for VietNews.")
    parser.add_argument("--dataset", default=config.DATASET_NAME)
    parser.add_argument("--samples", type=int, default=1000, help="Max samples per split to analyze")
    parser.add_argument("--full", action="store_true", help="Analyze ALL samples (slow)")
    parser.add_argument("--output", default="storage/results/dataset_stats.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    max_s = None if args.full else args.samples

    stats = analyze_dataset(args.dataset, max_s)
    print_thesis_table(stats)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to: {out_path}")
