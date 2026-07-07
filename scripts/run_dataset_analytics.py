#!/usr/bin/env python
"""
scripts/run_dataset_analytics.py — Run VietNews dataset analytics pipeline.

Usage:
    python scripts/run_dataset_analytics.py              # uses DATASET_ANALYTICS_LIMIT (0=full)
    python scripts/run_dataset_analytics.py --full --force
    python scripts/run_dataset_analytics.py --limit 500 --force   # 500 per split only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.utils import logger
from backend.services.dataset_analysis.loader import resolve_limit
from backend.services.dataset_analysis.pipeline import run_dataset_analytics_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VietNews dataset analytics pipeline")
    parser.add_argument("--dataset", default=config.DATASET_NAME, help="HuggingFace dataset name")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per split (0 or omit with --full = full dataset)",
    )
    parser.add_argument("--full", action="store_true", help="Analyze all samples (DATASET_ANALYTICS_LIMIT=0)")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache valid")
    parser.add_argument("--skip-charts", action="store_true", help="Skip PNG chart generation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    limit = args.limit if args.limit is not None else config.DATASET_ANALYTICS_LIMIT
    effective = resolve_limit(limit, full=args.full)

    logger.info(
        "Dataset analytics: %s | limit=%s | mode=%s",
        args.dataset,
        limit,
        "FULL" if effective is None else f"{effective}/split",
    )

    t0 = time.perf_counter()
    result = run_dataset_analytics_pipeline(
        dataset_name=args.dataset,
        limit_per_split=limit,
        full=args.full,
        force=args.force,
        skip_charts=args.skip_charts,
    )
    elapsed = time.perf_counter() - t0

    overview = result.get("overview", {})
    meta = result.get("metadata", {})
    print()
    print("=" * 60)
    print("  VIETNEWS DATASET ANALYTICS")
    print("=" * 60)
    print(f"  Dataset     : {overview.get('dataset_name')}")
    mode = "FULL" if overview.get("full_dataset") else f"limit {overview.get('limit_per_split')}/split"
    print(f"  Mode        : {mode}")
    print(f"  Documents   : {overview.get('total_documents', 0):,}")
    print(f"  Raw total   : {overview.get('total_raw_samples', 0):,}")
    print(f"  Splits      : {overview.get('splits')}")
    print(f"  Raw splits  : {overview.get('split_raw_counts', overview.get('splits'))}")
    print(f"  Avg art wrd : {overview.get('avg_article_words')}")
    print(f"  Avg sum wrd : {overview.get('avg_summary_words')}")
    print(f"  Compression : {overview.get('avg_compression_ratio')}")
    print(f"  Vocab size  : {overview.get('vocab_size', 0):,}")
    print(f"  Charts      : {len(result.get('charts', {}))}")
    print(f"  Duration    : {meta.get('analysis_duration_sec', round(elapsed, 2))}s")
    print(f"  Output      : {config.STORAGE_DIR / 'analytics'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
