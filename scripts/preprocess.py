"""CLI: clean and cache the VNExpress dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.dataset_loader import load_vnexpress_dataset
from src.utils import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, deduplicate, and cache VNExpress data.")
    parser.add_argument("--dataset_name", default="thanhnew2001/vnexpress")
    parser.add_argument("--local_data", default=None)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    dataset = load_vnexpress_dataset(
        local_csv_path=args.local_data,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        use_cache=not args.no_cache,
    )
    logger.info("Preprocessed dataset ready: train=%s validation=%s", len(dataset["train"]), len(dataset["validation"]))


if __name__ == "__main__":
    main()
