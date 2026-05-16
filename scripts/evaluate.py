"""CLI: evaluate selected algorithms on VNExpress validation samples."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.dashboard import summarize_all
from src.evaluate import aggregate_metric_rows
from src.utils import save_json
from train.dataset_loader import load_vnexpress_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate summarization algorithms on VNExpress.")
    parser.add_argument("--dataset_name", default=config.DATASET_NAME)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--algorithms", default="textrank,lexrank,lsa,vit5,mt5,bartpho")
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "evaluation_report.json"))
    parser.add_argument("--sentence_count", type=int, default=5)
    parser.add_argument("--max_output_length", type=int, default=config.MAX_OUTPUT_LENGTH)
    args = parser.parse_args()

    algorithms = [item.strip() for item in args.algorithms.split(",") if item.strip()]
    dataset = load_vnexpress_dataset(dataset_name=args.dataset_name, max_samples=args.max_samples)
    validation = dataset["validation"]
    total = min(args.samples, len(validation))

    per_algorithm: dict[str, list[dict]] = {key: [] for key in algorithms}
    samples = []
    for index in range(total):
        item = validation[index]
        result = summarize_all(
            item["article"],
            reference=item["title"],
            algorithms=algorithms,
            sentence_count=args.sentence_count,
            max_output_length=args.max_output_length,
        )
        samples.append({"index": index, "ranking": result["ranking"], "results": result["results"]})
        for row in result["results"]:
            per_algorithm.setdefault(row["key"], []).append(row["metrics"])

    aggregate = {
        key: aggregate_metric_rows(rows)
        for key, rows in per_algorithm.items()
        if rows
    }
    report = {
        "dataset": args.dataset_name,
        "samples": total,
        "algorithms": algorithms,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "aggregate": aggregate,
        "samples_detail": samples,
    }
    save_json(report, args.output)


if __name__ == "__main__":
    main()
