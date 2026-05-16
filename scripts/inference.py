"""CLI: run one-off summarization comparison for text or a file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.dashboard import summarize_all
from src.file_parser import extract_text_from_file
from src.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run summarization inference and comparison.")
    parser.add_argument("--text", default=None)
    parser.add_argument("--file", default=None)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--algorithms", default="textrank,lexrank,lsa,vit5")
    parser.add_argument("--sentence_count", type=int, default=5)
    parser.add_argument("--max_output_length", type=int, default=config.MAX_OUTPUT_LENGTH)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.file:
        text = extract_text_from_file(args.file)
    elif args.text:
        text = args.text
    else:
        raise SystemExit("Provide --text or --file.")

    algorithms = [item.strip() for item in args.algorithms.split(",") if item.strip()]
    result = summarize_all(
        text=text,
        reference=args.reference,
        algorithms=algorithms,
        sentence_count=args.sentence_count,
        max_output_length=args.max_output_length,
    )

    if args.output:
        save_json(result, args.output)
    else:
        best = result.get("best_model") or {}
        print(f"Best model: {best.get('algorithm')} ({best.get('combined_score')})")
        for row in result["results"]:
            print("\n" + "=" * 80)
            print(f"{row['algorithm']} [{row['group']}]")
            print(f"ROUGE-L={row['metrics']['rougeL']} BERTScore={row['metrics']['bertscore_f1']} Time={row['processing_time']}s")
            print(row["summary"])


if __name__ == "__main__":
    main()
