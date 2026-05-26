#!/usr/bin/env python3
"""Vietnamese dataset cleaning, splitting, and statistics for research experiments."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from src.preprocess import clean_text
from src.utils import count_words


def clean_record(row: dict) -> dict | None:
    article = clean_text(str(row.get("article") or row.get("content") or ""))
    summary = clean_text(str(row.get("summary") or row.get("abstract") or ""))
    if len(article.split()) < 40 or len(summary.split()) < 8:
        return None
    return {"article": article, "summary": summary}


def split_records(records: list[dict], val_ratio: float = 0.1, seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:cut], shuffled[cut:]


def stats(records: list[dict]) -> dict:
    article_lens = [count_words(r["article"]) for r in records]
    summary_lens = [count_words(r["summary"]) for r in records]
    return {
        "count": len(records),
        "article_words_avg": round(sum(article_lens) / max(1, len(article_lens)), 2),
        "summary_words_avg": round(sum(summary_lens) / max(1, len(summary_lens)), 2),
        "article_words_p95": sorted(article_lens)[int(0.95 * max(0, len(article_lens) - 1))] if article_lens else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VN summarization dataset pipeline")
    parser.add_argument("--dataset", default="thanhnew2001/vnexpress")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--output-dir", default="data/processed/vnexpress")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    cleaned: list[dict] = []
    dropped = 0
    for row in ds:
        item = clean_record(row)
        if item:
            cleaned.append(item)
        else:
            dropped += 1

    train, val = split_records(cleaned, val_ratio=args.val_ratio)
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train),
        encoding="utf-8",
    )
    (out / "validation.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val),
        encoding="utf-8",
    )

    report = {
        "dataset": args.dataset,
        "dropped": dropped,
        "train": stats(train),
        "validation": stats(val),
        "top_article_tokens": Counter(
            word for r in train[:500] for word in r["article"].split()[:200]
        ).most_common(30),
    }
    (out / "stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
