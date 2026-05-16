"""Load, clean, deduplicate, and cache the VNExpress dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from src import config
from src.preprocess import clean_dataset_record, deduplicate_records
from src.utils import logger


DEFAULT_MAX_SAMPLES = config.MAX_TRAIN_SAMPLES
DEFAULT_TEST_SPLIT = config.VALIDATION_RATIO
CACHE_DIR = config.DATASET_CACHE_DIR


def _detect_columns(column_names: list[str], article_col: str | None, title_col: str | None) -> tuple[str, str]:
    article_candidates = [article_col, "article", "text", "content", "body", "description", "document"]
    summary_candidates = [title_col, "summary", "title", "headline", "abstract", "short_description"]

    article = next((col for col in article_candidates if col and col in column_names), None)
    summary = next((col for col in summary_candidates if col and col in column_names), None)

    if not article or not summary:
        raise ValueError(f"Cannot detect article/summary columns from: {column_names}")
    return article, summary


def _clean_dataset(raw_ds: Dataset, article_col: str | None = None, title_col: str | None = None) -> Dataset:
    article_col, title_col = _detect_columns(raw_ds.column_names, article_col, title_col)
    records = []
    skipped = 0
    for item in raw_ds:
        cleaned = clean_dataset_record(str(item.get(article_col, "")), str(item.get(title_col, "")))
        if cleaned:
            records.append(cleaned)
        else:
            skipped += 1

    records = deduplicate_records(records)
    if not records:
        raise ValueError("Dataset has no valid samples after cleaning.")

    logger.info("Dataset cleaned: %s valid, %s skipped, duplicates removed", len(records), skipped)
    return Dataset.from_list([{"article": row["article"], "title": row["title"]} for row in records])


def _limit_dataset(raw_ds: Dataset, max_samples: int) -> Dataset:
    if max_samples and len(raw_ds) > max_samples:
        raw_ds = raw_ds.select(range(max_samples))
    return raw_ds


def load_from_csv(
    filepath: str,
    article_col: str = "article",
    title_col: str = "title",
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> DatasetDict:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw_ds = load_dataset("csv", data_files=str(path), split="train")
    elif suffix in {".json", ".jsonl"}:
        raw_ds = load_dataset("json", data_files=str(path), split="train")
    else:
        raise ValueError("Only CSV, JSON, and JSONL datasets are supported.")

    raw_ds = _limit_dataset(raw_ds, max_samples)
    cleaned = _clean_dataset(raw_ds, article_col=article_col, title_col=title_col)
    split = cleaned.train_test_split(test_size=DEFAULT_TEST_SPLIT, seed=42)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def load_from_huggingface(
    dataset_name: str = config.DATASET_NAME,
    article_col: str = "text",
    title_col: str = "title",
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> DatasetDict:
    logger.info("Loading dataset from HuggingFace: %s", dataset_name)
    try:
        raw_ds = load_dataset(dataset_name, split="train")
    except Exception:
        raw = load_dataset(dataset_name)
        first_split = "train" if "train" in raw else next(iter(raw.keys()))
        raw_ds = raw[first_split]

    raw_ds = _limit_dataset(raw_ds, max_samples)
    cleaned = _clean_dataset(raw_ds, article_col=article_col, title_col=title_col)
    split = cleaned.train_test_split(test_size=DEFAULT_TEST_SPLIT, seed=42)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def load_vnexpress_dataset(
    local_csv_path: Optional[str] = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    use_cache: bool = True,
    dataset_name: str = config.DATASET_NAME,
) -> DatasetDict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_name = f"{dataset_name.replace('/', '_')}_{max_samples}_clean"
    cache_path = CACHE_DIR / cache_name

    if use_cache and cache_path.exists():
        try:
            logger.info("Loading cleaned dataset cache: %s", cache_path)
            return load_from_disk(str(cache_path))
        except Exception as exc:
            logger.warning("Dataset cache failed, rebuilding: %s", exc)

    if local_csv_path:
        dataset = load_from_csv(local_csv_path, max_samples=max_samples)
    else:
        dataset = load_from_huggingface(dataset_name=dataset_name, max_samples=max_samples)

    if use_cache:
        dataset.save_to_disk(str(cache_path))
        logger.info("Saved cleaned dataset cache: %s", cache_path)
    return dataset
