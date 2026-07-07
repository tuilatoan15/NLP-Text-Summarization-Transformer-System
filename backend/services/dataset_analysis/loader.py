"""Load VietNews (nam194/vietnews) from HuggingFace or local disk cache."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from src import config
from src.utils import logger

# Batch size for streaming HF splits (1000–5000 recommended)
LOAD_BATCH_SIZE = int(getattr(config, "DATASET_ANALYTICS_BATCH_SIZE", 2000) or 2000)


@dataclass
class DatasetRecord:
    article: str
    abstract: str
    title: str = ""
    category: str = ""
    guid: str = ""
    split: str = ""


@dataclass
class LoadedDataset:
    dataset_name: str
    records: list[DatasetRecord] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    source: str = "huggingface"
    limit_per_split: int | None = None
    total_raw_samples: int = 0
    split_raw_counts: dict[str, int] = field(default_factory=dict)


def resolve_limit(limit_per_split: int | None, *, full: bool = False) -> int | None:
    """Resolve analytics sample limit.

    - ``full=True`` or ``limit <= 0`` or ``limit is None`` → full dataset (no cap).
    - ``limit > 0`` → at most *N samples per split* (train, validation, test each capped).
    """
    if full:
        return None
    if limit_per_split is None or limit_per_split <= 0:
        return None
    return limit_per_split


def _local_cache_path(dataset_name: str) -> Path:
    safe = dataset_name.replace("/", "_")
    return config.DATASET_CACHE_DIR / f"{safe}_analytics.jsonl"


def _iter_jsonl(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if limit is not None and count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1


def _save_jsonl_cache(records: list[DatasetRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(
                json.dumps(
                    {
                        "article": rec.article,
                        "abstract": rec.abstract,
                        "title": rec.title,
                        "category": rec.category,
                        "guid": rec.guid,
                        "split": rec.split,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _row_to_record(row: dict[str, Any], split: str) -> DatasetRecord:
    return DatasetRecord(
        article=str(row.get("article", "") or "").strip(),
        abstract=str(row.get("abstract", "") or "").strip(),
        title=str(row.get("title", "") or "").strip(),
        category=str(row.get("category", row.get("topic", "")) or "").strip(),
        guid=str(row.get("guid", row.get("id", "")) or "").strip(),
        split=split,
    )


def _iter_split_batches(
    split_ds: Any,
    *,
    limit_per_split: int | None,
    batch_size: int = LOAD_BATCH_SIZE,
    progress_cb: Callable[[str, float], None] | None = None,
    split_name: str = "",
) -> Iterator[list[DatasetRecord]]:
    """Yield record batches from a HF split without loading entire split into RAM."""
    total = len(split_ds)
    cap = total if limit_per_split is None else min(total, limit_per_split)
    taken = 0

    while taken < cap:
        end = min(taken + batch_size, cap)
        # Slice view — not select(range(N)) unless limit_per_split > 0 caps the split
        batch_rows = split_ds[taken:end]
        batch_len = end - taken
        records: list[DatasetRecord] = []

        if isinstance(batch_rows, dict):
            keys = list(batch_rows.keys())
            n_rows = len(batch_rows[keys[0]]) if keys else 0
            for i in range(n_rows):
                row = {k: batch_rows[k][i] for k in keys}
                rec = _row_to_record(row, split_name)
                if rec.article and rec.abstract:
                    records.append(rec)
        else:
            for item in batch_rows:
                rec = _row_to_record(item, split_name)
                if rec.article and rec.abstract:
                    records.append(rec)

        taken = end
        if progress_cb and cap:
            progress_cb(f"loading:{split_name}", taken / cap)
        if records:
            yield records


def _load_from_hf(
    dataset_name: str,
    limit_per_split: int | None,
    *,
    progress_cb: Callable[[str, float], None] | None = None,
) -> LoadedDataset:
    from datasets import load_dataset

    effective_limit = resolve_limit(limit_per_split)
    logger.info(
        "Loading HuggingFace dataset: %s (limit_per_split=%s → %s)",
        dataset_name,
        limit_per_split,
        "FULL" if effective_limit is None else effective_limit,
    )

    raw = None
    errors: list[str] = []
    for kwargs in (
        {},
        {"trust_remote_code": True},
        {"keep_in_memory": True},
        {"trust_remote_code": True, "keep_in_memory": True},
    ):
        try:
            raw = load_dataset(dataset_name, **kwargs)
            break
        except Exception as exc:
            errors.append(str(exc))
            logger.warning("load_dataset attempt failed: %s", exc)

    if raw is None:
        raise RuntimeError(
            f"Cannot load dataset {dataset_name}. Errors: {'; '.join(errors[:3])}"
        )

    records: list[DatasetRecord] = []
    splits: dict[str, int] = {}
    split_raw_counts: dict[str, int] = {}
    total_raw = 0
    columns = list(raw[list(raw.keys())[0]].column_names)

    split_names = [s for s in ("train", "validation", "test") if s in raw]
    for si, split_name in enumerate(split_names):
        split_ds = raw[split_name]
        split_raw_counts[split_name] = len(split_ds)
        total_raw += len(split_ds)

        def _split_progress(stage: str, frac: float) -> None:
            if progress_cb:
                overall = (si + frac) / max(len(split_names), 1)
                progress_cb(f"loading:{split_name}", overall)

        for batch in _iter_split_batches(
            split_ds,
            limit_per_split=effective_limit,
            progress_cb=_split_progress,
            split_name=split_name,
        ):
            records.extend(batch)

        splits[split_name] = sum(1 for r in records if r.split == split_name)

    return LoadedDataset(
        dataset_name=dataset_name,
        records=records,
        splits=splits,
        columns=columns,
        source="huggingface",
        limit_per_split=effective_limit,
        total_raw_samples=total_raw,
        split_raw_counts=split_raw_counts,
    )


def load_dataset_records(
    dataset_name: str | None = None,
    limit_per_split: int | None = None,
    use_local_cache: bool = True,
    force_reload: bool = False,
    *,
    full: bool = False,
    progress_cb: Callable[[str, float], None] | None = None,
) -> LoadedDataset:
    """Load VietNews records.

    ``limit_per_split``: 0/None = full dataset; >0 = cap *per split*.
    Local JSONL cache is only used when ``effective_limit is None`` (full run).
    """
    name = dataset_name or config.DATASET_NAME
    effective_limit = resolve_limit(limit_per_split, full=full)
    cache_path = _local_cache_path(name)

    if (
        use_local_cache
        and cache_path.exists()
        and not force_reload
        and effective_limit is None
    ):
        logger.info("Reading full analytics cache: %s", cache_path)
        records: list[DatasetRecord] = []
        splits: dict[str, int] = {}
        total = 0
        for row in _iter_jsonl(cache_path):
            total += 1
            rec = _row_to_record(row, str(row.get("split", "train")))
            if rec.article and rec.abstract:
                records.append(rec)
                splits[rec.split] = splits.get(rec.split, 0) + 1
        if records:
            return LoadedDataset(
                dataset_name=name,
                records=records,
                splits=splits,
                columns=["article", "abstract", "title", "category", "guid"],
                source="local_cache",
                limit_per_split=None,
                total_raw_samples=total,
                split_raw_counts=splits.copy(),
            )

    loaded = _load_from_hf(name, effective_limit, progress_cb=progress_cb)
    if use_local_cache and loaded.records and effective_limit is None:
        try:
            _save_jsonl_cache(loaded.records, cache_path)
            logger.info(
                "Saved local analytics cache: %s (%s records)", cache_path, len(loaded.records)
            )
        except Exception as exc:
            logger.warning("Failed to save local cache: %s", exc)
    return loaded
