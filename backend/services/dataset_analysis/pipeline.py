"""Orchestrate VietNews dataset analytics pipeline (compute once, cache JSON)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from backend.services.dataset_analysis.cache import (
    is_cache_valid,
    load_cached_bundle,
    set_rebuild_running,
    write_cache_metadata,
    write_progress,
)
from backend.services.dataset_analysis.exporter import export_all
from backend.services.dataset_analysis.loader import load_dataset_records, resolve_limit
from backend.services.dataset_analysis.statistics import compute_all_statistics
from backend.services.dataset_analysis.visualizer import generate_all_charts
from src import config
from src.utils import logger


def _progress_adapter(base: float, span: float):
    def _cb(stage: str, frac: float) -> None:
        write_progress(stage, base + span * frac)

    return _cb


def run_dataset_analytics_pipeline(
    *,
    dataset_name: str | None = None,
    limit_per_split: int | None = None,
    full: bool = False,
    force: bool = False,
    skip_charts: bool = False,
) -> dict[str, Any]:
    name = dataset_name or config.DATASET_NAME
    if limit_per_split is None and not full:
        limit_per_split = config.DATASET_ANALYTICS_LIMIT
    effective_limit = resolve_limit(limit_per_split, full=full)

    if not force and is_cache_valid(name, limit_per_split, full):
        cached = load_cached_bundle()
        if cached:
            logger.info("Using cached dataset analytics (key valid)")
            return cached

    t0 = time.perf_counter()
    set_rebuild_running(True)
    write_progress("starting", 0, message="Khởi động pipeline phân tích dataset")

    try:
        logger.info(
            "Running dataset analytics pipeline: dataset=%s limit=%s full=%s",
            name,
            effective_limit,
            effective_limit is None,
        )

        loaded = load_dataset_records(
            dataset_name=name,
            limit_per_split=effective_limit,
            force_reload=False,
            full=full,
            progress_cb=_progress_adapter(0, 30),
        )
        if not loaded.records:
            raise ValueError(f"No valid records loaded from {name}")

        write_progress("statistics", 30, message="Đang tính thống kê...")
        stats = compute_all_statistics(
            loaded,
            progress_cb=_progress_adapter(30, 45),
        )

        elapsed = time.perf_counter() - t0
        stats["metadata"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_name": name,
            "limit_per_split": effective_limit,
            "full_dataset": effective_limit is None,
            "limit_mode": "full" if effective_limit is None else f"{effective_limit}_per_split",
            "record_count": len(loaded.records),
            "total_raw_samples": loaded.total_raw_samples,
            "split_raw_counts": getattr(loaded, "split_raw_counts", loaded.splits),
            "source": loaded.source,
            "analysis_duration_sec": round(elapsed, 2),
        }

        charts: dict[str, str] = {}
        if not skip_charts:
            write_progress("charts", 75, message="Đang sinh biểu đồ PNG...")
            try:
                charts = generate_all_charts(stats)
            except Exception as exc:
                logger.warning("Chart generation partial failure: %s", exc)

        write_progress("export", 90, message="Đang xuất JSON...")
        meta_extra = write_cache_metadata(
            name,
            limit_per_split,
            full,
            extra={
                **stats["metadata"],
                "chart_count": len(charts),
            },
        )
        export_all(stats, charts=charts, metadata=meta_extra)

        total_elapsed = time.perf_counter() - t0
        stats["metadata"]["analysis_duration_sec"] = round(total_elapsed, 2)
        write_progress("complete", 100, message="Hoàn tất", extra={"duration_sec": total_elapsed})

        bundle = load_cached_bundle()
        return bundle or {**stats, "charts": charts}
    finally:
        set_rebuild_running(False)
