"""Read-only API service for cached VietNews dataset analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from backend.services.dataset_analysis.cache import (
    is_rebuild_running,
    load_cached_bundle,
    load_json_file,
    read_progress,
)
from backend.services.dataset_analysis.exporter import ANALYTICS_DIR, CHARTS_DIR
from backend.services.dataset_analysis.loader import resolve_limit
from backend.services.dataset_analysis.pipeline import run_dataset_analytics_pipeline
from src import config
from src.utils import logger


def _effective_limit() -> int | None:
    return resolve_limit(config.DATASET_ANALYTICS_LIMIT)


def _ensure_analytics(*, lazy_compute: bool = True, force: bool = False) -> dict[str, Any]:
    if not force:
        bundle = load_cached_bundle()
        if bundle:
            return bundle

    if not lazy_compute and not force:
        raise HTTPException(
            status_code=404,
            detail="Dataset analytics chưa được tính. Chạy: python scripts/run_dataset_analytics.py --full --force",
        )

    if force or config.DATASET_ANALYTICS_CACHE:
        logger.info("Computing dataset analytics (force=%s)...", force)
        try:
            limit = _effective_limit()
            return run_dataset_analytics_pipeline(
                limit_per_split=limit,
                full=limit is None,
                force=force,
            )
        except Exception as exc:
            logger.error("Analytics compute failed: %s", exc)
            raise HTTPException(status_code=503, detail=f"Dataset analytics failed: {exc}") from exc

    raise HTTPException(status_code=404, detail="Dataset analytics cache not found")


def get_dataset_analytics_payload(*, force: bool = False) -> dict[str, Any]:
    bundle = _ensure_analytics(force=force)
    return {
        "overview": bundle.get("overview", {}),
        "document_stats": bundle.get("document_stats", {}),
        "summary_stats": bundle.get("summary_stats", {}),
        "vocabulary": bundle.get("vocabulary", {}),
        "quality": bundle.get("quality", {}),
        "length_distribution": bundle.get("length_distribution", {}),
        "token_statistics": bundle.get("token_statistics", {}),
        "compression_statistics": bundle.get("compression_statistics", {}),
        "correlation": bundle.get("correlation", {}),
        "word_frequency": bundle.get("word_frequency", {}),
        "category_stats": bundle.get("category_stats", {}),
        "rouge_baseline": bundle.get("rouge_baseline", {}),
        "extractive_metrics": bundle.get("extractive_metrics", {}),
        "training_statistics": bundle.get("training_statistics", {}),
        "charts": bundle.get("charts", {}),
        "metadata": bundle.get("metadata", {}),
    }


def rebuild_dataset_analytics(*, force: bool = True) -> dict[str, Any]:
    if is_rebuild_running():
        return {
            "status": "running",
            "message": "Pipeline đang chạy",
            "progress": read_progress(),
        }
    limit = _effective_limit()
    result = run_dataset_analytics_pipeline(
        limit_per_split=limit,
        full=limit is None,
        force=force,
    )
    return {
        "status": "complete",
        "overview": result.get("overview", {}),
        "metadata": result.get("metadata", {}),
        "chart_count": len(result.get("charts", {})),
    }


def get_dataset_progress() -> dict[str, Any]:
    progress = read_progress()
    return {
        **progress,
        "running": is_rebuild_running(),
        "cache_available": analytics_available(),
    }


def get_dataset_overview() -> dict[str, Any]:
    data = load_json_file("dataset_overview.json")
    if data:
        return data
    return _ensure_analytics().get("overview", {})


def get_dataset_statistics() -> dict[str, Any]:
    data = load_json_file("dataset_statistics.json")
    if data:
        return data
    b = _ensure_analytics()
    return {"document_stats": b.get("document_stats", {}), "summary_stats": b.get("summary_stats", {})}


def get_dataset_quality() -> dict[str, Any]:
    data = load_json_file("dataset_quality.json")
    return data or _ensure_analytics().get("quality", {})


def get_token_statistics() -> dict[str, Any]:
    data = load_json_file("token_statistics.json")
    return data or _ensure_analytics().get("token_statistics", {})


def get_compression_statistics() -> dict[str, Any]:
    data = load_json_file("compression_statistics.json")
    return data or _ensure_analytics().get("compression_statistics", {})


def get_vocabulary_stats() -> dict[str, Any]:
    data = load_json_file("vocabulary.json")
    return data or _ensure_analytics().get("vocabulary", {})


def get_correlation_data() -> dict[str, Any]:
    data = load_json_file("correlation.json")
    return data or _ensure_analytics().get("correlation", {})


def get_charts_index() -> dict[str, Any]:
    data = load_json_file("charts_index.json")
    if data:
        return _with_chart_urls(data)
    bundle = _ensure_analytics()
    return _with_chart_urls({"charts": bundle.get("charts", {}), "metadata": bundle.get("metadata", {})})


def _with_chart_urls(payload: dict[str, Any]) -> dict[str, Any]:
    charts = payload.get("charts") or {}
    enriched = {}
    for key, path in charts.items():
        p = Path(path)
        enriched[key] = {
            "path": str(p),
            "filename": p.name,
            "url": f"/analytics/charts/file/{p.name}",
        }
    return {**payload, "charts": enriched, "chart_dir": str(CHARTS_DIR)}


def get_chart_file(filename: str) -> Path:
    safe = Path(filename).name
    path = CHARTS_DIR / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Chart not found: {safe}")
    return path


def analytics_available() -> bool:
    return (ANALYTICS_DIR / "dataset_analytics_bundle.json").exists()


def get_analytics_source() -> str | None:
    """Return metadata source (e.g. colab, huggingface) if bundle exists."""
    meta = load_json_file("metadata.json")
    if meta:
        return meta.get("source")
    bundle = load_cached_bundle()
    if bundle:
        return (bundle.get("metadata") or {}).get("source") or bundle.get("overview", {}).get("source")
    return None
