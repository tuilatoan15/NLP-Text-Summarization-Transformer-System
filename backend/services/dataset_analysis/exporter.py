"""Export analytics results to JSON files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config
from src.utils import logger


ANALYTICS_DIR = config.STORAGE_DIR / "analytics"
CHARTS_DIR = config.STORAGE_DIR / "charts"
RESULTS_BUNDLE_PATH = config.RESULTS_DIR / "dataset_analytics.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def export_all(
    stats: dict[str, Any],
    *,
    charts: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write split JSON files and return path mapping."""
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": stats.get("overview", {}).get("dataset_name", config.DATASET_NAME),
        "cache_size_bytes": _dir_size_bytes(ANALYTICS_DIR),
        "charts_size_bytes": _dir_size_bytes(CHARTS_DIR),
        **(metadata or {}),
    }
    _write_json(ANALYTICS_DIR / "metadata.json", meta)

    mapping = {
        "metadata": str(ANALYTICS_DIR / "metadata.json"),
        "dataset_overview": str(ANALYTICS_DIR / "dataset_overview.json"),
        "dataset_statistics": str(ANALYTICS_DIR / "dataset_statistics.json"),
        "dataset_quality": str(ANALYTICS_DIR / "dataset_quality.json"),
        "token_statistics": str(ANALYTICS_DIR / "token_statistics.json"),
        "compression_statistics": str(ANALYTICS_DIR / "compression_statistics.json"),
        "correlation": str(ANALYTICS_DIR / "correlation.json"),
        "word_frequency": str(ANALYTICS_DIR / "word_frequency.json"),
        "length_distribution": str(ANALYTICS_DIR / "length_distribution.json"),
        "training_statistics": str(ANALYTICS_DIR / "training_statistics.json"),
        "vocabulary": str(ANALYTICS_DIR / "vocabulary.json"),
        "category_stats": str(ANALYTICS_DIR / "category_stats.json"),
        "rouge_baseline": str(ANALYTICS_DIR / "rouge_baseline.json"),
        "extractive_metrics": str(ANALYTICS_DIR / "extractive_metrics.json"),
        "charts_index": str(ANALYTICS_DIR / "charts_index.json"),
        "results_bundle": str(RESULTS_BUNDLE_PATH),
    }

    _write_json(ANALYTICS_DIR / "dataset_overview.json", stats.get("overview", {}))
    _write_json(
        ANALYTICS_DIR / "dataset_statistics.json",
        {
            "document_stats": stats.get("document_stats", {}),
            "summary_stats": stats.get("summary_stats", {}),
        },
    )
    _write_json(ANALYTICS_DIR / "dataset_quality.json", stats.get("quality", {}))
    _write_json(ANALYTICS_DIR / "token_statistics.json", stats.get("token_statistics", {}))
    _write_json(ANALYTICS_DIR / "compression_statistics.json", stats.get("compression_statistics", {}))
    _write_json(ANALYTICS_DIR / "correlation.json", stats.get("correlation", {}))
    _write_json(ANALYTICS_DIR / "word_frequency.json", stats.get("word_frequency", {}))
    _write_json(ANALYTICS_DIR / "length_distribution.json", stats.get("length_distribution", {}))
    _write_json(ANALYTICS_DIR / "training_statistics.json", stats.get("training_statistics", {}))
    _write_json(ANALYTICS_DIR / "vocabulary.json", stats.get("vocabulary", {}))
    _write_json(ANALYTICS_DIR / "category_stats.json", stats.get("category_stats", {}))
    _write_json(ANALYTICS_DIR / "rouge_baseline.json", stats.get("rouge_baseline", {}))
    _write_json(ANALYTICS_DIR / "extractive_metrics.json", stats.get("extractive_metrics", {}))

    charts_payload = {
        "charts": charts or {},
        "chart_dir": str(CHARTS_DIR),
        "chart_count": len(charts or {}),
        **meta,
    }
    _write_json(ANALYTICS_DIR / "charts_index.json", charts_payload)

    bundle = {**stats, "metadata": meta, "charts": charts or {}}
    _write_json(ANALYTICS_DIR / "dataset_analytics_bundle.json", bundle)
    _write_json(RESULTS_BUNDLE_PATH, bundle)

    logger.info("Exported analytics JSON to %s and %s", ANALYTICS_DIR, RESULTS_BUNDLE_PATH)
    return mapping
