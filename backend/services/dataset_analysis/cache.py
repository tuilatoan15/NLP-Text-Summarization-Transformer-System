"""Cache management for dataset analytics — compute once, read JSON after."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config

from backend.services.dataset_analysis.exporter import ANALYTICS_DIR, RESULTS_BUNDLE_PATH
from backend.services.dataset_analysis.loader import resolve_limit

_progress_lock = threading.Lock()
_rebuild_lock = threading.Lock()
_rebuild_running = False


def _cache_key(dataset_name: str, limit_per_split: int | None, full: bool) -> str:
    eff = resolve_limit(limit_per_split, full=full)
    raw = f"{dataset_name}|{eff}|{full}|v2"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def cache_metadata_path() -> Path:
    return ANALYTICS_DIR / "metadata.json"


def progress_path() -> Path:
    return ANALYTICS_DIR / "progress.json"


def write_progress(
    stage: str,
    percent: float,
    *,
    message: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "stage": stage,
        "percent": round(min(100.0, max(0.0, percent)), 2),
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    with _progress_lock:
        progress_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_progress() -> dict[str, Any]:
    path = progress_path()
    if not path.exists():
        return {"stage": "idle", "percent": 0.0, "message": "Chưa chạy phân tích"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"stage": "unknown", "percent": 0.0}


def clear_progress() -> None:
    path = progress_path()
    if path.exists():
        path.unlink(missing_ok=True)


def is_rebuild_running() -> bool:
    with _rebuild_lock:
        return _rebuild_running


def set_rebuild_running(running: bool) -> None:
    global _rebuild_running
    with _rebuild_lock:
        _rebuild_running = running


def is_cache_valid(
    dataset_name: str | None = None,
    limit_per_split: int | None = None,
    full: bool = False,
) -> bool:
    if not config.DATASET_ANALYTICS_CACHE:
        return False

    meta_path = cache_metadata_path()
    bundle_path = ANALYTICS_DIR / "dataset_analytics_bundle.json"
    if not meta_path.exists() or not bundle_path.exists():
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    name = dataset_name or config.DATASET_NAME
    expected = _cache_key(name, limit_per_split, full)
    return meta.get("cache_key") == expected


def write_cache_metadata(
    dataset_name: str,
    limit_per_split: int | None,
    full: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eff = resolve_limit(limit_per_split, full=full)
    meta = {
        "cache_key": _cache_key(dataset_name, limit_per_split, full),
        "dataset_name": dataset_name,
        "limit_per_split": eff,
        "full_dataset": eff is None,
        "limit_mode": "full" if eff is None else f"{eff}_per_split",
        "cache_enabled": config.DATASET_ANALYTICS_CACHE,
        **(extra or {}),
    }
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    cache_metadata_path().write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_cached_bundle() -> dict[str, Any] | None:
    for path in (ANALYTICS_DIR / "dataset_analytics_bundle.json", RESULTS_BUNDLE_PATH):
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def load_json_file(name: str) -> dict[str, Any] | None:
    path = ANALYTICS_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
