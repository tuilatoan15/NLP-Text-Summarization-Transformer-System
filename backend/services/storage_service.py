"""storage.py - Local file storage with optional MongoDB persistence."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import os
import shutil

from src.utils import save_json, logger


STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "storage"))
UPLOAD_DIR = STORAGE_ROOT / "uploads"
RESULT_DIR = STORAGE_ROOT / "results"


def save_upload_file(file_obj, filename: str) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(filename)
    destination = UPLOAD_DIR / f"{uuid4().hex}_{safe_name}"
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file_obj, buffer)
    return destination


def persist_compare_result(compare: dict[str, Any], *, input_preview: str | None = None) -> dict:
    meta = compare.get("meta") or {}
    preview = (input_preview or meta.get("input_preview") or "")[:500]
    payload = {
        "type": "compare",
        "algorithms": compare.get("algorithms", []),
        "results": compare.get("results", []),
        "ranking": compare.get("ranking", []),
        "best_model": compare.get("best_model"),
        "meta": meta,
        "performance": compare.get("performance", {}),
        "warning": compare.get("warning"),
        "text_preview": preview,
        "processing_time_seconds": (compare.get("performance") or {}).get("total_wall_time_s", 0),
    }
    return persist_result(payload)


def persist_result(payload: dict[str, Any]) -> dict:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_id = payload.get("result_id") or uuid4().hex
    stored = {
        **payload,
        "result_id": result_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = RESULT_DIR / f"{result_id}.json"
    save_json(stored, str(path))

    mongo_id = _save_to_mongo(stored)
    return {
        "result_id": result_id,
        "local_path": str(path),
        "mongo_id": mongo_id,
    }


def _save_to_mongo(payload: dict[str, Any]) -> str | None:
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        return None
    try:
        from pymongo import MongoClient

        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=1500)
        db_name = os.getenv("MONGO_DB", "nlp_summarization")
        collection_name = os.getenv("MONGO_COLLECTION", "summary_results")
        result = client[db_name][collection_name].insert_one(payload)
        return str(result.inserted_id)
    except Exception as exc:
        logger.warning(f"MongoDB persistence skipped: {exc}")
        return None


def _safe_filename(filename: str) -> str:
    cleaned = "".join(char for char in filename if char.isalnum() or char in "._- ")
    cleaned = cleaned.strip().replace(" ", "_")
    return cleaned or "uploaded_file"
