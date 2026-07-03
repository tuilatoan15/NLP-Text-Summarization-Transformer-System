"""
tasks.py — Định nghĩa các background tasks chạy bất đồng bộ bằng Celery.
Đã được đồng bộ với summarize_all trong backend.services.dashboard_service.
"""
from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List

# Setup path so worker can import everything properly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.celery_app import celery_app
from src.storage import persist_compare_result

logger = logging.getLogger(__name__)

# Lazy loaded services inside the worker process
_worker_rag_service = None


def _get_rag_service():
    global _worker_rag_service
    if _worker_rag_service is None:
        from backend.services.rag import get_rag_service
        _worker_rag_service = get_rag_service()
    return _worker_rag_service


@celery_app.task(bind=True, name="workers.tasks.summarize_task")
def summarize_task(self, text: str, model_type: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tác vụ tóm tắt đơn lẻ bất đồng bộ.
    """
    logger.info(f"🚀 [TASK] Bắt đầu summarize_task (ID: {self.request.id}) với mô hình: {model_type}")
    
    # Một summarize_task chính là so sánh giữa model_type và textrank
    algorithms = [model_type, "textrank"] if model_type != "textrank" else ["textrank"]
    seen = set()
    unique_algos = [a for a in algorithms if not (a in seen or seen.add(a))]

    try:
        from backend.services.dashboard_service import summarize_all

        compare = summarize_all(
            text=text,
            reference=settings.get("reference"),
            algorithms=unique_algos,
            sentence_count=settings.get("extractiveSentences", 5),
            max_output_length=settings.get("maxLength", 200),
            target_length_ratio=settings.get("target_length_ratio", 20),
            use_length_ratio=settings.get("use_length_ratio", False)
        )
        
        if settings.get("save_result", True):
            compare["storage"] = persist_compare_result(compare)
            
        return {
            "status": "success",
            "task_id": self.request.id,
            "data": compare
        }

    except Exception as exc:
        logger.error(f"❌ [TASK ERROR] summarize_task thất bại: {exc}")
        return {
            "status": "failed",
            "task_id": self.request.id,
            "error": str(exc)
        }


@celery_app.task(bind=True, name="workers.tasks.compare_task")
def compare_task(self, text: str, models: List[str], settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tác vụ so sánh nhiều mô hình tóm tắt bất đồng bộ.
    """
    logger.info(f"🚀 [TASK] Bắt đầu compare_task (ID: {self.request.id}) với mô hình: {models}")
    
    try:
        from backend.services.dashboard_service import summarize_all

        compare = summarize_all(
            text=text,
            reference=settings.get("reference"),
            algorithms=models,
            sentence_count=settings.get("extractiveSentences", 5),
            max_output_length=settings.get("maxLength", 200),
            target_length_ratio=settings.get("target_length_ratio", 20),
            use_length_ratio=settings.get("use_length_ratio", True)
        )
        
        if settings.get("save_result", True):
            compare["storage"] = persist_compare_result(compare)
            
        return {
            "status": "success",
            "task_id": self.request.id,
            "data": compare
        }

    except Exception as exc:
        logger.error(f"❌ [TASK ERROR] compare_task thất bại: {exc}")
        return {
            "status": "failed",
            "task_id": self.request.id,
            "error": str(exc)
        }


@celery_app.task(bind=True, name="workers.tasks.ingest_document_task")
def ingest_document_task(self, file_path_str: str, filename: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tác vụ nạp (ingest), chia chunk và nhúng vector tài liệu nền bất đồng bộ.
    """
    logger.info(f"🚀 [TASK] Bắt đầu ingest_document_task (ID: {self.request.id}) cho file: {filename}")
    t_start = time.perf_counter()

    try:
        file_path = Path(file_path_str)
        rag_service = _get_rag_service()

        # Gọi nghiệp vụ trích xuất & nhúng vector
        result = rag_service.upload_document(
            path=file_path,
            filename=filename,
            chunk_size=settings.get("chunk_size", 500),
            chunk_overlap=settings.get("chunk_overlap", 50),
        )

        elapsed = time.perf_counter() - t_start
        result.update({
            "status": "success",
            "task_id": self.request.id,
            "processing_time_s": round(elapsed, 3),
        })
        return result

    except Exception as exc:
        logger.error(f"❌ [TASK ERROR] ingest_document_task thất bại: {exc}")
        return {
            "status": "failed",
            "task_id": self.request.id,
            "error": str(exc)
        }


@celery_app.task(bind=True, name="workers.tasks.build_raptor_task")
def build_raptor_task(self, document_id: str, embedding_model: str) -> Dict[str, Any]:
    """
    Dựng cây RAPTOR phân cấp nền sau khi upload — không chặn hot-path ingest.
    """
    logger.info(
        "🌲 [TASK] build_raptor_task (ID: %s) doc=%s model=%s",
        self.request.id, document_id, embedding_model,
    )
    t_start = time.perf_counter()
    try:
        rag_service = _get_rag_service()
        rag_service.build_raptor_from_db(document_id, embedding_model)
        elapsed = time.perf_counter() - t_start
        return {
            "status": "success",
            "task_id": self.request.id,
            "document_id": document_id,
            "raptor_status": "ready",
            "processing_time_s": round(elapsed, 3),
        }
    except Exception as exc:
        logger.error("❌ [TASK ERROR] build_raptor_task thất bại: %s", exc)
        return {
            "status": "failed",
            "task_id": self.request.id,
            "document_id": document_id,
            "error": str(exc),
        }
