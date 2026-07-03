"""
warmup.py — Preload và warm-up embedder/reranker RAG lúc startup.

Giảm cold-start latency query đầu tiên (~10–15s → <1s sau warm-up).
Hỗ trợ torch.compile tùy chọn với benchmark tự rollback nếu chậm hơn.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .rag_config import EMBEDDING_MODEL, PRELOAD_RAG_MODELS, RAG_TORCH_COMPILE

logger = logging.getLogger(__name__)

_WARMUP_DONE = False
_COMPILE_APPLIED: dict[str, bool] = {"embedder": False, "reranker": False}


def _benchmark_inference(fn, *, rounds: int = 3) -> float:
    """Đo latency trung bình (giây) sau 1 lần warm-up."""
    fn()
    start = time.perf_counter()
    for _ in range(rounds):
        fn()
    return (time.perf_counter() - start) / max(rounds, 1)


def _maybe_compile_sentence_model(model: Any, label: str) -> Any:
    """Áp dụng torch.compile nếu bật; rollback nếu benchmark chậm hơn baseline."""
    if not RAG_TORCH_COMPILE:
        return model
    try:
        from src.utils import cuda_is_usable

        if not cuda_is_usable():
            return model
        import torch
        torch_ver = tuple(int(x) for x in torch.__version__.split(".")[:2])
        if torch_ver < (2, 0):
            return model

        baseline_fn = lambda: model.encode(["warm-up benchmark câu tiếng Việt."], convert_to_numpy=True)
        baseline_s = _benchmark_inference(baseline_fn, rounds=2)

        compiled = torch.compile(model)
        compiled_fn = lambda: compiled.encode(["warm-up benchmark câu tiếng Việt."], convert_to_numpy=True)
        compiled_s = _benchmark_inference(compiled_fn, rounds=2)

        if compiled_s <= baseline_s * 1.05:
            logger.info(
                "⚡ torch.compile [%s] giữ lại — baseline=%.3fs compiled=%.3fs",
                label, baseline_s, compiled_s,
            )
            _COMPILE_APPLIED[label] = True
            return compiled
        logger.warning(
            "↩️ torch.compile [%s] rollback — chậm hơn (baseline=%.3fs > compiled=%.3fs)",
            label, baseline_s, compiled_s,
        )
        return model
    except Exception as exc:
        logger.warning("torch.compile [%s] bỏ qua: %s", label, exc)
        return model


def _maybe_compile_cross_encoder(model: Any) -> Any:
    if not RAG_TORCH_COMPILE:
        return model
    try:
        from src.utils import cuda_is_usable

        if not cuda_is_usable():
            return model
        import torch
        pairs = [("câu hỏi warm-up", "đoạn văn bản ngữ cảnh tiếng Việt.")]
        baseline_fn = lambda: model.predict(pairs, convert_to_numpy=True)
        baseline_s = _benchmark_inference(baseline_fn, rounds=2)

        compiled = torch.compile(model.model)
        original_model = model.model
        model.model = compiled
        compiled_fn = lambda: model.predict(pairs, convert_to_numpy=True)
        compiled_s = _benchmark_inference(compiled_fn, rounds=2)

        if compiled_s <= baseline_s * 1.05:
            logger.info(
                "⚡ torch.compile [reranker] giữ lại — baseline=%.3fs compiled=%.3fs",
                baseline_s, compiled_s,
            )
            _COMPILE_APPLIED["reranker"] = True
            return model
        model.model = original_model
        logger.warning(
            "↩️ torch.compile [reranker] rollback — chậm hơn (%.3fs > %.3fs)",
            baseline_s, compiled_s,
        )
        return model
    except Exception as exc:
        logger.warning("torch.compile [reranker] bỏ qua: %s", exc)
        return model


def preload_rag_models() -> dict[str, float]:
    """
    Tải embedder + reranker và chạy warm-up inference.
    Trả về dict timing (giây) cho monitoring.
    """
    global _WARMUP_DONE
    if not PRELOAD_RAG_MODELS:
        logger.info("ℹ️  PRELOAD_RAG_MODELS=0 — RAG models load lazily on first query")
        return {}

    if _WARMUP_DONE:
        return {}

    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    try:
        from .embedding_service import EmbeddingService

        embed_svc = EmbeddingService()
        t_embed = time.perf_counter()
        embed_svc.embed_query("warm-up truy vấn tiếng Việt.", EMBEDDING_MODEL)
        embed_svc.embed_documents(
            ["warm-up tài liệu một.", "warm-up tài liệu hai."],
            EMBEDDING_MODEL,
        )
        timings["embedder_warmup_s"] = round(time.perf_counter() - t_embed, 3)

        if RAG_TORCH_COMPILE:
            from .embedding_service import _embedders, _embedder_lock
            with _embedder_lock:
                embedder = _embedders.get(EMBEDDING_MODEL)
                if embedder is not None and embedder._model is not None:
                    embedder._model = _maybe_compile_sentence_model(embedder._model, "embedder")
    except Exception as exc:
        logger.warning("⚠️  RAG embedder preload lỗi: %s", exc)

    try:
        from .reranker import _get_reranker

        t_rerank = time.perf_counter()
        reranker = _get_reranker()
        if reranker is not None:
            reranker.predict(
                [("câu hỏi warm-up", "đoạn văn bản ngữ cảnh.")],
                convert_to_numpy=True,
            )
            if RAG_TORCH_COMPILE:
                from . import reranker as reranker_mod
                if reranker_mod._reranker_instance is not None:
                    reranker_mod._reranker_instance = _maybe_compile_cross_encoder(
                        reranker_mod._reranker_instance
                    )
        timings["reranker_warmup_s"] = round(time.perf_counter() - t_rerank, 3)
    except Exception as exc:
        logger.warning("⚠️  RAG reranker preload lỗi: %s", exc)

    timings["total_preload_s"] = round(time.perf_counter() - t0, 3)
    _WARMUP_DONE = True
    logger.info(
        "✅ RAG models preloaded — embedder=%.2fs reranker=%.2fs total=%.2fs compile=%s",
        timings.get("embedder_warmup_s", 0),
        timings.get("reranker_warmup_s", 0),
        timings["total_preload_s"],
        _COMPILE_APPLIED,
    )
    return timings


def is_warmup_done() -> bool:
    return _WARMUP_DONE
