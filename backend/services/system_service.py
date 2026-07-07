"""Read-only system diagnostics for dashboard UI (GPU, node, models, config, search)."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from typing import Any

from src import config
from src.model_loader import registry_status
from src.model_registry import list_algorithms
from src.utils import get_device_info, get_model_device_str, logger


def _safe_psutil() -> dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.05)
        return {
            "cpu_percent": round(cpu, 1),
            "ram_total_gb": round(vm.total / (1024**3), 2),
            "ram_used_gb": round(vm.used / (1024**3), 2),
            "ram_percent": round(vm.percent, 1),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _nvidia_smi_fields() -> dict[str, Any]:
    """Optional GPU telemetry via nvidia-smi (Windows/Linux)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 6:
            return {}
        return {
            "gpu_name_smi": parts[0],
            "temperature_c": float(parts[1]) if parts[1] not in ("N/A", "[N/A]") else None,
            "gpu_utilization_percent": float(parts[2]) if parts[2] not in ("N/A", "[N/A]") else None,
            "memory_utilization_percent": float(parts[3]) if parts[3] not in ("N/A", "[N/A]") else None,
            "vram_total_mb_smi": float(parts[4]) if parts[4] not in ("N/A", "[N/A]") else None,
            "vram_used_mb_smi": float(parts[5]) if parts[5] not in ("N/A", "[N/A]") else None,
            "driver_version": parts[6] if len(parts) > 6 else None,
        }
    except Exception:
        return {}


def _pynvml_fields() -> dict[str, Any]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        return {
            "gpu_name_pynvml": name,
            "temperature_c": float(temp),
            "gpu_utilization_percent": float(util.gpu),
            "memory_utilization_percent": float(util.memory),
            "vram_total_mb_pynvml": round(mem.total / (1024**2), 1),
            "vram_used_mb_pynvml": round(mem.used / (1024**2), 1),
        }
    except Exception:
        return {}


_GPU_STATUS_CACHE: tuple[float, dict[str, Any]] | None = None
_GPU_STATUS_TTL_S = 5.0


def _compute_gpu_status() -> dict[str, Any]:
    """GPU + host memory snapshot. Returns available=False when no GPU — never fakes data."""
    base = get_device_info()
    host = _safe_psutil()
    extra = _pynvml_fields() or _nvidia_smi_fields()

    cuda_ok = bool(base.get("cuda_available"))
    available = cuda_ok and bool(base.get("gpu_name") or extra.get("gpu_name_smi") or extra.get("gpu_name_pynvml"))

    total_mb = base.get("total_vram_mb") or extra.get("vram_total_mb_smi") or extra.get("vram_total_mb_pynvml")
    used_mb = base.get("allocated_vram_mb") or extra.get("vram_used_mb_smi") or extra.get("vram_used_mb_pynvml")
    util_pct = extra.get("gpu_utilization_percent")
    if util_pct is None and total_mb and used_mb:
        util_pct = round(float(used_mb) / float(total_mb) * 100, 1)

    reg = registry_status()
    loaded_names = list((reg.get("models") or {}).keys())

    return {
        "available": available,
        "status": "ok" if available else "unavailable",
        "device": base.get("device", "cpu"),
        "gpu_name": base.get("gpu_name") or extra.get("gpu_name_smi") or extra.get("gpu_name_pynvml"),
        "cuda_version": base.get("cuda_build"),
        "torch_version": base.get("torch_version"),
        "compute_capability": base.get("compute_capability"),
        "fp16_enabled": reg.get("fp16"),
        "total_vram_mb": total_mb,
        "allocated_vram_mb": base.get("allocated_vram_mb"),
        "free_vram_mb": base.get("free_vram_mb"),
        "vram_used_mb": used_mb,
        "gpu_utilization_percent": util_pct,
        "memory_utilization_percent": extra.get("memory_utilization_percent"),
        "temperature_c": extra.get("temperature_c"),
        "driver_version": extra.get("driver_version"),
        "models_on_gpu": loaded_names,
        "host": host,
    }


def get_gpu_status() -> dict[str, Any]:
    """Cached wrapper — tránh gọi nvidia-smi/pynvml liên tục khi sidebar poll."""
    global _GPU_STATUS_CACHE
    now = time.monotonic()
    if _GPU_STATUS_CACHE is not None:
        cached_at, payload = _GPU_STATUS_CACHE
        if (now - cached_at) < _GPU_STATUS_TTL_S:
            return payload
    payload = _compute_gpu_status()
    _GPU_STATUS_CACHE = (now, payload)
    return payload


def _rag_model_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    try:
        from backend.services.rag.embedding_service import _embedders

        for name, embedder in _embedders.items():
            model = getattr(embedder, "_model", None)
            entries.append(
                {
                    "key": f"embedding:{name}",
                    "name": name,
                    "role": "embedding",
                    "loaded": model is not None,
                    "device": get_model_device_str(model) if model is not None else "not_loaded",
                    "fp16": getattr(embedder, "_fp16", None),
                }
            )
    except Exception:
        pass

    try:
        from backend.services.rag import reranker as reranker_mod

        rer = getattr(reranker_mod, "_reranker_instance", None)
        if rer is not None:
            entries.append(
                {
                    "key": "crossencoder",
                    "name": "CrossEncoder",
                    "role": "reranker",
                    "loaded": True,
                    "device": get_model_device_str(rer.model),
                    "fp16": getattr(rer, "fp16", None),
                }
            )
    except Exception:
        pass

    try:
        from backend.services.rag.rag_config import OLLAMA_MODEL, RAG_GENERATOR_TYPE

        entries.append(
            {
                "key": "llm",
                "name": OLLAMA_MODEL if RAG_GENERATOR_TYPE == "ollama" else RAG_GENERATOR_TYPE,
                "role": "generator",
                "loaded": True,
                "device": "api" if RAG_GENERATOR_TYPE != "local" else "local",
                "fp16": None,
            }
        )
    except Exception:
        pass

    return entries


def get_models_status() -> dict[str, Any]:
    """Summarization registry + RAG stack model status."""
    reg = registry_status()
    summarizers = []
    for key, info in (reg.get("models") or {}).items():
        summarizers.append(
            {
                "key": key,
                "name": key,
                "role": "summarizer",
                "loaded": info.get("loaded", True),
                "device": info.get("device", "unknown"),
                "fp16": info.get("fp16"),
                "load_time_s": info.get("load_time_s"),
                "lazy": not reg.get("preloaded", False),
            }
        )

    expected = ["vit5", "mt5", "bartpho", "bge-m3", "crossencoder", "qwen"]
    loaded_keys = {s["key"] for s in summarizers}
    rag_entries = _rag_model_entries()
    for e in rag_entries:
        loaded_keys.add(e["key"].split(":")[-1])

    return {
        "preloaded": reg.get("preloaded", False),
        "device": reg.get("device"),
        "fp16": reg.get("fp16"),
        "torch_compile": reg.get("torch_compile"),
        "summarizers": summarizers,
        "rag_models": rag_entries,
        "expected_models": expected,
        "loaded_count": len(summarizers) + sum(1 for e in rag_entries if e.get("loaded")),
    }


def get_node_status() -> dict[str, Any]:
    """Live node health for sidebar / overview."""
    gpu = get_gpu_status()
    reg = registry_status()
    host = gpu.get("host") or {}

    gpu_busy = False
    util = gpu.get("gpu_utilization_percent")
    if util is not None and float(util) > 70:
        gpu_busy = True

    cpu_busy = False
    cpu_pct = host.get("cpu_percent")
    if cpu_pct is not None and float(cpu_pct) > 85:
        cpu_busy = True

    if gpu_busy or cpu_busy:
        inference_state = "busy"
        status = "busy"
    elif reg.get("preloaded") or reg.get("models"):
        inference_state = "idle"
        status = "healthy"
    else:
        inference_state = "lazy"
        status = "healthy"

    return {
        "node_id": socket.gethostname(),
        "status": status,
        "inference_state": inference_state,
        "api_version": config.API_VERSION,
        "models_preloaded": reg.get("preloaded", False),
        "gpu_busy": gpu_busy,
        "cpu_busy": cpu_busy,
        "gpu": {
            "available": gpu.get("available", False),
            "name": gpu.get("gpu_name"),
            "utilization_percent": util,
            "temperature_c": gpu.get("temperature_c"),
        },
        "cpu_percent": cpu_pct,
        "ram_used_gb": host.get("ram_used_gb"),
        "ram_total_gb": host.get("ram_total_gb"),
        "timestamp": time.time(),
    }


def get_system_config() -> dict[str, Any]:
    """Read-only safe config for Settings / Configuration page."""
    from backend.services.rag import rag_config as rc

    redis_configured = bool(config.REDIS_URL)
    db_configured = bool(config.DATABASE_URL) and config.ENABLE_DB_PERSISTENCE

    return {
        "api": {
            "version": config.API_VERSION,
            "host": config.API_HOST,
            "port": config.API_PORT,
        },
        "gpu": {
            "force_cpu": config.FORCE_CPU,
            "use_fp16": config.USE_FP16,
            "use_torch_compile": config.USE_TORCH_COMPILE,
            "preload_models": config.PRELOAD_MODELS,
            "preload_models_list": config.PRELOAD_MODELS_LIST,
            "max_gpu_concurrent": config.MAX_GPU_CONCURRENT,
            "gpu_vram_limit_gb": config.GPU_VRAM_LIMIT_GB,
        },
        "inference": {
            "max_input_tokens": config.MAX_INPUT_TOKENS,
            "max_output_length": config.MAX_OUTPUT_LENGTH,
            "train_batch_size": config.TRAIN_BATCH_SIZE,
            "eval_batch_size": config.EVAL_BATCH_SIZE,
            "extractive_workers": config.EXTRACTIVE_WORKERS,
        },
        "rag": {
            "generator_type": rc.RAG_GENERATOR_TYPE,
            "embedding_model": config.DEFAULT_EMBEDDING_MODEL,
            "top_k_default": getattr(rc, "RETRIEVAL_FINAL_TOP_K", 10),
            "context_compression": rc.RAG_CONTEXT_COMPRESSION,
            "adaptive_context": rc.RAG_ADAPTIVE_CONTEXT,
            "response_cache": rc.RAG_RESPONSE_CACHE,
            "retrieval_cache": rc.RAG_RETRIEVAL_CACHE,
            "embedding_fp16": rc.RAG_EMBEDDING_FP16,
            "reranker_fp16": rc.RAG_RERANKER_FP16,
            "embedding_batch_size": rc.RAG_EMBEDDING_BATCH_SIZE,
            "reranker_batch_size": rc.RAG_RERANKER_BATCH_SIZE,
            "torch_compile": rc.RAG_TORCH_COMPILE,
        },
        "infra": {
            "vector_backend": config.VECTOR_BACKEND,
            "redis_configured": redis_configured,
            "database_configured": db_configured,
            "warm_analytics_cache": config.WARM_ANALYTICS_CACHE,
        },
    }


def _document_count() -> int:
    total = 0
    try:
        from backend.services.rag import get_rag_service

        total += len(get_rag_service().list_documents())
    except Exception as exc:
        logger.debug("RAG document count failed: %s", exc)

    try:
        from backend.services.document_service import DocumentService

        total += len(DocumentService().list_documents(limit=500))
    except Exception:
        pass

    return total


def _chat_session_count() -> int:
    try:
        from backend.services.rag import get_rag_service

        items = get_rag_service().repository.list_conversations(limit=500, offset=0)
        return len(items)
    except Exception:
        return 0


def get_overview_aggregates(dashboard_metrics: dict | None = None) -> dict[str, Any]:
    metrics = dashboard_metrics or {}
    pt_list_avg = metrics.get("avg_processing_time_seconds")

    dataset_overview: dict[str, Any] = {}
    try:
        from backend.services.dataset_analytics_service import get_dataset_overview, analytics_available

        if analytics_available():
            dataset_overview = get_dataset_overview()
    except Exception:
        pass

    return {
        "document_count": _document_count(),
        "chat_session_count": _chat_session_count(),
        "compare_run_count": metrics.get("total_runs", 0),
        "algorithm_output_count": metrics.get("total_algorithm_outputs", 0),
        "avg_rouge_l": (metrics.get("avg_rouge") or {}).get("rougeL", 0.0),
        "avg_processing_time_seconds": pt_list_avg if pt_list_avg is not None else 0.0,
        "avg_bertscore_f1": metrics.get("avg_bertscore_f1", 0.0),
        "dataset": dataset_overview,
        "dataset_total_documents": dataset_overview.get("total_documents"),
        "dataset_avg_compression": dataset_overview.get("avg_compression_ratio"),
        "dataset_vocab_size": dataset_overview.get("vocab_size"),
    }


def _extract_search_snippet(text: str, query: str, *, radius: int = 48) -> str:
    """Return a short excerpt around the first case-insensitive match."""
    if not text:
        return ""
    needle = (query or "").strip()
    if not needle:
        return text[:80]
    lower = text.lower()
    idx = lower.find(needle.lower())
    if idx < 0:
        return text[:80].strip()
    start = max(0, idx - radius)
    end = min(len(text), idx + len(needle) + radius)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = f"…{snippet}"
    if end < len(text):
        snippet = f"{snippet}…"
    return snippet


def _search_summarize_history(query: str, *, max_hits: int = 15) -> list[dict[str, Any]]:
    """Search persisted compare runs by source text, reference, and algorithm outputs."""
    q = (query or "").strip().lower()
    if not q:
        return []

    try:
        from backend.services.analytics_service import _load_all_results
    except Exception:
        return []

    hits: list[dict[str, Any]] = []
    seen: set[str] = set()

    for record in reversed(_load_all_results()):
        if record.get("type") != "compare":
            continue
        result_id = str(record.get("result_id") or "")
        if not result_id or result_id in seen:
            continue

        meta = record.get("meta") or {}
        full_text = str(record.get("full_text") or meta.get("full_text") or "")
        reference = str(record.get("reference_text") or meta.get("reference_text") or "")
        preview = str(record.get("text_preview") or meta.get("input_preview") or "")

        match_field: str | None = None
        match_text = ""

        if q in full_text.lower():
            match_field, match_text = "source", full_text
        elif q in preview.lower():
            match_field, match_text = "source", preview
        elif q in reference.lower():
            match_field, match_text = "reference", reference
        else:
            for alg in record.get("results") or []:
                summary = str(alg.get("summary") or "")
                if q in summary.lower():
                    match_field, match_text = "summary", summary
                    break
                algo_name = str(alg.get("algorithm") or alg.get("key") or "")
                if q in algo_name.lower():
                    match_field, match_text = "algorithm", algo_name
                    break

        best = record.get("best_model") or {}
        if not best and record.get("ranking"):
            best = record["ranking"][0]
        best_algo = str(best.get("algorithm") or best.get("key") or "")

        if not match_field and q in best_algo.lower():
            match_field, match_text = "algorithm", best_algo

        if not match_field:
            continue

        seen.add(result_id)
        hits.append(
            {
                "type": "summarize_history",
                "id": result_id,
                "result_id": result_id,
                "title": best_algo or "Compare run",
                "subtitle": _extract_search_snippet(match_text or preview, query),
                "match_field": match_field,
                "link": f"/summarize?result={result_id}",
            }
        )
        if len(hits) >= max_hits:
            break

    return hits


def search_dashboard(query: str, limit: int = 20) -> dict[str, Any]:
    q = (query or "").strip().lower()
    if not q:
        return {"query": query, "results": []}

    results: list[dict[str, Any]] = []

    for algo in list_algorithms():
        name = str(algo.get("name", ""))
        key = str(algo.get("key", ""))
        if q in name.lower() or q in key.lower():
            results.append(
                {
                    "type": "algorithm",
                    "id": key,
                    "title": name,
                    "subtitle": algo.get("group", ""),
                    "link": f"/compare?algo={key}",
                }
            )

    try:
        results.extend(_search_summarize_history(query, max_hits=limit))
    except Exception:
        pass

    try:
        from backend.services.rag import get_rag_service

        for doc in get_rag_service().list_documents():
            title = str(doc.get("filename") or doc.get("document_id") or "")
            if q in title.lower():
                results.append(
                    {
                        "type": "document",
                        "id": doc.get("document_id"),
                        "title": title,
                        "subtitle": f"{doc.get('chunk_count', 0)} chunks",
                        "link": "/chat",
                    }
                )
    except Exception:
        pass

    try:
        from backend.services.rag import get_rag_service

        for conv in get_rag_service().repository.list_conversations(limit=50, offset=0):
            title = str(conv.get("title") or "")
            if q in title.lower():
                results.append(
                    {
                        "type": "conversation",
                        "id": conv.get("id") or conv.get("conversation_id"),
                        "title": title,
                        "subtitle": "Chat conversation",
                        "link": "/chat",
                    }
                )
    except Exception:
        pass

    try:
        from api.research import _load_leaderboard_only

        data = _load_leaderboard_only()
        board = data.get("leaderboard") or []
        rows = board if isinstance(board, list) else list(board.values())
        for row in rows:
            name = str(row.get("name") or row.get("key") or "")
            if q in name.lower():
                results.append(
                    {
                        "type": "leaderboard",
                        "id": row.get("key", name),
                        "title": name,
                        "subtitle": f"ROUGE-1 {(row.get('rouge1') or 0):.3f}",
                        "link": "/benchmark",
                    }
                )
    except Exception:
        pass

    return {"query": query, "results": results[:limit]}
