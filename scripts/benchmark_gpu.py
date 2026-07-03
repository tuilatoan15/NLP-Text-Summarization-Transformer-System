#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_gpu.py — Benchmark GPU utilization cho embedding, reranker, summarizer, RAG.

Ghi kết quả vào storage/results/gpu_benchmark.json (merge before/after nếu có).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("RAG_RESPONSE_CACHE", "0")
os.environ.setdefault("RAG_RETRIEVAL_CACHE", "0")
os.environ.setdefault("PRELOAD_RAG_MODELS", "0")

RESULTS_PATH = PROJECT_ROOT / "storage" / "results" / "gpu_benchmark.json"

SAMPLE_TEXTS = [
    f"Câu văn bản tiếng Việt số {i} dùng để đo throughput embedding trên GPU."
    for i in range(100)
]
SAMPLE_PAIRS = [
    (f"Câu hỏi benchmark số {i}?", f"Đoạn ngữ cảnh tài liệu tiếng Việt số {i} chứa thông tin liên quan.")
    for i in range(100)
]
SUMMARY_SOURCE = (
    "Nhu cầu tiêu thụ điện trong mùa nắng nóng tiếp tục tăng cao tại nhiều địa phương. "
    "Các nhà máy thủy điện miền Bắc phải vận hành thận trọng do mực nước hồ chứa chưa phục hồi. "
    "EVN khuyến nghị theo dõi phụ tải giờ cao điểm và duy trì nguồn điện dự phòng."
) * 3


def _vram_mb() -> dict[str, float]:
    try:
        import torch
        from src.utils import cuda_is_usable

        if not cuda_is_usable():
            return {}
        idx = torch.cuda.current_device()
        return {
            "allocated_mb": round(torch.cuda.memory_allocated(idx) / 1024 ** 2, 1),
            "reserved_mb": round(torch.cuda.memory_reserved(idx) / 1024 ** 2, 1),
        }
    except Exception:
        return {}


def _device_of(model: Any) -> str:
    from src.utils import get_model_device_str
    return get_model_device_str(model)


def benchmark_embedding(*, force_cpu: bool = False) -> dict[str, Any]:
    from pipeline.schema import EmbeddingConfig
    from embeddings.embedder import SentenceTransformerEmbedder
    from backend.services.rag.rag_config import EMBEDDING_MODEL
    from src.utils import resolve_torch_device_str

    device = "cpu" if force_cpu else resolve_torch_device_str()
    cfg = EmbeddingConfig(
        model_name=EMBEDDING_MODEL,
        device=device,
        use_fp16=not force_cpu,
        batch_size=32,
        show_progress=False,
    )
    embedder = SentenceTransformerEmbedder(cfg)
    model = embedder._load_model()

    # warm-up
    embedder.embed_documents(SAMPLE_TEXTS[:4])
    t0 = time.perf_counter()
    embedder.embed_documents(SAMPLE_TEXTS)
    elapsed = time.perf_counter() - t0

    return {
        "device": _device_of(model) if model else "hash",
        "count": len(SAMPLE_TEXTS),
        "elapsed_s": round(elapsed, 4),
        "throughput_per_s": round(len(SAMPLE_TEXTS) / max(elapsed, 1e-6), 2),
        "vram_after_mb": _vram_mb(),
    }


def benchmark_reranker(*, force_cpu: bool = False) -> dict[str, Any]:
    from sentence_transformers import CrossEncoder
    from backend.services.rag.rag_config import RERANKER_MODEL, RAG_RERANKER_BATCH_SIZE
    from src.utils import resolve_torch_device_str

    device = "cpu" if force_cpu else resolve_torch_device_str()
    rer = CrossEncoder(RERANKER_MODEL, max_length=512, device=device)

    rer.predict(SAMPLE_PAIRS[:4], batch_size=RAG_RERANKER_BATCH_SIZE, convert_to_numpy=True)
    t0 = time.perf_counter()
    rer.predict(SAMPLE_PAIRS, batch_size=RAG_RERANKER_BATCH_SIZE, convert_to_numpy=True)
    elapsed = time.perf_counter() - t0

    return {
        "device": _device_of(rer.model),
        "count": len(SAMPLE_PAIRS),
        "elapsed_s": round(elapsed, 4),
        "throughput_per_s": round(len(SAMPLE_PAIRS) / max(elapsed, 1e-6), 2),
        "vram_after_mb": _vram_mb(),
    }


def benchmark_summarize(model_key: str = "vit5", runs: int = 3) -> dict[str, Any]:
    from backend.services.rag.summarizer import _run_transformer_generate
    from backend.services.rag.rag_config import GENERATION_PROFILES
    from src.model_loader import get_loaded_model

    profile = GENERATION_PROFILES.get(model_key)
    if profile is None:
        return {"error": f"unknown model {model_key}"}

    get_loaded_model(model_key)
    latencies: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _run_transformer_generate(model_key, SUMMARY_SOURCE, profile)
        latencies.append(time.perf_counter() - t0)

    loaded = get_loaded_model(model_key)
    return {
        "model": model_key,
        "device": _device_of(loaded.model),
        "runs": runs,
        "avg_s": round(sum(latencies) / len(latencies), 4),
        "min_s": round(min(latencies), 4),
        "vram_after_mb": _vram_mb(),
    }


def benchmark_rag_chat() -> dict[str, Any]:
    from backend.services.rag.service import RAGChatService

    service = RAGChatService()
    docs = service.list_documents()
    if not docs:
        return {"skipped": True, "reason": "no_documents"}

    doc_id = docs[0]["id"]
    query = "Tóm tắt nội dung chính của tài liệu."
    t0 = time.perf_counter()
    res = service.chat(query=query, conversation_id=None, document_ids=[doc_id])
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_s": round(elapsed, 4),
        "answer_len": len(res.get("answer", "") or ""),
        "vram_after_mb": _vram_mb(),
    }


def run_suite(label: str, *, force_cpu: bool = False, skip_rag: bool = False) -> dict[str, Any]:
    from src.utils import get_device_info
    from embeddings.embedder import SentenceTransformerEmbedder

    # Tránh cache class-level gây nhầm device giữa CPU/GPU runs
    SentenceTransformerEmbedder._model_cache.clear()

    suite: dict[str, Any] = {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu_info": get_device_info(),
    }

    print(f"[{label}] embedding …")
    suite["embedding_100"] = benchmark_embedding(force_cpu=force_cpu)

    print(f"[{label}] reranker …")
    suite["rerank_100"] = benchmark_reranker(force_cpu=force_cpu)

    print(f"[{label}] summarize vit5 …")
    try:
        suite["summarize_vit5"] = benchmark_summarize("vit5", runs=3)
    except Exception as exc:
        suite["summarize_vit5"] = {"error": str(exc)}

    if not skip_rag:
        print(f"[{label}] RAG chat …")
        try:
            suite["rag_chat"] = benchmark_rag_chat()
        except Exception as exc:
            suite["rag_chat"] = {"error": str(exc)}

    return suite


def _speedup(before: float, after: float) -> float | None:
    if before <= 0 or after <= 0:
        return None
    return round(before / after, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU benchmark suite")
    parser.add_argument("--label", default="after", choices=["before", "after", "current"])
    parser.add_argument("--force-cpu", action="store_true", help="Simulate CPU baseline")
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Run CPU then GPU and compare")
    args = parser.parse_args()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if RESULTS_PATH.exists():
        try:
            payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    if args.compare:
        print("=== CPU baseline (force) ===")
        payload["before"] = run_suite("before_cpu", force_cpu=True, skip_rag=args.skip_rag)
        print("=== GPU optimized ===")
        payload["after"] = run_suite("after_gpu", force_cpu=False, skip_rag=args.skip_rag)
        b = payload["before"]
        a = payload["after"]
        payload["speedup"] = {
            "embedding": _speedup(b["embedding_100"]["elapsed_s"], a["embedding_100"]["elapsed_s"]),
            "reranker": _speedup(b["rerank_100"]["elapsed_s"], a["rerank_100"]["elapsed_s"]),
        }
        if "avg_s" in b.get("summarize_vit5", {}) and "avg_s" in a.get("summarize_vit5", {}):
            payload["speedup"]["summarize_vit5"] = _speedup(
                b["summarize_vit5"]["avg_s"], a["summarize_vit5"]["avg_s"]
            )
    else:
        payload[args.label] = run_suite(
            args.label,
            force_cpu=args.force_cpu,
            skip_rag=args.skip_rag,
        )

    RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nSaved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
