#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_rag_hotpath.py — Benchmark nhẹ cho hot-path RAG (không cần server HTTP).

Đo: startup import, embedding, retrieval, rerank, generation, tổng latency.
Ghi kết quả JSON vào storage/results/rag_hotpath_benchmark.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Tắt RAPTOR sync để đo upload path nhanh hơn trong benchmark
os.environ.setdefault("RAG_USE_RAPTOR", "0")
os.environ.setdefault("RAG_RESPONSE_CACHE", "0")
os.environ.setdefault("RAG_RETRIEVAL_CACHE", "0")
os.environ.setdefault("RAG_VERBOSE_LOG", "0")
os.environ.setdefault("PRELOAD_RAG_MODELS", "1")

RESULTS_PATH = PROJECT_ROOT / "storage" / "results" / "rag_hotpath_benchmark.json"

SAMPLE_QUERIES = [
    "Tóm tắt mục tiêu nghiên cứu của đề cương.",
    "Mô hình PhoBERT được sử dụng như thế nào?",
    "So sánh PhoBERT và spaCy NER.",
]


def _parse_latency_s(value: str | float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace("s", "").strip() or 0)


def run_benchmark(label: str = "current", *, preload_rag: bool = True) -> dict[str, Any]:
    t_import = time.perf_counter()
    from backend.services.rag.service import RAGChatService
    import_elapsed = time.perf_counter() - t_import

    rag_preload_s = 0.0
    if preload_rag:
        t_preload = time.perf_counter()
        from backend.services.rag.warmup import preload_rag_models
        preload_rag_models()
        rag_preload_s = time.perf_counter() - t_preload

    t_init = time.perf_counter()
    service = RAGChatService()
    init_elapsed = time.perf_counter() - t_init

    docs = service.list_documents()
    if not docs:
        return {
            "label": label,
            "error": "no_documents",
            "import_time_s": round(import_elapsed, 4),
            "rag_preload_s": round(rag_preload_s, 4),
            "init_time_s": round(init_elapsed, 4),
        }

    doc_id = docs[0]["id"]
    latencies: list[dict[str, float]] = []

    for query in SAMPLE_QUERIES:
        t0 = time.perf_counter()
        res = service.chat(
            query=query,
            conversation_id=None,
            document_ids=[doc_id],
        )
        total = time.perf_counter() - t0
        details = res.get("latency_details", {})
        latencies.append({
            "query": query[:40],
            "embedding_s": _parse_latency_s(details.get("embedding")),
            "retrieval_s": _parse_latency_s(details.get("retrieval")),
            "reranking_s": _parse_latency_s(details.get("reranking")),
            "generation_s": _parse_latency_s(details.get("generation")),
            "total_s": _parse_latency_s(details.get("total")) or total,
        })

    def avg(key: str) -> float:
        vals = [row[key] for row in latencies]
        return sum(vals) / max(len(vals), 1)

    # Sentence embed cache micro-benchmark
    t_sent = time.perf_counter()
    sentences = ["Câu thử nghiệm số một.", "Câu thử nghiệm số hai.", "Câu thử nghiệm số một."]
    model = service._resolve_embedding_model([doc_id])
    service.embedding_service.embed_sentences_cached(sentences, model)
    t_sent_miss = time.perf_counter() - t_sent
    t_sent2 = time.perf_counter()
    service.embedding_service.embed_sentences_cached(sentences, model)
    t_sent_hit = time.perf_counter() - t_sent2

    # Query expansion cache micro-benchmark
    from backend.services.rag.agent import expand_query
    t_exp = time.perf_counter()
    expand_query("Mô hình PhoBERT được sử dụng như thế nào?")
    t_exp_cold = time.perf_counter() - t_exp
    t_exp2 = time.perf_counter()
    expand_query("Mô hình PhoBERT được sử dụng như thế nào?")
    t_exp_warm = time.perf_counter() - t_exp2

    report = {
        "label": label,
        "timestamp": int(time.time()),
        "import_time_s": round(import_elapsed, 4),
        "rag_preload_s": round(rag_preload_s, 4),
        "init_time_s": round(init_elapsed, 4),
        "startup_time_s": round(import_elapsed + rag_preload_s + init_elapsed, 4),
        "document_count": len(docs),
        "queries": latencies,
        "avg_embedding_s": round(avg("embedding_s"), 4),
        "avg_retrieval_s": round(avg("retrieval_s"), 4),
        "avg_reranking_s": round(avg("reranking_s"), 4),
        "avg_generation_s": round(avg("generation_s"), 4),
        "avg_total_s": round(avg("total_s"), 4),
        "first_query_total_s": latencies[0]["total_s"] if latencies else 0,
        "warm_query_avg_s": round(
            sum(r["total_s"] for r in latencies[1:]) / max(len(latencies) - 1, 1), 4
        ) if len(latencies) > 1 else 0,
        "sentence_embed_cold_s": round(t_sent_miss, 4),
        "sentence_embed_warm_s": round(t_sent_hit, 4),
        "expansion_cold_s": round(t_exp_cold, 6),
        "expansion_warm_s": round(t_exp_warm, 6),
        "throughput_qps": round(1.0 / max(avg("total_s"), 0.001), 3),
    }
    return report


def main() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = run_benchmark("round3")
    RESULTS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
