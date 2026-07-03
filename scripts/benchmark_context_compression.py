#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_context_compression.py — So sánh latency/quality trước và sau Context Compression.

Ghi kết quả: storage/results/context_compression_benchmark.json
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

os.environ.setdefault("RAG_USE_RAPTOR", "0")
os.environ.setdefault("RAG_RESPONSE_CACHE", "0")
os.environ.setdefault("RAG_RETRIEVAL_CACHE", "0")
os.environ.setdefault("RAG_VERBOSE_LOG", "0")
os.environ.setdefault("PRELOAD_RAG_MODELS", "0")

RESULTS_PATH = PROJECT_ROOT / "storage" / "results" / "context_compression_benchmark.json"

SAMPLE_QUERIES = [
    "Tóm tắt mục tiêu nghiên cứu chính trong tài liệu.",
    "Các phương pháp NLP nào được đề cập?",
    "Kết quả thực nghiệm và số liệu quan trọng là gì?",
]


def _parse_latency_s(value: str | float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace("s", "").strip() or 0)


def _run_mode(label: str, *, compression_on: bool) -> dict[str, Any]:
    os.environ["RAG_CONTEXT_COMPRESSION"] = "1" if compression_on else "0"

    # Reload config-dependent modules
    import importlib
    import backend.services.rag.rag_config as rag_config
    importlib.reload(rag_config)

    from backend.services.rag.service import RAGChatService

    service = RAGChatService()
    docs = service.list_documents()
    if not docs:
        return {"label": label, "error": "no_documents"}

    doc_id = docs[0]["id"]
    rows: list[dict[str, Any]] = []

    for query in SAMPLE_QUERIES:
        t0 = time.perf_counter()
        res = service.chat(query=query, conversation_id=None, document_ids=[doc_id])
        wall = time.perf_counter() - t0
        details = res.get("latency_details", {})
        cc = res.get("context_compression", {})
        prompt = res.get("prompt_template", "")
        rows.append({
            "query": query[:50],
            "embedding_s": _parse_latency_s(details.get("embedding")),
            "retrieval_s": _parse_latency_s(details.get("retrieval")),
            "reranking_s": _parse_latency_s(details.get("reranking")),
            "hybrid_summary_s": _parse_latency_s(details.get("hybrid_summary")),
            "generation_s": _parse_latency_s(details.get("generation")),
            "total_s": _parse_latency_s(details.get("total")) or wall,
            "compression_enabled": cc.get("enabled", False),
            "compression_ratio": cc.get("compression_ratio") or details.get("compression_ratio"),
            "prompt_chars": len(prompt),
            "answer_chars": len(res.get("answer", "")),
        })

    def avg(key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return sum(vals) / max(len(vals), 1)

    return {
        "label": label,
        "compression_on": compression_on,
        "queries": rows,
        "avg_total_s": round(avg("total_s"), 4),
        "avg_hybrid_summary_s": round(avg("hybrid_summary_s"), 4),
        "avg_generation_s": round(avg("generation_s"), 4),
        "avg_prompt_chars": round(avg("prompt_chars"), 1),
        "avg_compression_ratio": round(avg("compression_ratio"), 4),
    }


def main() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    before = _run_mode("before_compression_off", compression_on=False)
    after = _run_mode("after_compression_on", compression_on=True)

    report = {
        "timestamp": int(time.time()),
        "before": before,
        "after": after,
        "delta": {},
    }
    if not before.get("error") and not after.get("error"):
        report["delta"] = {
            "total_latency_delta_s": round(after["avg_total_s"] - before["avg_total_s"], 4),
            "prompt_chars_reduction": round(
                1.0 - after["avg_prompt_chars"] / max(before["avg_prompt_chars"], 1), 4
            ),
            "generation_latency_delta_s": round(
                after["avg_generation_s"] - before["avg_generation_s"], 4
            ),
        }

    RESULTS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
