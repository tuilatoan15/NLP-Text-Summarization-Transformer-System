#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_adaptive_context.py — So sánh Old RAG vs Context Compression vs Adaptive Context Builder.

Ghi kết quả: storage/results/adaptive_context_benchmark.json
"""
from __future__ import annotations

import importlib
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
os.environ.setdefault("RAG_ADAPTIVE_CONTEXT_CACHE", "0")
os.environ.setdefault("RAG_VERBOSE_LOG", "0")
os.environ.setdefault("PRELOAD_RAG_MODELS", "0")

RESULTS_PATH = PROJECT_ROOT / "storage" / "results" / "adaptive_context_benchmark.json"

SAMPLE_QUERIES = [
    "Tóm tắt mục tiêu nghiên cứu chính trong tài liệu.",
    "Các phương pháp NLP nào được đề cập?",
    "Kết quả thực nghiệm và số liệu quan trọng là gì?",
    "So sánh các thuật toán extractive và abstractive.",
]


def _parse_latency_s(value: str | float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace("s", "").strip() or 0)


def _reload_modules() -> None:
    import backend.services.rag.rag_config as rag_config
    importlib.reload(rag_config)
    import backend.services.rag.context_compression as cc
    importlib.reload(cc)
    import backend.services.rag.adaptive_context_builder as acb
    importlib.reload(acb)


def _run_mode(label: str, *, adaptive: bool, compression: bool) -> dict[str, Any]:
    os.environ["RAG_ADAPTIVE_CONTEXT"] = "1" if adaptive else "0"
    os.environ["RAG_CONTEXT_COMPRESSION"] = "1" if compression else "0"
    _reload_modules()

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
        ctx = res.get("context_details") or {}
        prompt = res.get("prompt_template", "")
        rows.append({
            "query": query[:60],
            "embedding_s": _parse_latency_s(details.get("embedding")),
            "retrieval_s": _parse_latency_s(details.get("retrieval")),
            "reranking_s": _parse_latency_s(details.get("reranking")),
            "context_build_s": _parse_latency_s(details.get("context_compression")),
            "generation_s": _parse_latency_s(details.get("generation")),
            "total_s": _parse_latency_s(details.get("total")) or wall,
            "compression_enabled": cc.get("enabled", False),
            "compression_ratio": cc.get("compression_ratio") or details.get("compression_ratio"),
            "compression_tier": cc.get("compression_tier") or ctx.get("compression_tier"),
            "token_reduction": ctx.get("token_reduction") or details.get("token_reduction"),
            "chunks_kept": ctx.get("chunks_kept") or cc.get("top_original_count"),
            "summary_tokens": ctx.get("summary_tokens") or cc.get("summary_tokens"),
            "prompt_chars": len(prompt),
            "answer_chars": len(res.get("answer", "")),
            "mode": cc.get("mode", "legacy"),
        })

    def avg(key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return sum(vals) / max(len(vals), 1)

    return {
        "label": label,
        "adaptive": adaptive,
        "compression": compression,
        "queries": rows,
        "avg_total_s": round(avg("total_s"), 4),
        "avg_context_build_s": round(avg("context_build_s"), 4),
        "avg_generation_s": round(avg("generation_s"), 4),
        "avg_prompt_chars": round(avg("prompt_chars"), 1),
        "avg_compression_ratio": round(avg("compression_ratio"), 4),
        "avg_token_reduction": round(avg("token_reduction"), 4),
        "avg_chunks_kept": round(avg("chunks_kept"), 2),
    }


def main() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    old_rag = _run_mode("old_rag", adaptive=False, compression=False)
    context_compression = _run_mode("context_compression", adaptive=False, compression=True)
    adaptive_builder = _run_mode("adaptive_context_builder", adaptive=True, compression=True)

    report: dict[str, Any] = {
        "timestamp": int(time.time()),
        "modes": {
            "old_rag": old_rag,
            "context_compression": context_compression,
            "adaptive_context_builder": adaptive_builder,
        },
        "delta": {},
    }

    if not any(m.get("error") for m in report["modes"].values()):
        old = old_rag
        cc = context_compression
        acb = adaptive_builder
        report["delta"] = {
            "adaptive_vs_old_total_s": round(acb["avg_total_s"] - old["avg_total_s"], 4),
            "adaptive_vs_old_prompt_reduction": round(
                1.0 - acb["avg_prompt_chars"] / max(old["avg_prompt_chars"], 1), 4
            ),
            "adaptive_vs_compression_prompt_reduction": round(
                1.0 - acb["avg_prompt_chars"] / max(cc["avg_prompt_chars"], 1), 4
            ),
            "adaptive_token_reduction": acb["avg_token_reduction"],
            "adaptive_chunks_kept": acb["avg_chunks_kept"],
        }

    RESULTS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
