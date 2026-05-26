#!/usr/bin/env python3
"""Compare hash vs sentence-transformer embedding models on a document or sample text."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from embeddings.benchmark import benchmark_document, DEFAULT_MODELS
from embeddings.embedder import EmbeddingConfig, SentenceTransformerEmbedder
from pipeline.schema import IngestConfig


SAMPLE_VI_TEXT = (
    "Nhu cầu tiêu thụ điện trong mùa nắng nóng tiếp tục tăng cao tại nhiều địa phương. "
    "Các nhà máy thủy điện miền Bắc phải vận hành thận trọng do mực nước hồ chứa chưa phục hồi. "
    "EVN khuyến nghị theo dõi phụ tải giờ cao điểm và duy trì nguồn điện dự phòng."
)


def benchmark_hash(texts: list[str]) -> dict:
    embedder = SentenceTransformerEmbedder(EmbeddingConfig(model_name="hash", show_progress=False))
    start = time.perf_counter()
    batch = embedder.embed_documents(texts)
    elapsed = time.perf_counter() - start
    return {
        "model_name": "hash",
        "provider": batch.provider,
        "dimension": batch.dimension,
        "elapsed_seconds": round(elapsed, 4),
        "docs_per_second": round(len(texts) / max(elapsed, 1e-6), 2),
    }


def benchmark_from_text(text: str, models: list[str]) -> dict:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    texts = sentences or [text]
    rows = [benchmark_hash(texts)]
    for model in models:
        cfg = EmbeddingConfig(model_name=model, show_progress=False, fallback_to_hashing=True)
        embedder = SentenceTransformerEmbedder(cfg)
        start = time.perf_counter()
        try:
            batch = embedder.embed_documents(texts)
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "model_name": model,
                    "provider": batch.provider,
                    "dimension": batch.dimension,
                    "elapsed_seconds": round(elapsed, 4),
                    "docs_per_second": round(len(texts) / max(elapsed, 1e-6), 2),
                }
            )
        except Exception as exc:
            rows.append({"model_name": model, "error": str(exc)})
    return {"mode": "text", "samples": len(texts), "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding benchmark: hash vs ST models")
    parser.add_argument("--document", type=str, help="Path to PDF/DOCX/TXT")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS[:3]))
    parser.add_argument("--query", default="thủy điện miền Bắc")
    parser.add_argument("--output", default="research/embedding_benchmark.json")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.document:
        payload = benchmark_document(
            args.document,
            models=["hash", *models],
            query=args.query,
            config=IngestConfig(enable_embeddings=False),
        )
    else:
        payload = benchmark_from_text(SAMPLE_VI_TEXT, models)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
