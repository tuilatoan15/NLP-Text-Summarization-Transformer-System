"""Benchmark extraction, chunking, embeddings, and retrieval quality impact."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from embeddings.embedder import EmbeddingConfig, EmbeddingModelRegistry, SentenceTransformerEmbedder
from pipeline.ingest_pipeline import IngestPipeline
from pipeline.schema import IngestConfig
from utils.logger import logger
from utils.metrics import chunk_coherence_score, lexical_overlap, retrieval_accuracy_at_k


DEFAULT_MODELS = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "jinaai/jina-embeddings-v3",
    "keepitreal/vietnamese-sbert",
    "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
]


def benchmark_document(
    path: str | Path,
    models: list[str] | None = None,
    query: str | None = None,
    reference_summary: str | None = None,
    config: IngestConfig | None = None,
) -> dict[str, Any]:
    models = models or DEFAULT_MODELS
    base_config = config or IngestConfig()
    base_config.enable_embeddings = False

    extraction_start = time.perf_counter()
    ingest_result = IngestPipeline(base_config).ingest(path, include_embeddings=False)
    extraction_time = time.perf_counter() - extraction_start
    chunks = ingest_result.chunks
    chunk_texts = [chunk["text"] for chunk in chunks]

    model_results = []
    for model_name in models:
        logger.info("Benchmark embedding model: %s", model_name)
        embedding_config = EmbeddingConfig(
            model_name=model_name,
            batch_size=base_config.embedding.batch_size,
            device=base_config.embedding.device,
            normalize_embeddings=True,
            fallback_to_hashing=base_config.embedding.fallback_to_hashing,
            show_progress=base_config.embedding.show_progress,
        )
        embedder = SentenceTransformerEmbedder(embedding_config)
        start = time.perf_counter()
        batch = embedder.embed_documents(chunk_texts)
        embedding_time = time.perf_counter() - start

        retrieval = {}
        if query:
            query_vector = embedder.embed_query(query)
            relevant = _infer_relevant_chunks(query, chunk_texts, reference_summary)
            retrieval = retrieval_accuracy_at_k(query_vector, batch.embeddings, relevant, k=min(5, len(chunks)))

        summarization_proxy = {}
        if reference_summary:
            top_text = _top_k_text(reference_summary, embedder, batch.embeddings, chunk_texts, k=min(5, len(chunks)))
            summarization_proxy = {
                "reference_lexical_coverage_top_chunks": lexical_overlap(reference_summary, top_text),
                "top_chunk_chars": len(top_text),
            }

        model_results.append(
            {
                "model_name": model_name,
                "registry": EmbeddingModelRegistry.defaults_for(model_name),
                "provider": batch.provider,
                "dimension": batch.dimension,
                "embedding_seconds": round(embedding_time, 4),
                "chunks_per_second": round(len(chunks) / max(embedding_time, 1e-9), 2),
                "coherence": chunk_coherence_score(batch.embeddings),
                "retrieval": retrieval,
                "summarization_quality_proxy": summarization_proxy,
            }
        )

    return {
        "document": {
            "path": str(path),
            "document_id": ingest_result.document_id,
            "metadata": ingest_result.metadata,
        },
        "extraction": {
            "seconds": round(extraction_time, 4),
            "quality": ingest_result.quality.get("extraction", {}),
        },
        "chunking": {
            "chunk_count": len(chunks),
            "avg_chunk_tokens": ingest_result.quality.get("avg_chunk_tokens"),
            "coverage": ingest_result.quality.get("coverage"),
        },
        "models": model_results,
    }


def _infer_relevant_chunks(query: str, chunks: list[str], reference_summary: str | None = None) -> list[int]:
    anchor = reference_summary or query
    scored = [(idx, lexical_overlap(anchor, chunk)) for idx, chunk in enumerate(chunks)]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, score in scored[: max(1, min(3, len(scored)))] if score > 0]


def _top_k_text(
    reference_summary: str,
    embedder: SentenceTransformerEmbedder,
    chunk_embeddings: np.ndarray,
    chunks: list[str],
    k: int,
) -> str:
    if not chunks:
        return ""
    query_vector = embedder.embed_query(reference_summary)
    sims = query_vector.reshape(1, -1) @ chunk_embeddings.T
    top = list(np.argsort(-sims.reshape(-1))[:k])
    return "\n\n".join(chunks[idx] for idx in top)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ingest and embedding models.")
    parser.add_argument("path", type=str, help="Path to PDF/DOCX/TXT.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Embedding model names.")
    parser.add_argument("--query", type=str, default=None, help="Optional semantic retrieval query.")
    parser.add_argument("--reference-summary", type=str, default=None, help="Optional reference summary text.")
    parser.add_argument("--config", type=str, default="configs/ingest.json")
    parser.add_argument("--output", type=str, default="storage/results/ingest_benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = IngestConfig.from_json(args.config) if Path(args.config).exists() else IngestConfig()
    result = benchmark_document(args.path, args.models, args.query, args.reference_summary, config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Benchmark saved: %s", output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
