"""End-to-end benchmark for the Document Intelligence workflow.

Example:
    python -m scripts.benchmark_document_intelligence data/report.pdf --embedding-model hash --algorithms textrank lexrank lsa
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.document_intelligence import DocumentIntelligenceService


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ingest, retrieval, summarization and citation grounding.")
    parser.add_argument("path", type=str)
    parser.add_argument("--query", type=str, default="nội dung chính của tài liệu")
    parser.add_argument("--reference", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default="hash")
    parser.add_argument("--algorithms", nargs="*", default=["textrank", "lexrank", "lsa"])
    parser.add_argument("--output", type=str, default="storage/results/document_intelligence_benchmark.json")
    args = parser.parse_args()

    service = DocumentIntelligenceService()
    t0 = time.perf_counter()
    document = service.ingest_file(args.path, include_embeddings=True, embedding_model=args.embedding_model)
    ingest_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    search = service.semantic_search(document["document_id"], args.query, top_k=5)
    search_seconds = time.perf_counter() - t1

    t2 = time.perf_counter()
    compare = service.compare_summaries(
        document["document_id"],
        reference=args.reference,
        algorithms=args.algorithms,
        target_length_ratio=35,
    )
    compare_seconds = time.perf_counter() - t2

    payload = {
        "document_id": document["document_id"],
        "metadata": document.get("metadata", {}),
        "timing": {
            "ingest_seconds": round(ingest_seconds, 4),
            "search_seconds": round(search_seconds, 4),
            "compare_seconds": round(compare_seconds, 4),
            "total_seconds": round(time.perf_counter() - t0, 4),
        },
        "quality": document.get("quality", {}),
        "retrieval": {
            "query": args.query,
            "top_scores": [row["score"] for row in search.get("results", [])],
        },
        "summary_research_matrix": compare.get("research_matrix", {}),
        "citation_coverage": [
            {
                "algorithm": row.get("algorithm"),
                "grounded": sum(1 for item in row.get("citations", []) if item.get("status") == "grounded"),
                "total": len(row.get("citations", [])),
            }
            for row in compare.get("results", [])
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
