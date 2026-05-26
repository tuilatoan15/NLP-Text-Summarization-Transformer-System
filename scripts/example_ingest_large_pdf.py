"""Example config for ingesting a large Vietnamese PDF on limited VRAM.

Run:
    python -m scripts.example_ingest_large_pdf data/bao-cao-dien-luc.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.ingest_pipeline import IngestPipeline
from pipeline.schema import ChunkingConfig, EmbeddingConfig, ExtractionConfig, IngestConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a large Vietnamese PDF with low-VRAM settings.")
    parser.add_argument("pdf_path", type=str)
    parser.add_argument("--output", type=str, default="storage/ingest/large_pdf_ingest.json")
    parser.add_argument("--model", type=str, default="BAAI/bge-m3")
    args = parser.parse_args()

    config = IngestConfig(
        extraction=ExtractionConfig(
            enable_ocr=True,
            ocr_languages=("vie", "eng"),
            preserve_tables=True,
            max_pages=None,
        ),
        chunking=ChunkingConfig(
            target_tokens=360,
            min_tokens=100,
            max_tokens=560,
            overlap_tokens=80,
            semantic_model_name=None,
        ),
        embedding=EmbeddingConfig(
            model_name=args.model,
            batch_size=4,
            use_fp16=True,
            max_seq_length=4096,
            normalize_embeddings=True,
            fallback_to_hashing=False,
        ),
        enable_embeddings=True,
    )
    result = IngestPipeline(config).ingest(args.pdf_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")
    print(f"Chunks: {len(result.chunks)} | Quality: {result.quality.get('extraction', {}).get('score')}")


if __name__ == "__main__":
    main()
