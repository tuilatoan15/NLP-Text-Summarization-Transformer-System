"""End-to-end ingest pipeline for high-fidelity summarization and RAG."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Iterable

from embeddings.embedder import SentenceTransformerEmbedder
from loaders.docx_loader import DOCXLoader
from loaders.pdf_loader import PDFLoader
from loaders.txt_loader import TXTLoader
from pipeline.schema import IngestConfig, IngestResult
from preprocess.chunker import SemanticChunker
from preprocess.cleaner import DocumentCleaner
from utils.logger import logger
from utils.metrics import chunk_coherence_score, compression_coverage_proxy


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class IngestPipeline:
    """Config-driven orchestrator for extraction, cleaning, chunking, and embeddings."""

    def __init__(self, config: IngestConfig | None = None) -> None:
        self.config = config or IngestConfig()
        self.cleaner = DocumentCleaner(self.config.cleaning)
        self.chunker = SemanticChunker(self.config.chunking)
        self.embedder = SentenceTransformerEmbedder(self.config.embedding) if self.config.enable_embeddings else None

    def ingest(self, path: str | Path, include_embeddings: bool | None = None) -> IngestResult:
        start = time.perf_counter()
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported document type: {file_path.suffix}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

        logger.info("Ingest start: %s", file_path)
        extracted = self._load(file_path)
        cleaned = self.cleaner.clean(extracted)
        if self.config.fail_on_low_quality and float(cleaned.quality.get("score", 0.0)) < self.config.cleaning.min_quality_score:
            raise ValueError(f"Extraction quality too low: {cleaned.quality}")

        chunks = self.chunker.chunk(extracted.document_id, cleaned.elements, fallback_text=cleaned.text)
        chunk_dicts = [chunk.to_dict() for chunk in chunks]

        should_embed = self.config.enable_embeddings if include_embeddings is None else include_embeddings
        embeddings_list: list[list[float]] | None = None
        embedding_quality: dict[str, Any] = {}
        if should_embed and self.embedder is not None and chunks:
            embedding_result = self.embedder.embed_documents([chunk.text for chunk in chunks])
            embeddings = embedding_result.embeddings
            if self.config.output_embeddings_as_list:
                embeddings_list = embeddings.tolist()
            embedding_quality = {
                "embedding_model": embedding_result.model_name,
                "embedding_provider": embedding_result.provider,
                "embedding_dimension": embedding_result.dimension,
                **chunk_coherence_score(embeddings),
            }

        quality = {
            "extraction": cleaned.quality,
            "chunk_count": len(chunks),
            "avg_chunk_tokens": round(sum(chunk.token_count for chunk in chunks) / max(1, len(chunks)), 2),
            "coverage": compression_coverage_proxy(cleaned.text, [chunk.text for chunk in chunks]),
            "elapsed_seconds": round(time.perf_counter() - start, 4),
            **embedding_quality,
        }
        structure = {
            **extracted.structure,
            "chunking": {
                "strategy": "heading+paragraph+semantic-boundary+token-aware",
                "target_tokens": self.config.chunking.target_tokens,
                "max_tokens": self.config.chunking.max_tokens,
                "overlap_tokens": self.config.chunking.overlap_tokens,
            },
        }
        logger.info(
            "Ingest complete: %s chunks, quality=%s, elapsed=%.3fs",
            len(chunks),
            cleaned.quality.get("score"),
            time.perf_counter() - start,
        )
        return IngestResult(
            document_id=extracted.document_id,
            metadata=extracted.metadata.to_dict(),
            clean_text=cleaned.text,
            chunks=chunk_dicts,
            embeddings=embeddings_list,
            structure=structure,
            quality=quality,
            warnings=cleaned.warnings,
        )

    async def ingest_async(self, path: str | Path, include_embeddings: bool | None = None) -> IngestResult:
        return await asyncio.to_thread(self.ingest, path, include_embeddings)

    async def ingest_many_async(
        self,
        paths: Iterable[str | Path],
        include_embeddings: bool | None = None,
        concurrency: int = 2,
    ) -> list[IngestResult]:
        semaphore = asyncio.Semaphore(concurrency)

        async def _run(item: str | Path) -> IngestResult:
            async with semaphore:
                return await self.ingest_async(item, include_embeddings)

        return await asyncio.gather(*[_run(path) for path in paths])

    def _load(self, path: Path):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return PDFLoader(self.config.extraction).load(path)
        if suffix == ".docx":
            return DOCXLoader(self.config.extraction).load(path)
        if suffix in {".txt", ".md"}:
            return TXTLoader().load(path)
        raise ValueError(f"Unsupported document type: {suffix}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a PDF/DOCX/TXT document for summarization and RAG.")
    parser.add_argument("path", type=str, help="Path to PDF, DOCX, TXT, or MD document.")
    parser.add_argument("--config", type=str, default="configs/ingest.json", help="JSON config path.")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip embedding generation.")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = IngestConfig.from_json(args.config) if Path(args.config).exists() else IngestConfig()
    pipeline = IngestPipeline(config)
    result = pipeline.ingest(args.path, include_embeddings=not args.no_embeddings)
    payload = result.to_dict()
    output = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output, encoding="utf-8")
        logger.info("Saved ingest result: %s", target)
    else:
        print(output)


if __name__ == "__main__":
    main()
