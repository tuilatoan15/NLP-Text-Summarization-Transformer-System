"""Production document ingest contracts for summarization and RAG."""

from pipeline.schema import (
    ChunkingConfig,
    DocumentElement,
    DocumentMetadata,
    EmbeddingConfig,
    ExtractionConfig,
    ExtractedDocument,
    IngestConfig,
    IngestResult,
    TextChunk,
)

__all__ = [
    "ChunkingConfig",
    "DocumentElement",
    "DocumentMetadata",
    "EmbeddingConfig",
    "ExtractionConfig",
    "ExtractedDocument",
    "IngestConfig",
    "IngestPipeline",
    "IngestResult",
    "TextChunk",
    "HybridSummarizer",
]


def __getattr__(name: str):
    if name == "IngestPipeline":
        from pipeline.ingest_pipeline import IngestPipeline
        return IngestPipeline
    if name == "HybridSummarizer":
        from pipeline.hybrid_summarizer import HybridSummarizer
        return HybridSummarizer
    raise AttributeError(name)
