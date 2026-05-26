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
]


def __getattr__(name: str):
    if name == "IngestPipeline":
        from pipeline.ingest_pipeline import IngestPipeline

        return IngestPipeline
    raise AttributeError(name)
