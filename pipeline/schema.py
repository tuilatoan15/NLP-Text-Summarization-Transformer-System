"""Typed data contracts shared by loaders, cleaners, chunkers, and embedders."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


ElementType = Literal["title", "heading", "paragraph", "bullet", "table", "caption", "footer", "unknown"]


@dataclass(slots=True)
class DocumentMetadata:
    source_path: str
    source_type: str
    title: str | None = None
    author: str | None = None
    pages: int | None = None
    language: str | None = None
    created_at: str | None = None
    modified_at: str | None = None
    producer: str | None = None
    extraction_engine: str | None = None
    is_scanned: bool = False
    quality_score: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocumentElement:
    text: str
    element_type: ElementType = "paragraph"
    page_number: int | None = None
    section_path: list[str] = field(default_factory=list)
    level: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExtractedDocument:
    document_id: str
    metadata: DocumentMetadata
    text: str
    elements: list[DocumentElement] = field(default_factory=list)
    structure: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "metadata": self.metadata.to_dict(),
            "text": self.text,
            "elements": [element.to_dict() for element in self.elements],
            "structure": self.structure,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class TextChunk:
    chunk_id: str
    document_id: str
    text: str
    index: int
    token_count: int
    word_count: int
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = field(default_factory=list)
    source_element_ids: list[int] = field(default_factory=list)
    overlap_from_previous: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExtractionConfig:
    pdf_primary_engine: str = "pymupdf"
    pdf_fallback_engines: tuple[str, ...] = ("pdfplumber", "unstructured")
    docx_primary_engine: str = "python-docx"
    docx_use_mammoth_fallback: bool = True
    enable_ocr: bool = True
    ocr_languages: tuple[str, ...] = ("vie", "eng")
    scanned_text_min_chars_per_page: int = 40
    scanned_text_min_words_per_page: int = 8
    preserve_tables: bool = True
    preserve_bullets: bool = True
    max_pages: int | None = None


@dataclass(slots=True)
class CleaningConfig:
    remove_headers_footers: bool = True
    min_repeated_header_pages: int = 2
    normalize_unicode: bool = True
    remove_page_numbers: bool = True
    reconstruct_paragraphs: bool = True
    semantic_filtering: bool = True
    min_quality_score: float = 0.25
    clean_vietnamese_admin: bool = True


@dataclass(slots=True)
class ChunkingConfig:
    target_tokens: int = 420
    min_tokens: int = 120
    max_tokens: int = 640
    overlap_tokens: int = 64
    dynamic_size: bool = True
    respect_headings: bool = True
    split_long_paragraphs: bool = True
    semantic_model_name: str | None = None
    semantic_similarity_threshold: float = 0.58
    token_model_name: str | None = None
    use_vietnamese_segmenter: bool = False


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 16
    device: str | None = "cpu"
    normalize_embeddings: bool = True
    trust_remote_code: bool = True
    max_seq_length: int = 8192
    use_fp16: bool = False
    show_progress: bool = True
    fallback_to_hashing: bool = True
    query_prefix: str | None = None
    passage_prefix: str | None = None


@dataclass(slots=True)
class IngestConfig:
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    enable_embeddings: bool = True
    fail_on_low_quality: bool = False
    output_embeddings_as_list: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IngestConfig":
        return cls(
            extraction=ExtractionConfig(**data.get("extraction", {})),
            cleaning=CleaningConfig(**data.get("cleaning", {})),
            chunking=ChunkingConfig(**data.get("chunking", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            enable_embeddings=bool(data.get("enable_embeddings", True)),
            fail_on_low_quality=bool(data.get("fail_on_low_quality", False)),
            output_embeddings_as_list=bool(data.get("output_embeddings_as_list", True)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "IngestConfig":
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IngestResult:
    document_id: str
    metadata: dict[str, Any]
    clean_text: str
    chunks: list[dict[str, Any]]
    embeddings: list[list[float]] | None
    structure: dict[str, Any]
    quality: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "metadata": self.metadata,
            "clean_text": self.clean_text,
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "structure": self.structure,
            "quality": self.quality,
            "warnings": self.warnings,
        }
