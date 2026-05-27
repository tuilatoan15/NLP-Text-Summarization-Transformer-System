from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RAGSettings:
    top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 80
    temperature: float = 0.2
    threshold: float = 0.35
    retrieval_mode: str = "hybrid"
    use_reranking: bool = False
    embedding_model: str = "BAAI/bge-m3"


@dataclass(slots=True)
class ChunkRecord:
    id: str
    document_id: str
    filename: str
    text: str
    page: int | None
    chunk_index: int
    embedding_model: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalHit:
    chunk_id: str
    document_id: str
    filename: str
    page: int | None
    text: str
    embedding_score: float
    bm25_score: float
    combined_score: float
    rank: int

