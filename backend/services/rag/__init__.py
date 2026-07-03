import threading

from .service import RAGChatService
from .reranker import CrossEncoderReranker
from .summarizer import RAGTransformerSummarizer
from .rag_config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_FINAL_TOP_K,
    RETRIEVAL_THRESHOLD,
    GENERATION_PROFILES,
)

_rag_service: RAGChatService | None = None
_rag_service_lock = threading.Lock()


def get_rag_service() -> RAGChatService:
    """Return a process-wide singleton RAGChatService (shared models, DB, vector store)."""
    global _rag_service
    if _rag_service is not None:
        return _rag_service
    with _rag_service_lock:
        if _rag_service is None:
            _rag_service = RAGChatService()
        return _rag_service


__all__ = [
    "RAGChatService",
    "get_rag_service",
    "CrossEncoderReranker",
    "RAGTransformerSummarizer",
    "EMBEDDING_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVAL_FINAL_TOP_K",
    "RETRIEVAL_THRESHOLD",
    "GENERATION_PROFILES",
]

