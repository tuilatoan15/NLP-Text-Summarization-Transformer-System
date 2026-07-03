from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

from pipeline.schema import EmbeddingConfig
from embeddings.embedder import SentenceTransformerEmbedder, EmbeddingModelRegistry
from src.utils import resolve_torch_device_str
from .rag_config import RAG_EMBEDDING_FP16, RAG_EMBEDDING_BATCH_SIZE, RAG_SENTENCE_EMBED_CACHE


class _QueryEmbeddingCache:
    """Thread-safe LRU cache for query embeddings (avoids re-embedding identical queries)."""

    def __init__(self, maxsize: int = 512) -> None:
        self.maxsize = maxsize
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    def _key(self, text: str, model_name: str) -> str:
        digest = hashlib.blake2b(
            f"{model_name}\0{text}".encode("utf-8", errors="ignore"),
            digest_size=16,
        ).hexdigest()
        return digest

    def get(self, text: str, model_name: str) -> list[float] | None:
        key = self._key(text, model_name)
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def set(self, text: str, model_name: str, vector: list[float]) -> None:
        key = self._key(text, model_name)
        with self._lock:
            self._cache[key] = vector
            self._cache.move_to_end(key)
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)


_query_cache = _QueryEmbeddingCache(maxsize=512)
_embedder_lock = threading.Lock()
_embedders: dict[str, SentenceTransformerEmbedder] = {}


class _SentenceEmbeddingCache:
    """LRU cache cho embedding từng câu — tăng tốc semantic chunking khi upload."""

    def __init__(self, maxsize: int = 4096) -> None:
        self.maxsize = maxsize
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    def _key(self, text: str, model_name: str) -> str:
        digest = hashlib.blake2b(
            f"{model_name}\0{text}".encode("utf-8", errors="ignore"),
            digest_size=16,
        ).hexdigest()
        return digest

    def get_many(
        self, texts: list[str], model_name: str
    ) -> tuple[list[int], list[list[float] | None]]:
        """Trả về (indices cần embed, vectors đã cache theo thứ tự texts)."""
        cached: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        with self._lock:
            for i, text in enumerate(texts):
                key = self._key(text, model_name)
                if key in self._cache:
                    self._cache.move_to_end(key)
                    cached[i] = self._cache[key]
                else:
                    missing_indices.append(i)
        return missing_indices, cached

    def set_many(self, texts: list[str], model_name: str, vectors: list[list[float]]) -> None:
        with self._lock:
            for text, vector in zip(texts, vectors):
                key = self._key(text, model_name)
                self._cache[key] = vector
                self._cache.move_to_end(key)
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)


_sentence_cache = _SentenceEmbeddingCache(maxsize=4096)


class EmbeddingService:
    def list_models(self) -> dict:
        return EmbeddingModelRegistry.list_models()

    def _get_embedder(self, model_name: str) -> SentenceTransformerEmbedder:
        if model_name in _embedders:
            return _embedders[model_name]
        with _embedder_lock:
            if model_name not in _embedders:
                config = EmbeddingConfig(
                    model_name=model_name,
                    device=resolve_torch_device_str(),
                    use_fp16=RAG_EMBEDDING_FP16,
                    batch_size=RAG_EMBEDDING_BATCH_SIZE,
                )
                _embedders[model_name] = SentenceTransformerEmbedder(config)
            return _embedders[model_name]

    def embed_documents(self, texts: list[str], model_name: str) -> list[list[float]]:
        result = self._get_embedder(model_name).embed_documents(texts)
        return result.embeddings.tolist()

    def embed_query(self, text: str, model_name: str) -> list[float]:
        cached = _query_cache.get(text, model_name)
        if cached is not None:
            return cached
        vector = self._get_embedder(model_name).embed_query(text).tolist()
        _query_cache.set(text, model_name, vector)
        return vector

    def embed_queries_batch(self, texts: list[str], model_name: str) -> list[list[float]]:
        """Embed nhiều query — tái sử dụng cache, batch phần còn thiếu."""
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        for i, text in enumerate(texts):
            cached = _query_cache.get(text, model_name)
            if cached is not None:
                results[i] = cached
            else:
                missing_indices.append(i)
                missing_texts.append(text)
        if missing_texts:
            fresh = self.embed_documents(missing_texts, model_name)
            for idx, text, vec in zip(missing_indices, missing_texts, fresh):
                _query_cache.set(text, model_name, vec)
                results[idx] = vec
        return [v for v in results if v is not None]

    def embed_sentences_cached(self, texts: list[str], model_name: str) -> list[list[float]]:
        """Embed danh sách câu với LRU cache — dùng cho semantic chunking."""
        if not texts:
            return []
        if not RAG_SENTENCE_EMBED_CACHE:
            return self.embed_documents(texts, model_name)

        missing_indices, result = _sentence_cache.get_many(texts, model_name)
        if not missing_indices:
            return [v for v in result if v is not None]

        missing_texts = [texts[i] for i in missing_indices]
        fresh_vectors = self.embed_documents(missing_texts, model_name)
        _sentence_cache.set_many(missing_texts, model_name, fresh_vectors)

        for idx, vec in zip(missing_indices, fresh_vectors):
            result[idx] = vec
        return [v for v in result if v is not None]
