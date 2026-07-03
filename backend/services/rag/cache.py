"""
cache.py — Cache hiệu năng cho pipeline RAG (LRU + Redis tùy chọn).

Cache:
  - embedding query (đã có trong embedding_service)
  - intent classification
  - kết quả retrieval + rerank
  - phản hồi chat đầy đủ (query → answer)
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

from .rag_config import (
    RAG_RESPONSE_CACHE,
    RAG_RESPONSE_CACHE_TTL,
    RAG_RETRIEVAL_CACHE,
    RAG_RETRIEVAL_CACHE_TTL,
    RAG_EXPANSION_CACHE,
    RAG_EXPANSION_CACHE_TTL,
    RAG_ADAPTIVE_CONTEXT_CACHE,
    RAG_ADAPTIVE_CONTEXT_CACHE_TTL,
)


def _stable_hash(parts: list[str]) -> str:
    payload = "\x1f".join(parts).encode("utf-8", errors="ignore")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


class _TTLCache:
    """Thread-safe LRU cache với TTL đơn giản."""

    def __init__(self, maxsize: int = 256, ttl_seconds: int = 300) -> None:
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str, now: float) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, value = entry
            if now - ts > self.ttl_seconds:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any, now: float) -> None:
        with self._lock:
            self._store[key] = (now, value)
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def delete_prefix(self, prefix: str) -> int:
        removed = 0
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for key in keys:
                self._store.pop(key, None)
                removed += 1
        return removed

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_intent_cache = _TTLCache(maxsize=512, ttl_seconds=3600)
_expansion_cache = _TTLCache(maxsize=512, ttl_seconds=RAG_EXPANSION_CACHE_TTL)
_retrieval_cache = _TTLCache(maxsize=256, ttl_seconds=RAG_RETRIEVAL_CACHE_TTL)
_response_cache = _TTLCache(maxsize=128, ttl_seconds=RAG_RESPONSE_CACHE_TTL)
_adaptive_context_cache = _TTLCache(maxsize=128, ttl_seconds=RAG_ADAPTIVE_CONTEXT_CACHE_TTL)
_redis_service = None
_redis_lock = threading.Lock()


def _redis():
    global _redis_service
    if _redis_service is not None:
        return _redis_service
    with _redis_lock:
        if _redis_service is None:
            try:
                from backend.services.cache_service import CacheService

                _redis_service = CacheService()
            except Exception:
                _redis_service = False
        return _redis_service if _redis_service is not False else None


def intent_cache_key(query: str) -> str:
    return _stable_hash(["intent", query.strip().lower()])


def get_cached_intent(query: str) -> str | None:
    import time

    return _intent_cache.get(intent_cache_key(query), time.monotonic())


def set_cached_intent(query: str, intent: str) -> None:
    import time

    _intent_cache.set(intent_cache_key(query), intent, time.monotonic())


def expansion_cache_key(query: str) -> str:
    return _stable_hash(["expansion", query.strip().lower()])


def get_cached_expansion(query: str) -> list[str] | None:
    if not RAG_EXPANSION_CACHE:
        return None
    import time

    cached = _expansion_cache.get(expansion_cache_key(query), time.monotonic())
    if cached is not None:
        return list(cached)
    return None


def set_cached_expansion(query: str, variants: list[str]) -> None:
    if not RAG_EXPANSION_CACHE:
        return
    import time

    _expansion_cache.set(expansion_cache_key(query), list(variants), time.monotonic())


def retrieval_cache_key(
    query: str,
    document_ids: list[str] | None,
    embedding_model: str,
    retrieval_mode: str,
    use_reranking: bool,
    top_k: int,
    threshold: float,
) -> str:
    doc_part = ",".join(sorted(document_ids or []))
    return _stable_hash(
        [
            "retrieval",
            query.strip(),
            doc_part,
            embedding_model,
            retrieval_mode,
            str(use_reranking),
            str(top_k),
            f"{threshold:.4f}",
        ]
    )


def get_cached_retrieval(key: str) -> list[dict[str, Any]] | None:
    if not RAG_RETRIEVAL_CACHE:
        return None
    import time

    cached = _retrieval_cache.get(key, time.monotonic())
    if cached is not None:
        return cached
    svc = _redis()
    if svc:
        raw = svc.get_json(f"rag:retrieval:{key}")
        if raw and "chunks" in raw:
            return raw["chunks"]
    return None


def set_cached_retrieval(key: str, chunks: list[dict[str, Any]]) -> None:
    if not RAG_RETRIEVAL_CACHE:
        return
    import time

    _retrieval_cache.set(key, chunks, time.monotonic())
    svc = _redis()
    if svc:
        svc.set_json(
            f"rag:retrieval:{key}",
            {"chunks": chunks},
            ttl_seconds=RAG_RETRIEVAL_CACHE_TTL,
        )


def response_cache_key(
    query: str,
    document_ids: list[str] | None,
    conversation_id: str | None,
    embedding_model: str,
    retrieval_mode: str,
    use_reranking: bool,
    top_k: int,
    threshold: float,
    temperature: float,
) -> str:
    doc_part = ",".join(sorted(document_ids or []))
    return _stable_hash(
        [
            "response",
            query.strip(),
            doc_part,
            conversation_id or "",
            embedding_model,
            retrieval_mode,
            str(use_reranking),
            str(top_k),
            f"{threshold:.4f}",
            f"{temperature:.2f}",
        ]
    )


def get_cached_response(key: str) -> dict[str, Any] | None:
    if not RAG_RESPONSE_CACHE:
        return None
    import time

    cached = _response_cache.get(key, time.monotonic())
    if cached is not None:
        return cached
    svc = _redis()
    if svc:
        return svc.get_json(f"rag:response:{key}")
    return None


def set_cached_response(key: str, payload: dict[str, Any]) -> None:
    if not RAG_RESPONSE_CACHE:
        return
    import time

    _response_cache.set(key, payload, time.monotonic())
    svc = _redis()
    if svc:
        svc.set_json(f"rag:response:{key}", payload, ttl_seconds=RAG_RESPONSE_CACHE_TTL)


def invalidate_document_caches(document_id: str) -> None:
    """Xóa cache retrieval/response liên quan tới tài liệu sau upload/xóa."""
    prefix = f"doc:{document_id}:"
    _retrieval_cache.delete_prefix(prefix)
    _response_cache.delete_prefix(prefix)
    svc = _redis()
    if svc and svc._client:
        try:
            for pattern in (f"rag:retrieval:*{document_id}*", f"rag:response:*{document_id}*"):
                for key in svc._client.scan_iter(match=pattern, count=50):
                    svc._client.delete(key)
        except Exception:
            pass


def document_cache_prefix(document_id: str) -> str:
    return f"doc:{document_id}:"


def adaptive_context_cache_key(query: str, chunks: list[dict[str, Any]]) -> str:
    chunk_ids = ",".join(sorted(
        str(c.get("chunk_id") or c.get("id", "")) for c in chunks
    ))
    return _stable_hash(["adaptive_ctx", query.strip(), chunk_ids])


def get_cached_adaptive_context(key: str) -> Any | None:
    if not RAG_ADAPTIVE_CONTEXT_CACHE:
        return None
    import time
    return _adaptive_context_cache.get(key, time.monotonic())


def set_cached_adaptive_context(key: str, context: Any) -> None:
    if not RAG_ADAPTIVE_CONTEXT_CACHE:
        return
    import time
    _adaptive_context_cache.set(key, context, time.monotonic())
