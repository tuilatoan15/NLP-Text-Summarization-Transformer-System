"""Optional Redis cache with in-memory LRU fallback for document analysis jobs."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any

from backend.core.settings import settings
from src.utils import logger


class InMemoryLRUCache:
    def __init__(self, maxsize: int = 500):
        self.maxsize = maxsize
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self.lock:
            if key in self.cache:
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                self.cache[key] = value
                if len(self.cache) > self.maxsize:
                    self.cache.popitem(last=False)

    def delete(self, key: str) -> None:
        with self.lock:
            self.cache.pop(key, None)


class CacheService:
    def __init__(self, redis_url: str | None = None) -> None:
        self.redis_url = redis_url or settings.redis_url
        self._client = None
        self._memory_cache = InMemoryLRUCache(maxsize=500)
        
        if self.redis_url:
            try:
                import redis

                self._client = redis.from_url(self.redis_url, decode_responses=True)
                self._client.ping()
                logger.info("Redis cache connected")
            except Exception as exc:
                logger.warning("Redis unavailable, falling back to thread-safe In-Memory LRU cache: %s", exc)
                self._client = None
        else:
            logger.info("No Redis URL configured, using thread-safe In-Memory LRU cache")

    @property
    def available(self) -> bool:
        # Cache is always available now, either through Redis or in-memory fallback
        return True

    def get_json(self, key: str) -> dict[str, Any] | None:
        if self._client:
            try:
                raw = self._client.get(key)
                return json.loads(raw) if raw else None
            except Exception as exc:
                logger.warning("Redis read error, falling back to memory cache: %s", exc)
                return self._memory_cache.get(key)
        return self._memory_cache.get(key)

    def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int = 3600) -> None:
        if self._client:
            try:
                self._client.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=False))
                return
            except Exception as exc:
                logger.warning("Redis write error, writing to memory cache: %s", exc)
        self._memory_cache.set(key, value)

    def delete(self, key: str) -> None:
        if self._client:
            try:
                self._client.delete(key)
                return
            except Exception as exc:
                logger.warning("Redis delete error, deleting from memory cache: %s", exc)
        self._memory_cache.delete(key)

