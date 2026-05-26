"""Optional Redis cache for document analysis jobs."""

from __future__ import annotations

import json
from typing import Any

from backend.core.settings import settings
from src.utils import logger


class CacheService:
    def __init__(self, redis_url: str | None = None) -> None:
        self.redis_url = redis_url or settings.redis_url
        self._client = None
        if self.redis_url:
            try:
                import redis

                self._client = redis.from_url(self.redis_url, decode_responses=True)
                self._client.ping()
                logger.info("Redis cache connected")
            except Exception as exc:
                logger.warning("Redis unavailable, using in-memory noop cache: %s", exc)
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def get_json(self, key: str) -> dict[str, Any] | None:
        if not self._client:
            return None
        raw = self._client.get(key)
        return json.loads(raw) if raw else None

    def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int = 3600) -> None:
        if not self._client:
            return
        self._client.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=False))

    def delete(self, key: str) -> None:
        if self._client:
            self._client.delete(key)
