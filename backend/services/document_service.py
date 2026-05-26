"""Document Intelligence service wrapper with Postgres persistence and Redis caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from backend.core.settings import settings
from backend.db.repository import DocumentRepository
from src.document_intelligence import DocumentIntelligenceService
from src.utils import logger


class DocumentService:
    def __init__(self, store_dir: str | Path | None = None) -> None:
        self._inner = DocumentIntelligenceService(store_dir or settings.document_store_dir)
        self._repo = DocumentRepository()

    def ingest_file(self, path: str | Path, **kwargs: Any) -> dict[str, Any]:
        payload = self._inner.ingest_file(path, **kwargs)
        if self._repo.enabled:
            asyncio.run(self._persist_ingest(payload))
        return payload

    async def ingest_file_async(self, path: str | Path, **kwargs: Any) -> dict[str, Any]:
        payload = await asyncio.to_thread(self._inner.ingest_file, path, **kwargs)
        if self._repo.enabled:
            await self._persist_ingest(payload)
        return payload

    def get_document(self, document_id: str) -> dict[str, Any]:
        if self._repo.enabled:
            try:
                # Sync-over-async loop executor for safety
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Running in thread pool, safe to wait on future
                    future = asyncio.run_coroutine_threadsafe(self._repo.get_document(document_id), loop)
                    payload = future.result()
                else:
                    payload = loop.run_until_complete(self._repo.get_document(document_id))
                if payload:
                    return payload
            except Exception as exc:
                logger.warning("Synchronous repository fetch failed, fallback to file: %s", exc)
        return self._inner.get_document(document_id)

    async def get_document_async(self, document_id: str) -> dict[str, Any]:
        if self._repo.enabled:
            try:
                payload = await self._repo.get_document(document_id)
                if payload:
                    # Sync local file-cache for fast retrieval
                    cache_key = f"docintel:doc:{document_id}"
                    self._inner._cache_service().set_json(cache_key, payload, ttl_seconds=600)
                    return payload
            except Exception as exc:
                logger.warning("Repository get_document failed, fallback to file: %s", exc)
        return await asyncio.to_thread(self._inner.get_document, document_id)

    def list_documents(self, limit: int = 50) -> list[dict[str, Any]]:
        if self._repo.enabled:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(self._repo.list_documents(limit), loop)
                    return future.result()
                else:
                    return loop.run_until_complete(self._repo.list_documents(limit))
            except Exception as exc:
                logger.warning("Repository list failed: %s", exc)
        return self._inner.list_documents(limit=limit)

    async def list_documents_async(self, limit: int = 50) -> list[dict[str, Any]]:
        if self._repo.enabled:
            try:
                return await self._repo.list_documents(limit)
            except Exception as exc:
                logger.warning("Repository list failed: %s", exc)
        return await asyncio.to_thread(self._inner.list_documents, limit)

    def semantic_search(self, document_id: str, query: str, top_k: int = 5) -> dict[str, Any]:
        return self._inner.semantic_search(document_id, query, top_k)

    def compare_summaries(self, document_id: str, **kwargs: Any) -> dict[str, Any]:
        cache_key = self._compare_cache_key(document_id, **kwargs)
        cache_service = self._inner._cache_service()
        if cache_service.available:
            cached = cache_service.get_json(cache_key)
            if cached:
                logger.info(f"Redis cache hit (sync): {document_id}")
                return cached

        compare = self._inner.compare_summaries(document_id, **kwargs)
        if self._repo.enabled:
            asyncio.run(self._persist_compare(document_id, compare))

        if cache_service.available:
            cache_service.set_json(cache_key, compare, ttl_seconds=1800)
        return compare

    async def compare_summaries_async(self, document_id: str, **kwargs: Any) -> dict[str, Any]:
        cache_key = self._compare_cache_key(document_id, **kwargs)
        cache_service = self._inner._cache_service()
        if cache_service.available:
            try:
                cached = cache_service.get_json(cache_key)
                if cached:
                    logger.info(f"Redis cache hit (async): {document_id}")
                    return cached
            except Exception as exc:
                logger.warning("Redis fetch error: %s", exc)

        compare = await asyncio.to_thread(self._inner.compare_summaries, document_id, **kwargs)
        if self._repo.enabled:
            await self._persist_compare(document_id, compare)

        if cache_service.available:
            try:
                cache_service.set_json(cache_key, compare, ttl_seconds=1800)
            except Exception as exc:
                logger.warning("Redis store error: %s", exc)
        return compare

    def hierarchical_summarize(self, document_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._inner.hierarchical_summarize(document_id, **kwargs)

    def explain_extractive(self, document_id: str, algorithm: str = "textrank") -> dict[str, Any]:
        return self._inner.explain_extractive(document_id, algorithm)

    def generate_assets(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._inner.generate_assets(payload)

    def build_visualization(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._inner.build_visualization(payload)

    async def _persist_ingest(self, payload: dict[str, Any]) -> None:
        try:
            doc_id = payload.get("document_id", "")
            await self._repo.save_document(payload)
            await self._repo.save_chunks(doc_id, payload.get("chunks") or [])
            
            # Save learning assets directly to PostgreSQL tables
            assets = payload.get("analysis_assets") or {}
            if assets.get("quiz"):
                await self._repo.save_quiz(doc_id, assets["quiz"])
            if assets.get("podcast"):
                await self._repo.save_podcast_script(doc_id, assets["podcast"])
            if assets.get("reports"):
                await self._repo.save_report(doc_id, assets["reports"], "research")
            logger.info("Successfully persisted ingestion and assets to PostgreSQL")
        except Exception as exc:
            logger.error("Failed to persist ingestion to PostgreSQL: %s", exc, exc_info=True)

    async def _persist_compare(self, document_id: str, compare: dict[str, Any]) -> None:
        try:
            await self._repo.save_compare_results(document_id, compare)
            logger.info("Successfully persisted comparison results to PostgreSQL")
        except Exception as exc:
            logger.error("Failed to persist comparison results: %s", exc)

    def _compare_cache_key(self, document_id: str, **kwargs: Any) -> str:
        param_str = json.dumps(kwargs, sort_keys=True)
        param_hash = hashlib.sha1(param_str.encode("utf-8")).hexdigest()
        return f"docintel:compare:{document_id}:{param_hash}"
