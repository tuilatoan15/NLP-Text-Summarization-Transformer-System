"""Typed settings facade over src.config."""

from __future__ import annotations

from dataclasses import dataclass

from src import config


@dataclass(frozen=True)
class Settings:
    api_host: str = config.API_HOST
    api_port: int = config.API_PORT
    api_version: str = config.API_VERSION
    database_url: str = config.DATABASE_URL
    redis_url: str = config.REDIS_URL
    vector_backend: str = config.VECTOR_BACKEND
    document_store_dir: str = str(config.DOCUMENT_INTELLIGENCE_DIR)
    enable_db: bool = config.ENABLE_DB_PERSISTENCE
    default_embedding_model: str = config.DEFAULT_EMBEDDING_MODEL
    preload_models: bool = config.PRELOAD_MODELS


settings = Settings()
