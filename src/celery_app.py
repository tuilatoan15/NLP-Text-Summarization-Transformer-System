"""
celery_app.py — Celery application configuration and initialization.
Handles asynchronous task processing for long-running summarization and embedding jobs.
"""
from __future__ import annotations

import os
import logging
from celery import Celery

# Set default logging level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve Redis URL from environment or configuration
# Default fallback to redis://localhost:6379/0
redis_url = os.getenv("REDIS_URL", "")
if not redis_url:
    # Try loading from local config if possible
    try:
        from src import config
        redis_url = config.REDIS_URL
    except Exception:
        pass

if not redis_url:
    redis_url = "redis://localhost:6379/0"

logger.info(f"🔌 Initializing Celery with Broker: {redis_url}")

# Create Celery instance
# We configure it to look for tasks in the 'workers.tasks' module
celery_app = Celery(
    "nlp_document_hub",
    broker=redis_url,
    backend=redis_url,
    include=["workers.tasks"]
)

# Celery Configurations
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,
    # Avoid pre-fetching too many heavy ML tasks
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    result_expires=3600,  # 1 hour
)

if __name__ == "__main__":
    celery_app.start()
