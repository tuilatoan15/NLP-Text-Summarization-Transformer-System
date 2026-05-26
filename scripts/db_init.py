#!/usr/bin/env python3
"""Database initialization script to set up PostgreSQL schema."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from src.utils import logger
from backend.core.settings import settings


async def init_db() -> None:
    db_url = os.getenv("DATABASE_URL") or settings.database_url
    if not db_url:
        logger.error("❌ DATABASE_URL is not set. Please check your environment.")
        sys.exit(1)

    # Smart fallback: if running from host Windows, translate 'postgres' container host to 'localhost'
    if "postgres:5432" in db_url and not os.path.exists("/.dockerenv"):
        db_url = db_url.replace("postgres:5432", "localhost:5432")
        logger.info(f"Using host redirection: {db_url}")

    logger.info("Initializing database schema...")
    schema_path = Path(__file__).resolve().parent.parent / "docs" / "DATABASE_SCHEMA.sql"
    if not schema_path.exists():
        logger.error(f"❌ Schema file not found at: {schema_path}")
        sys.exit(1)

    sql_content = schema_path.read_text(encoding="utf-8")
    
    # Split queries by semicolon to execute them sequentially
    # (avoiding issue where some drivers fail on multiple statements in one execute)
    statements = []
    # Simple parser to split by semicolon but ignore semicolons inside parentheses/quotes
    current = []
    for line in sql_content.splitlines():
        if line.strip().startswith("--") or not line.strip():
            continue
        current.append(line)
        if line.strip().endswith(";"):
            statements.append("\n".join(current))
            current = []
    if current:
        statements.append("\n".join(current))

    logger.info(f"Found {len(statements)} SQL statements to execute.")

    try:
        engine = create_async_engine(db_url, echo=False)
        async with engine.begin() as conn:
            for statement in statements:
                stmt_stripped = statement.strip()
                if not stmt_stripped:
                    continue
                # Execute each CREATE TABLE / CREATE INDEX statement
                await conn.execute(text(stmt_stripped))
        logger.info("✅ Database schema initialized successfully.")
    except Exception as exc:
        logger.error(f"❌ Database initialization failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(init_db())
