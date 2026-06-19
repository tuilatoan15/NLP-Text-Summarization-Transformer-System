from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RAGRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            
            # Documents and embeddings
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                """
            )
            
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    page INTEGER,
                    chunk_index INTEGER NOT NULL,
                    text_content TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES rag_documents(id) ON DELETE CASCADE
                );
                """
            )
            
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    vector_json TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(chunk_id) REFERENCES rag_chunks(id) ON DELETE CASCADE
                );
                """
            )

            # Updated conversations and messages tables
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    is_archived INTEGER DEFAULT 0,
                    user_id TEXT
                );
                """
            )
            
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL,
                    retrieval_threshold REAL,
                    citations_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    model_used TEXT,
                    evaluation_json TEXT,
                    metadata_json TEXT DEFAULT '{}',
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                """
            )
            
            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at);")

            # Check and migrate legacy SQLite tables: rag_conversations and rag_messages
            cursor = conn.cursor()
            
            # Disable foreign keys temporarily during migration to avoid DROP TABLE failures due to FK constraints
            conn.execute("PRAGMA foreign_keys = OFF;")
            
            # Check rag_conversations
            res = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rag_conversations'").fetchone()
            if res:
                logger.info("Migrating rag_conversations table to conversations...")
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at, message_count, is_archived)
                        SELECT id, title, created_at, updated_at, 0, 0 FROM rag_conversations;
                    """)
                    conn.execute("DROP TABLE rag_conversations;")
                except Exception as e:
                    logger.error(f"Failed to migrate rag_conversations: {e}")
                    
            # Check rag_messages
            res = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rag_messages'").fetchone()
            if res:
                logger.info("Migrating rag_messages table to messages...")
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO messages (id, conversation_id, role, content, confidence, retrieval_threshold, citations_json, created_at, model_used, evaluation_json)
                        SELECT id, conversation_id, role, content, confidence, retrieval_threshold, citations_json, created_at, model_used, evaluation_json FROM rag_messages;
                    """)
                    conn.execute("DROP TABLE rag_messages;")
                except Exception as e:
                    logger.error(f"Failed to migrate rag_messages: {e}")

            # Re-enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON;")

            # Ensure columns exist in conversations
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN message_count INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN is_archived INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
            except sqlite3.OperationalError:
                pass
                
            # Ensure columns exist in messages
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN metadata_json TEXT DEFAULT '{}'")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN evaluation_json TEXT")
            except sqlite3.OperationalError:
                pass

            # Recalculate message_count for conversations to be accurate
            try:
                conn.execute("""
                    UPDATE conversations
                    SET message_count = (
                        SELECT COUNT(*) FROM messages WHERE messages.conversation_id = conversations.id
                    )
                """)
            except Exception as e:
                logger.error(f"Failed to update message counts: {e}")

    def create_document(self, filename: str, source_type: str, metadata: dict[str, Any]) -> str:
        document_id = str(uuid.uuid4())
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rag_documents (id, filename, source_type, status, created_at, metadata_json)
                VALUES (?, ?, ?, 'ready', ?, ?)
                """,
                (document_id, filename, source_type, now, json.dumps(metadata, ensure_ascii=False)),
            )
        return document_id

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, filename, source_type, status, created_at, metadata_json FROM rag_documents ORDER BY created_at DESC"
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            items.append(
                {
                    "id": row["id"],
                    "filename": row["filename"],
                    "source_type": row["source_type"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                }
            )
        return items

    def delete_document(self, document_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM rag_embeddings WHERE chunk_id IN (SELECT id FROM rag_chunks WHERE document_id = ?)", (document_id,))
            conn.execute("DELETE FROM rag_chunks WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM rag_documents WHERE id = ?", (document_id,))

    def save_chunks(self, chunks: list[dict[str, Any]], vectors: list[list[float]], model_name: str) -> None:
        now = _now_iso()
        with self._connect() as conn:
            for chunk, vector in zip(chunks, vectors):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO rag_chunks
                    (id, document_id, filename, page, chunk_index, text_content, embedding_model, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk["id"],
                        chunk["document_id"],
                        chunk["filename"],
                        chunk.get("page"),
                        chunk["chunk_index"],
                        chunk["text"],
                        model_name,
                        json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                    ),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO rag_embeddings
                    (chunk_id, vector_json, dimension, model_name, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (chunk["id"], json.dumps(vector), len(vector), model_name, now),
                )

    def list_chunks(
        self, document_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT c.id, c.document_id, c.filename, c.page, c.chunk_index, c.text_content, c.embedding_model, "
            "c.metadata_json, e.vector_json FROM rag_chunks c JOIN rag_embeddings e ON c.id = e.chunk_id"
        )
        params: list[Any] = []
        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            query += f" WHERE c.document_id IN ({placeholders})"
            params.extend(document_ids)
        query += " ORDER BY c.filename, c.chunk_index"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        records: list[dict[str, Any]] = []
        for row in rows:
            records.append(
                {
                    "id": row["id"],
                    "document_id": row["document_id"],
                    "filename": row["filename"],
                    "page": row["page"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text_content"],
                    "embedding_model": row["embedding_model"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                    "vector": json.loads(row["vector_json"]),
                }
            )
        return records

    def ensure_conversation(self, conversation_id: str | None, title: str = "New chat") -> str:
        if conversation_id:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
            if row:
                return conversation_id

        new_id = str(uuid.uuid4())
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at, message_count, is_archived) VALUES (?, ?, ?, ?, 0, 0)",
                (new_id, title, now, now),
            )
        return new_id

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at, message_count, is_archived, user_id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_conversations(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at, message_count, is_archived, user_id FROM conversations ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_conversation(self, title: str = "New chat", user_id: str | None = None) -> dict[str, Any]:
        new_id = str(uuid.uuid4())
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at, message_count, is_archived, user_id) VALUES (?, ?, ?, ?, 0, 0, ?)",
                (new_id, title, now, now, user_id),
            )
        return {
            "id": new_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "is_archived": 0,
            "user_id": user_id
        }

    def rename_conversation(self, conversation_id: str, new_title: str) -> None:
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (new_title, now, conversation_id),
            )

    def delete_conversation(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def search_conversations(self, query: str, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        like_query = f"%{query}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT c.id, c.title, c.created_at, c.updated_at, c.message_count, c.is_archived, c.user_id
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.title LIKE ? OR m.content LIKE ?
                ORDER BY c.updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (like_query, like_query, limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        confidence: float | None = None,
        retrieval_threshold: float | None = None,
        model_used: str | None = None,
        evaluation: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        message_id = str(uuid.uuid4())
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages
                (id, conversation_id, role, content, confidence, retrieval_threshold, citations_json, created_at, model_used, evaluation_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    confidence,
                    retrieval_threshold,
                    json.dumps(citations or [], ensure_ascii=False),
                    now,
                    model_used,
                    json.dumps(evaluation or {}, ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?",
                (now, conversation_id),
            )
        return message_id

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, confidence, retrieval_threshold, citations_json, created_at, model_used, evaluation_json, metadata_json
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            record["citations"] = json.loads(record.pop("citations_json") or "[]")
            eval_str = record.pop("evaluation_json", None)
            record["evaluation"] = json.loads(eval_str or "{}") if eval_str else None
            meta_str = record.pop("metadata_json", None)
            record["metadata"] = json.loads(meta_str or "{}") if meta_str else {}
            records.append(record)
        return records

