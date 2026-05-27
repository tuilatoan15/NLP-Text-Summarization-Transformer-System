from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RAGRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    page INTEGER,
                    chunk_index INTEGER NOT NULL,
                    text_content TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES rag_documents(id)
                );

                CREATE TABLE IF NOT EXISTS rag_embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    vector_json TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(chunk_id) REFERENCES rag_chunks(id)
                );

                CREATE TABLE IF NOT EXISTS rag_conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rag_messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL,
                    retrieval_threshold REAL,
                    citations_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES rag_conversations(id)
                );
                """
            )

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
                    "SELECT id FROM rag_conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
            if row:
                return conversation_id

        new_id = str(uuid.uuid4())
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO rag_conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (new_id, title, now, now),
            )
        return new_id

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM rag_conversations ORDER BY updated_at DESC LIMIT ?",
                (limit,),
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
    ) -> str:
        message_id = str(uuid.uuid4())
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rag_messages
                (id, conversation_id, role, content, confidence, retrieval_threshold, citations_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
            conn.execute(
                "UPDATE rag_conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        return message_id

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, confidence, retrieval_threshold, citations_json, created_at
                FROM rag_messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            record["citations"] = json.loads(record.pop("citations_json") or "[]")
            records.append(record)
        return records

