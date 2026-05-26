"""PostgreSQL persistence aligned with docs/DATABASE_SCHEMA.sql.

Important: This module must be import-safe even when SQLAlchemy is not installed.
When SQLAlchemy is missing, the repository becomes a disabled no-op.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, TYPE_CHECKING

from backend.core.settings import settings
from src.utils import logger


class DocumentRepository:
    """Async SQLAlchemy repository; no-op when disabled."""

    def __init__(self) -> None:
        self.enabled = bool(settings.enable_db and settings.database_url)
        self._engine = None
        if self.enabled:
            try:
                from sqlalchemy.ext.asyncio import create_async_engine

                db_url = settings.database_url
                # host redirection if running from host OS Windows instead of inside docker network
                import os
                if "postgres:5432" in db_url and not os.path.exists("/.dockerenv"):
                    db_url = db_url.replace("postgres:5432", "localhost:5432")

                self._engine = create_async_engine(db_url, echo=False)
            except Exception as exc:
                logger.warning("Database disabled (SQLAlchemy missing/unavailable): %s", exc)
                self.enabled = False

    async def save_document(self, payload: dict[str, Any]) -> None:
        if not self._engine:
            return
        from sqlalchemy import text
        doc_id = payload.get("document_id")
        metadata = payload.get("metadata", {})
        quality = payload.get("quality", {}).get("extraction", {})
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO uploaded_documents (
                        id, title, source_filename, source_type, page_count,
                        word_count, quality_score, metadata, structure
                    )
                    VALUES (
                        :id, :title, :filename, :stype, :pages,
                        :words, :score, CAST(:meta AS jsonb), CAST(:structure AS jsonb)
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        word_count = EXCLUDED.word_count,
                        quality_score = EXCLUDED.quality_score,
                        metadata = EXCLUDED.metadata,
                        structure = EXCLUDED.structure
                    """
                ),
                {
                    "id": doc_id,
                    "title": metadata.get("title"),
                    "filename": metadata.get("filename") or doc_id,
                    "stype": metadata.get("source_type", "txt"),
                    "pages": metadata.get("pages"),
                    "words": quality.get("word_count"),
                    "score": quality.get("score"),
                    "meta": json.dumps(metadata, ensure_ascii=False),
                    "structure": json.dumps(payload.get("structure", {}), ensure_ascii=False),
                },
            )

    async def save_chunks(self, document_id: str, chunks: list[dict[str, Any]]) -> None:
        if not self._engine or not chunks:
            return
        from sqlalchemy import text
        async with self._engine.begin() as conn:
            for idx, chunk in enumerate(chunks):
                chunk_id = chunk.get("chunk_id") or f"{document_id}-chunk-{idx}"
                await conn.execute(
                    text(
                        """
                        INSERT INTO chunks (
                            id, document_id, chunk_index, content, section_path,
                            page_start, page_end, token_count, word_count,
                            keywords, semantic_tags, parent_section, overlap_from_previous
                        )
                        VALUES (
                            :id, :doc, :idx, :content, CAST(:section AS jsonb),
                            :pstart, :pend, :tokens, :words,
                            CAST(:keywords AS jsonb), CAST(:tags AS jsonb),
                            :parent, :overlap
                        )
                        ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content
                        """
                    ),
                    {
                        "id": chunk_id,
                        "doc": document_id,
                        "idx": idx,
                        "content": chunk.get("text", ""),
                        "section": json.dumps(chunk.get("section_path", []), ensure_ascii=False),
                        "pstart": chunk.get("page_start"),
                        "pend": chunk.get("page_end"),
                        "tokens": chunk.get("token_count"),
                        "words": len((chunk.get("text") or "").split()),
                        "keywords": json.dumps(chunk.get("keywords", []), ensure_ascii=False),
                        "tags": json.dumps(chunk.get("semantic_tags", []), ensure_ascii=False),
                        "parent": (chunk.get("section_path") or [None])[0],
                        "overlap": bool(chunk.get("overlap_from_previous")),
                    },
                )

    async def save_compare_results(self, document_id: str, compare: dict[str, Any]) -> None:
        if not self._engine:
            return
        from sqlalchemy import text
        run_id = str(uuid.uuid4())
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO research_runs (id, document_id, run_type, config, results)
                    VALUES (
                        :id, :doc, 'compare',
                        CAST(:config AS jsonb), CAST(:results AS jsonb)
                    )
                    """
                ),
                {
                    "id": run_id,
                    "doc": document_id,
                    "config": json.dumps(
                        {
                            "algorithms": [r.get("key") for r in compare.get("results", [])],
                            "best_model": (compare.get("best_model") or {}).get("key"),
                        },
                        ensure_ascii=False,
                    ),
                    "results": json.dumps(
                        {
                            "research_matrix": compare.get("research_matrix"),
                            "summary_tree": compare.get("summary_tree"),
                        },
                        ensure_ascii=False,
                    ),
                },
            )

        for row in compare.get("results", []):
            await self.save_summary(
                document_id=document_id,
                algorithm_key=row.get("key", ""),
                algorithm_group=row.get("group", ""),
                summary_text=row.get("summary", ""),
                citations=row.get("citations", []),
                consistency=row.get("consistency", {}),
                metrics=row.get("metrics", {}),
            )

    async def save_summary(
        self,
        *,
        document_id: str,
        algorithm_key: str,
        algorithm_group: str,
        summary_text: str,
        citations: list[dict[str, Any]],
        consistency: dict[str, Any],
        metrics: dict[str, Any],
    ) -> str:
        if not self._engine:
            return ""
        from sqlalchemy import text
        summary_id = str(uuid.uuid4())
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO summaries (
                        id, document_id, algorithm_key, algorithm_group,
                        summary_text, word_count, citations, consistency
                    )
                    VALUES (
                        :id, :doc, :key, :grp, :text, :words,
                        CAST(:citations AS jsonb), CAST(:consistency AS jsonb)
                    )
                    """
                ),
                {
                    "id": summary_id,
                    "doc": document_id,
                    "key": algorithm_key,
                    "grp": algorithm_group,
                    "text": summary_text,
                    "words": len(summary_text.split()),
                    "citations": json.dumps(citations, ensure_ascii=False),
                    "consistency": json.dumps(consistency, ensure_ascii=False),
                },
            )
            await conn.execute(
                text(
                    """
                    INSERT INTO evaluations (
                        id, summary_id, document_id, rouge1, rouge2, rougel, bleu,
                        bertscore_f1, semantic_similarity, factual_consistency,
                        compression_ratio, latency_seconds, memory_usage_mb, metrics
                    )
                    VALUES (
                        :eid, :sid, :doc, :r1, :r2, :rl, :bleu, :bert, :sem, :fact,
                        :comp, :lat, :mem, CAST(:metrics AS jsonb)
                    )
                    """
                ),
                {
                    "eid": str(uuid.uuid4()),
                    "sid": summary_id,
                    "doc": document_id,
                    "r1": metrics.get("rouge1"),
                    "r2": metrics.get("rouge2"),
                    "rl": metrics.get("rougeL"),
                    "bleu": metrics.get("bleu"),
                    "bert": metrics.get("bertscore_f1"),
                    "sem": metrics.get("semantic_similarity"),
                    "fact": consistency.get("consistency_score"),
                    "comp": metrics.get("compression_ratio"),
                    "lat": metrics.get("processing_time"),
                    "mem": metrics.get("memory_usage_mb"),
                    "metrics": json.dumps(metrics, ensure_ascii=False),
                },
            )
        return summary_id

    async def save_podcast_script(self, document_id: str, script: dict[str, Any], audio_uri: str | None = None) -> str:
        if not self._engine:
            return ""
        from sqlalchemy import text
        podcast_id = str(uuid.uuid4())
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO podcasts (id, document_id, script, audio_uri, tts_provider)
                    VALUES (:id, :doc, CAST(:script AS jsonb), :audio, :provider)
                    """
                ),
                {
                    "id": podcast_id,
                    "doc": document_id,
                    "script": json.dumps(script, ensure_ascii=False),
                    "audio": audio_uri,
                    "provider": script.get("tts_provider") or "gTTS",
                },
            )
        return podcast_id

    # ── Database Retrieval Extensions ────────────────────────────────────────

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        if not self._engine:
            return None
        from sqlalchemy import text
        async with self._engine.connect() as conn:
            res_doc = await conn.execute(
                text("SELECT id, title, source_filename, source_type, page_count, word_count, quality_score, metadata, structure, created_at FROM uploaded_documents WHERE id = :id"),
                {"id": document_id}
            )
            doc_row = res_doc.fetchone()
            if not doc_row:
                return None

            # Retrieve chunks
            res_chunks = await conn.execute(
                text("SELECT id, chunk_index, content, section_path, page_start, page_end, token_count, keywords, semantic_tags, overlap_from_previous FROM chunks WHERE document_id = :id ORDER BY chunk_index ASC"),
                {"id": document_id}
            )
            chunks = []
            for r in res_chunks.fetchall():
                chunks.append({
                    "chunk_id": r[0],
                    "chunk_index": r[1],
                    "text": r[2],
                    "section_path": r[3] if isinstance(r[3], list) else json.loads(r[3] or "[]"),
                    "page_start": r[4],
                    "page_end": r[5],
                    "token_count": r[6],
                    "keywords": r[7] if isinstance(r[7], list) else json.loads(r[7] or "[]"),
                    "semantic_tags": r[8] if isinstance(r[8], list) else json.loads(r[8] or "[]"),
                    "overlap_from_previous": r[9]
                })

            metadata = doc_row[7] if isinstance(doc_row[7], dict) else json.loads(doc_row[7] or "{}")
            structure = doc_row[8] if isinstance(doc_row[8], dict) else json.loads(doc_row[8] or "{}")
            created_at_val = doc_row[9].isoformat() if doc_row[9] else ""

            # Fetch Quiz, Podcast, and Report if present
            quiz_data = await self.get_quiz(document_id) or []
            podcast_data = await self.get_podcast(document_id) or {}
            report_data = await self.get_report(document_id, "research") or {}

            # Construct standard payload representation
            payload = {
                "document_id": doc_row[0],
                "metadata": {
                    "title": doc_row[1] or metadata.get("title"),
                    "filename": doc_row[2] or metadata.get("filename"),
                    "source_type": doc_row[3] or metadata.get("source_type"),
                    "pages": doc_row[4] or metadata.get("pages"),
                    **metadata
                },
                "quality": {
                    "extraction": {
                        "word_count": doc_row[5],
                        "score": doc_row[6]
                    }
                },
                "structure": structure,
                "chunks": chunks,
                "created_at": created_at_val,
                "analysis_assets": {
                    "overview": {
                        "document_overview": metadata.get("overview") or "",
                        "key_insights": metadata.get("key_insights") or [],
                        "keywords": [k.get("term") if isinstance(k, dict) else k for k in (metadata.get("keywords") or [])[:12]],
                        "source_words": doc_row[5],
                        "chunk_count": len(chunks),
                    },
                    "quiz": quiz_data,
                    "podcast": podcast_data,
                    "reports": report_data
                }
            }
            return payload

    async def list_documents(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self._engine:
            return []
        from sqlalchemy import text
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT id, title, source_type, created_at, word_count, quality_score
                    FROM uploaded_documents
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit}
            )
            records = []
            for r in result.fetchall():
                records.append({
                    "document_id": r[0],
                    "title": r[1],
                    "source_type": r[2],
                    "created_at": r[3].isoformat() if r[3] else "",
                    "word_count": r[4],
                    "quality_score": r[5]
                })
            return records

    async def get_chunks(self, document_id: str) -> list[dict[str, Any]]:
        if not self._engine:
            return []
        from sqlalchemy import text
        async with self._engine.connect() as conn:
            res_chunks = await conn.execute(
                text("SELECT id, chunk_index, content, section_path, page_start, page_end, token_count, keywords, semantic_tags, overlap_from_previous FROM chunks WHERE document_id = :id ORDER BY chunk_index ASC"),
                {"id": document_id}
            )
            chunks = []
            for r in res_chunks.fetchall():
                chunks.append({
                    "chunk_id": r[0],
                    "chunk_index": r[1],
                    "text": r[2],
                    "section_path": r[3] if isinstance(r[3], list) else json.loads(r[3] or "[]"),
                    "page_start": r[4],
                    "page_end": r[5],
                    "token_count": r[6],
                    "keywords": r[7] if isinstance(r[7], list) else json.loads(r[7] or "[]"),
                    "semantic_tags": r[8] if isinstance(r[8], list) else json.loads(r[8] or "[]"),
                    "overlap_from_previous": r[9]
                })
            return chunks

    async def save_quiz(self, document_id: str, questions: list[dict[str, Any]], difficulty: str = "medium") -> None:
        if not self._engine:
            return
        from sqlalchemy import text
        quiz_id = str(uuid.uuid4())
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO quizzes (id, document_id, difficulty, questions)
                    VALUES (:id, :doc, :diff, CAST(:questions AS jsonb))
                    ON CONFLICT (id) DO UPDATE SET questions = EXCLUDED.questions
                    """
                ),
                {
                    "id": quiz_id,
                    "doc": document_id,
                    "diff": difficulty,
                    "questions": json.dumps(questions, ensure_ascii=False)
                }
            )

    async def get_quiz(self, document_id: str) -> list[dict[str, Any]] | None:
        if not self._engine:
            return None
        from sqlalchemy import text
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text("SELECT questions FROM quizzes WHERE document_id = :doc ORDER BY created_at DESC LIMIT 1"),
                {"doc": document_id}
            )
            row = res.fetchone()
            if row:
                return row[0] if isinstance(row[0], list) else json.loads(row[0] or "[]")
            return None

    async def get_podcast(self, document_id: str) -> dict[str, Any] | None:
        if not self._engine:
            return None
        from sqlalchemy import text
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text("SELECT script, audio_uri, tts_provider FROM podcasts WHERE document_id = :doc ORDER BY created_at DESC LIMIT 1"),
                {"doc": document_id}
            )
            row = res.fetchone()
            if row:
                script = row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")
                return {
                    **script,
                    "audio_uri": row[1],
                    "tts_provider": row[2]
                }
            return None

    async def save_report(self, document_id: str, report_data: dict[str, Any], report_type: str = "research") -> None:
        if not self._engine:
            return
        from sqlalchemy import text
        report_id = str(uuid.uuid4())
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO generated_reports (id, document_id, report_type, title, content)
                    VALUES (:id, :doc, :type, :title, CAST(:content AS jsonb))
                    ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content
                    """
                ),
                {
                    "id": report_id,
                    "doc": document_id,
                    "type": report_type,
                    "title": report_data.get("title", f"Báo cáo {report_type}"),
                    "content": json.dumps(report_data, ensure_ascii=False)
                }
            )

    async def get_report(self, document_id: str, report_type: str = "research") -> dict[str, Any] | None:
        if not self._engine:
            return None
        from sqlalchemy import text
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text("SELECT content FROM generated_reports WHERE document_id = :doc AND report_type = :type ORDER BY created_at DESC LIMIT 1"),
                {"doc": document_id, "type": report_type}
            )
            row = res.fetchone()
            if row:
                return row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")
            return None
