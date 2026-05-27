from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator

from src import config

from .chunker import ChunkingPipeline
from .document_loader import DocumentLoader
from .embedding_service import EmbeddingService
from .generator import GroundedGenerator
from .repository import RAGRepository
from .retriever import HybridRetriever
from .vector_store import VectorStoreManager


class RAGChatService:
    def __init__(self) -> None:
        rag_dir = config.DOCUMENT_INTELLIGENCE_DIR / "rag"
        self.repository = RAGRepository(rag_dir / "rag_chat.db")
        self.loader = DocumentLoader()
        self.chunker = ChunkingPipeline()
        self.embedding_service = EmbeddingService()
        self.retriever = HybridRetriever()
        self.generator = GroundedGenerator()
        self.vector_store = VectorStoreManager(rag_dir)

    def list_embedding_models(self) -> dict:
        return self.embedding_service.list_models()

    def upload_document(
        self,
        *,
        path: Path,
        filename: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
    ) -> dict[str, Any]:
        loaded = self.loader.load(path)
        metadata = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
            "page_count": len(loaded["pages"]),
            "uploaded_at": int(time.time()),
        }
        document_id = self.repository.create_document(
            filename=filename,
            source_type=path.suffix.lower().lstrip("."),
            metadata=metadata,
        )
        chunks = self.chunker.split(
            text=loaded["text"],
            pages=loaded["pages"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            document_id=document_id,
            filename=filename,
        )
        vectors = self.embedding_service.embed_documents([c["text"] for c in chunks], embedding_model)
        self.repository.save_chunks(chunks, vectors, embedding_model)
        self.vector_store.upsert_chunks(chunks, vectors)
        return {
            "document_id": document_id,
            "filename": filename,
            "status": "ready",
            "chunk_count": len(chunks),
            "metadata": metadata,
        }

    def list_documents(self) -> list[dict[str, Any]]:
        return self.repository.list_documents()

    def list_conversations(self) -> list[dict[str, Any]]:
        return self.repository.list_conversations()

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return self.repository.list_messages(conversation_id)

    def delete_document(self, document_id: str) -> None:
        self.vector_store.delete_document(document_id)
        self.repository.delete_document(document_id)

    def chat(
        self,
        *,
        query: str,
        conversation_id: str | None,
        document_ids: list[str] | None,
        top_k: int,
        threshold: float,
        retrieval_mode: str,
        use_reranking: bool,
        embedding_model: str,
        temperature: float,
    ) -> dict[str, Any]:
        conv_id = self.repository.ensure_conversation(conversation_id, title=query[:60] or "New chat")
        self.repository.append_message(conv_id, "user", query)

        query_vector = self.embedding_service.embed_query(query, embedding_model)
        if retrieval_mode == "bm25":
            chunks = self.repository.list_chunks(document_ids=document_ids or None)
        else:
            chunks = self.vector_store.query(
                query_vector=query_vector,
                top_k=max(top_k * 8, 40),
                document_ids=document_ids or None,
            )
        retrieved = self.retriever.retrieve(
            query=query,
            query_vector=query_vector,
            chunks=chunks,
            top_k=top_k,
            threshold=threshold,
            retrieval_mode=retrieval_mode,
            use_reranking=use_reranking,
        )
        generation = self.generator.build_answer(query, retrieved, temperature=temperature)
        prompt = self.generator.prompt_template(retrieved, query)

        response = {
            "conversation_id": conv_id,
            "answer": generation["answer"],
            "confidence": generation["confidence"],
            "grounded": generation["grounded"],
            "retrieved_context": retrieved,
            "retrieval_threshold": threshold,
            "prompt_template": prompt,
        }
        self.repository.append_message(
            conv_id,
            "assistant",
            response["answer"],
            citations=retrieved,
            confidence=response["confidence"],
            retrieval_threshold=threshold,
        )
        return response

    async def stream_chat(self, **kwargs: Any) -> AsyncIterator[str]:
        response = await asyncio.to_thread(self.chat, **kwargs)
        answer = response["answer"]
        words = answer.split(" ")
        assembled = ""
        for word in words:
            assembled = f"{assembled} {word}".strip()
            payload = {
                "event": "token",
                "content": assembled,
                "conversation_id": response["conversation_id"],
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)
        yield f"data: {json.dumps({'event': 'done', 'response': response}, ensure_ascii=False)}\n\n"

