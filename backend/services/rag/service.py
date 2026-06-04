"""
service.py — RAGChatService với cấu hình RAG hoàn toàn tự động.

Mọi tham số retrieval đều HARDCODE từ rag_config.py.
Người dùng không cần và không được cấu hình thủ công.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator

from src import config

from .chunker import ChunkingPipeline
from .document_loader import DocumentLoader
from .embedding_service import EmbeddingService
from .generator import GroundedGenerator
from .rag_config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    RETRIEVAL_FINAL_TOP_K,
    RETRIEVAL_INITIAL_TOP_K,
    RETRIEVAL_THRESHOLD,
)
from .repository import RAGRepository
from .retriever import HybridRetriever
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


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

    # ─────────────────────────────────────────────────────────────────────────
    # Upload — chunk_size và embedding_model được hardcode
    # ─────────────────────────────────────────────────────────────────────────

    def upload_document(
        self,
        *,
        path: Path,
        filename: str,
        # Tham số dưới đây giữ để tương thích API nhưng bị override bởi hardcode
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> dict[str, Any]:
        # Hardcode: luôn dùng cấu hình tối ưu
        chunk_size = CHUNK_SIZE
        chunk_overlap = CHUNK_OVERLAP
        embedding_model = EMBEDDING_MODEL

        logger.info(
            "📄 Upload: %s — chunk_size=%d, overlap=%d, embedding=%s",
            filename, chunk_size, chunk_overlap, embedding_model,
        )

        loaded = self.loader.load(path)
        metadata = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
            "page_count": len(loaded["pages"]),
            "uploaded_at": int(time.time()),
            "rag_config_version": "v2-hardcoded",
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
        vectors = self.embedding_service.embed_documents(
            [c["text"] for c in chunks], embedding_model
        )
        self.repository.save_chunks(chunks, vectors, embedding_model)
        self.vector_store.upsert_chunks(chunks, vectors)

        # RAPTOR-lite hierarchical indexing
        try:
            from .raptor import RaptorIndexer
            indexer = RaptorIndexer(
                repository=self.repository,
                vector_store=self.vector_store,
                embedding_service=self.embedding_service,
                generator=self.generator
            )
            indexer.build_tree(document_id, chunks, vectors, embedding_model)
        except Exception as exc:
            logger.error(f"❌ Failed to build RAPTOR tree for document {document_id}: {exc}", exc_info=True)

        logger.info("✅ Upload xong: %s — %d chunks", filename, len(chunks))
        return {
            "document_id": document_id,
            "filename": filename,
            "status": "ready",
            "chunk_count": len(chunks),
            "metadata": metadata,
            "rag_info": {
                "embedding_model": embedding_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "retrieval_mode": "hybrid",
                "reranking": True,
                "reranker": "BAAI/bge-reranker-v2-m3",
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # CRUD
    # ─────────────────────────────────────────────────────────────────────────

    def list_documents(self) -> list[dict[str, Any]]:
        return self.repository.list_documents()

    def list_conversations(self) -> list[dict[str, Any]]:
        return self.repository.list_conversations()

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return self.repository.list_messages(conversation_id)

    def delete_document(self, document_id: str) -> None:
        self.vector_store.delete_document(document_id)
        self.repository.delete_document(document_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Chat Q&A — hardcode retrieval params
    # ─────────────────────────────────────────────────────────────────────────

    def chat(
        self,
        *,
        query: str,
        conversation_id: str | None,
        document_ids: list[str] | None,
        # Các tham số dưới đây bị IGNORE — hardcode từ rag_config
        top_k: int = RETRIEVAL_FINAL_TOP_K,
        threshold: float = RETRIEVAL_THRESHOLD,
        retrieval_mode: str = "hybrid",
        use_reranking: bool = True,
        embedding_model: str = EMBEDDING_MODEL,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        # Hardcode: luôn dùng cấu hình tối ưu
        top_k = RETRIEVAL_FINAL_TOP_K
        threshold = RETRIEVAL_THRESHOLD
        retrieval_mode = "hybrid"
        use_reranking = True
        embedding_model = EMBEDDING_MODEL

        conv_id = self.repository.ensure_conversation(
            conversation_id, title=query[:60] or "New chat"
        )
        
        # Lấy lịch sử tin nhắn trước khi append tin nhắn hiện tại
        chat_history = self.repository.list_messages(conv_id)
        
        self.repository.append_message(conv_id, "user", query)

        # Classify intent
        from .agent import classify_intent, expand_query
        intent = classify_intent(query, document_ids)

        # Embed query
        query_vector = self.embedding_service.embed_query(query, embedding_model)

        # 1. GENERAL INTENT (No retrieval needed)
        if intent == "GENERAL":
            generation = self.generator.build_answer(
                query, [], chat_history=chat_history, temperature=temperature, general_chat=True
            )
            response = {
                "conversation_id": conv_id,
                "answer": generation["answer"],
                "confidence": 1.0,
                "grounded": False,
                "model_used": generation.get("model_used"),
                "fallback_used": generation.get("fallback_used", False),
                "retrieved_context": [],
                "retrieval_threshold": threshold,
                "prompt_template": f"General Chat: {query}",
                "intent": "general",
                "rag_config": {
                    "embedding_model": embedding_model,
                    "retrieval_mode": "none",
                    "top_k": 0,
                    "threshold": threshold,
                    "reranking": False,
                },
            }
            self.repository.append_message(
                conv_id,
                "assistant",
                response["answer"],
                citations=[],
                confidence=1.0,
                retrieval_threshold=threshold,
                model_used=response.get("model_used"),
            )
            return response

        # 2. DOCUMENT_QA INTENT (With Multi-query Expansion)
        elif intent == "DOCUMENT_QA":
            # Lấy candidates từ query gốc
            candidates = self.vector_store.query(
                query_vector=query_vector,
                top_k=RETRIEVAL_INITIAL_TOP_K,
                document_ids=document_ids or None,
            )
            # Query expansion
            expanded = expand_query(query)
            if expanded:
                seen_ids = {c["id"] for c in candidates}
                for eq in expanded:
                    eq_vector = self.embedding_service.embed_query(eq, embedding_model)
                    eq_candidates = self.vector_store.query(
                        query_vector=eq_vector,
                        top_k=5,
                        document_ids=document_ids or None,
                    )
                    for eq_c in eq_candidates:
                        if eq_c["id"] not in seen_ids:
                            candidates.append(eq_c)
                            seen_ids.add(eq_c["id"])
        
        # 3. SUMMARIZE INTENT (Hierarchical RAPTOR Retrieval)
        else: # intent == "SUMMARIZE"
            candidates = self.vector_store.query(
                query_vector=query_vector,
                top_k=RETRIEVAL_INITIAL_TOP_K,
                document_ids=document_ids or None,
            )
            summary_candidates = [c for c in candidates if c.get("metadata", {}).get("chunk_type") == "summary"]
            if summary_candidates:
                candidates = summary_candidates

        # Hybrid retrieval + Cross-Encoder reranking
        retrieved = self.retriever.retrieve(
            query=query,
            query_vector=query_vector,
            chunks=candidates,
            top_k=top_k,
            threshold=threshold,
            retrieval_mode=retrieval_mode,
            use_reranking=use_reranking,
        )

        # Sinh câu trả lời bằng Transformer
        generation = self.generator.build_answer(
            query, retrieved, chat_history=chat_history, temperature=temperature
        )
        prompt = self.generator.prompt_template(retrieved, query)

        # Đánh giá chất lượng câu trả lời (Fact-checking/Hallucination risk)
        eval_metrics = None
        if retrieved:
            try:
                from evaluation.hallucination import audit_summary
                source_text = "\n\n".join(c["text"] for c in retrieved)
                formatted_chunks = [{"chunk_id": c["id"], "text": c["text"]} for c in retrieved]
                audit_res = audit_summary(generation["answer"], source_text, chunks=formatted_chunks, mode="fast")
                eval_metrics = {
                    "consistency_score": float(audit_res.get("consistency_score", 0.0)),
                    "grounding_coverage": float(audit_res.get("grounding_coverage", 0.0)),
                    "semantic_coverage": float(audit_res.get("semantic_coverage", 0.0)),
                    "hallucination_risk": str(audit_res.get("hallucination_risk", "low")),
                }
            except Exception as exc:
                logger.error("Failed to run hallucination audit on chat response: %s", exc)

        response: dict[str, Any] = {
            "conversation_id": conv_id,
            "answer": generation["answer"],
            "confidence": generation["confidence"],
            "grounded": generation["grounded"],
            "model_used": generation.get("model_used"),
            "fallback_used": generation.get("fallback_used", False),
            "retrieved_context": retrieved,
            "retrieval_threshold": threshold,
            "prompt_template": prompt,
            "intent": intent.lower(),
            "evaluation": eval_metrics,
            "rag_config": {
                "embedding_model": embedding_model,
                "retrieval_mode": retrieval_mode,
                "vector_weight": 0.70,
                "bm25_weight": 0.30,
                "top_k": top_k,
                "threshold": threshold,
                "reranking": True,
            },
        }

        self.repository.append_message(
            conv_id,
            "assistant",
            response["answer"],
            citations=retrieved,
            confidence=response["confidence"],
            retrieval_threshold=threshold,
            model_used=response.get("model_used"),
            evaluation=response.get("evaluation"),
        )
        return response

    # ─────────────────────────────────────────────────────────────────────────
    # Document Summarization — TÓM TẮT TÀI LIỆU (tính năng mới)
    # ─────────────────────────────────────────────────────────────────────────

    def summarize_document(
        self,
        *,
        document_id: str,
        query: str = "Tóm tắt nội dung chính của tài liệu",
    ) -> dict[str, Any]:
        """
        Tóm tắt toàn bộ tài liệu bằng RAG pipeline:
          1. Retrieve các chunks quan trọng nhất bằng Hybrid + Reranker
          2. Sinh tóm tắt bằng Transformer (BARTPho/ViT5/mT5)

        Args:
            document_id: ID tài liệu cần tóm tắt
            query:       Query tóm tắt (mặc định là "Tóm tắt nội dung chính")

        Returns:
            dict với summary, model_used, word_count, retrieved_context
        """
        logger.info("📝 Bắt đầu tóm tắt tài liệu: %s", document_id)
        t_start = time.perf_counter()

        # Lấy embedding model từ metadata của document
        docs = self.repository.list_documents()
        embedding_model = EMBEDDING_MODEL
        for doc in docs:
            if doc["id"] == document_id:
                embedding_model = doc.get("metadata", {}).get("embedding_model", EMBEDDING_MODEL)
                break

        # Embed query tóm tắt
        query_vector = self.embedding_service.embed_query(query, embedding_model)

        # Lấy nhiều candidates
        candidates = self.vector_store.query(
            query_vector=query_vector,
            top_k=RETRIEVAL_INITIAL_TOP_K,
            document_ids=[document_id],
        )

        if not candidates:
            # Nếu vector store trống, lấy trực tiếp từ DB
            candidates = self.repository.list_chunks(document_ids=[document_id])

        # Hybrid retrieval + reranking để lấy chunks cốt lõi nhất
        retrieved = self.retriever.retrieve(
            query=query,
            query_vector=query_vector,
            chunks=candidates,
            top_k=RETRIEVAL_FINAL_TOP_K,
            threshold=max(0.2, RETRIEVAL_THRESHOLD - 0.1),  # nới lỏng 1 chút cho summarize
            retrieval_mode="hybrid",
            use_reranking=True,
        )

        if not retrieved and candidates:
            # Nếu vẫn trống, dùng top 5 candidates theo combined_score
            retrieved = sorted(candidates, key=lambda x: x.get("combined_score", 0), reverse=True)[:5]
            for i, item in enumerate(retrieved, 1):
                item["rank"] = i
                item.setdefault("rerank_score", item.get("combined_score", 0))

        # Sinh tóm tắt bằng Transformer
        summary_result = self.generator.build_document_summary(retrieved)
        elapsed = time.perf_counter() - t_start

        logger.info(
            "✅ Tóm tắt xong: %d từ, model=%s, %.2fs",
            summary_result["word_count"],
            summary_result.get("model_used"),
            elapsed,
        )

        return {
            "document_id": document_id,
            "summary": summary_result["summary"],
            "word_count": summary_result["word_count"],
            "model_used": summary_result.get("model_used"),
            "fallback_used": summary_result.get("fallback_used", False),
            "retrieved_context": retrieved,
            "processing_time_s": round(elapsed, 3),
            "rag_config": {
                "embedding_model": embedding_model,
                "retrieval_mode": "hybrid",
                "reranking": True,
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Streaming
    # ─────────────────────────────────────────────────────────────────────────

    async def stream_chat(self, **kwargs: Any) -> AsyncIterator[str]:
        """Stream chat response token by token."""
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

    async def stream_summarize(
        self,
        *,
        document_id: str,
        query: str = "Tóm tắt nội dung chính của tài liệu",
    ) -> AsyncIterator[str]:
        """Stream tóm tắt tài liệu từng token."""
        # Chạy summarize trong thread để không block event loop
        result = await asyncio.to_thread(
            self.summarize_document,
            document_id=document_id,
            query=query,
        )
        summary = result["summary"]

        # Yield metadata trước
        yield f"data: {json.dumps({'event': 'start', 'document_id': document_id, 'model_used': result.get('model_used')}, ensure_ascii=False)}\n\n"

        # Stream từng từ
        words = summary.split(" ")
        assembled = ""
        for word in words:
            assembled = f"{assembled} {word}".strip()
            payload = {
                "event": "token",
                "content": assembled,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.015)

        # Yield kết quả cuối cùng đầy đủ
        yield f"data: {json.dumps({'event': 'done', 'result': result}, ensure_ascii=False)}\n\n"

    def summarize_documents(
        self,
        *,
        document_ids: list[str],
        query: str = "Tóm tắt nội dung chính của các tài liệu",
    ) -> dict[str, Any]:
        """
        Tóm tắt đồng thời nhiều tài liệu hoặc tài liệu dài (Multi-document synthesis).
        1. Với 1 tài liệu: Gọi summarize_document trực tiếp.
        2. Với nhiều tài liệu:
           - Tóm tắt từng tài liệu trước (dùng RAPTOR tree summary hoặc trích xuất chunk cốt lõi).
           - Gom các bản tóm tắt đó lại và sinh ra một bản tóm tắt tổng hợp chéo (cross-document summary).
        """
        if not document_ids:
            return {
                "summary": "Không có tài liệu nào được chọn để tóm tắt.",
                "word_count": 0,
                "model_used": None,
                "fallback_used": True,
            }

        if len(document_ids) == 1:
            return self.summarize_document(document_id=document_ids[0], query=query)

        logger.info("📝 Bắt đầu tóm tắt đa tài liệu: %s", document_ids)
        t_start = time.perf_counter()

        individual_summaries = []
        model_used = None
        fallback_used = False

        for doc_id in document_ids:
            chunks = self.repository.list_chunks(document_ids=[doc_id])
            summaries = [c for c in chunks if c.get("metadata", {}).get("chunk_type") == "summary"]
            
            if summaries:
                doc_summary = "\n".join(s["text"] for s in summaries)
            else:
                res = self.summarize_document(document_id=doc_id, query=query)
                doc_summary = res["summary"]
                if res.get("fallback_used"):
                    fallback_used = True
                if res.get("model_used"):
                    model_used = res["model_used"]

            individual_summaries.append(f"--- Tóm tắt tài liệu (ID: {doc_id}): ---\n{doc_summary}")

        combined_context = "\n\n".join(individual_summaries)

        prompt = (
            "Bạn là chuyên gia phân tích thông tin cao cấp. Dưới đây là các bản tóm tắt riêng lẻ của nhiều tài liệu khác nhau.\n"
            "Hãy viết một bản tóm tắt tổng hợp chéo (cross-document synthesis) kết hợp và đối chiếu các thông tin cốt lõi, "
            "vạch ra những điểm tương đồng và khác biệt chính giữa chúng một cách khoa học, khách quan và mạch lạc:\n\n"
            f"{combined_context}\n\n"
            "Bản tóm tắt tổng hợp tiếng Việt chuẩn xác:"
        )

        from .rag_config import RAG_GENERATOR_TYPE
        summary = ""

        if RAG_GENERATOR_TYPE in {"gemini", "openai", "ollama"}:
            from .summarizer import _run_llm_api
            summary = _run_llm_api(prompt, RAG_GENERATOR_TYPE)
            if summary:
                model_used = f"{RAG_GENERATOR_TYPE}_api"

        if not summary:
            from .summarizer import _pick_available_model, _run_transformer_generate, GENERATION_PROFILES
            model_key = _pick_available_model()
            if model_key:
                profile = GENERATION_PROFILES[model_key]
                summary = _run_transformer_generate(model_key, prompt, profile)
                model_used = model_key

        if not summary:
            summary = "\n\n".join(individual_summaries)
            fallback_used = True
            model_used = "extractive_fallback"

        elapsed = time.perf_counter() - t_start
        word_count = len(summary.split())

        return {
            "document_ids": document_ids,
            "summary": summary,
            "word_count": word_count,
            "model_used": model_used,
            "fallback_used": fallback_used,
            "processing_time_s": round(elapsed, 3),
        }

    async def stream_summarize_documents(
        self,
        *,
        document_ids: list[str],
        query: str = "Tóm tắt nội dung chính của các tài liệu",
    ) -> AsyncIterator[str]:
        """Stream tóm tắt đa tài liệu từng token."""
        result = await asyncio.to_thread(
            self.summarize_documents,
            document_ids=document_ids,
            query=query,
        )
        summary = result["summary"]

        yield f"data: {json.dumps({'event': 'start', 'document_ids': document_ids, 'model_used': result.get('model_used')}, ensure_ascii=False)}\n\n"

        words = summary.split(" ")
        assembled = ""
        for word in words:
            assembled = f"{assembled} {word}".strip()
            payload = {
                "event": "token",
                "content": assembled,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.015)

        yield f"data: {json.dumps({'event': 'done', 'result': result}, ensure_ascii=False)}\n\n"
