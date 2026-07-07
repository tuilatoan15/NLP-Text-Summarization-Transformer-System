"""
service.py — RAGChatService với cấu hình RAG hoàn toàn tự động.

Mọi tham số retrieval đều HARDCODE từ rag_config.py.
Người dùng không cần và không được cấu hình thủ công.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
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
    RAG_RESPONSE_CACHE,
    RAG_USE_LLM_QUERY_EXPANSION,
    RAG_USE_RAPTOR,
    RAG_RAPTOR_BACKGROUND,
    RAG_VERBOSE_LOG,
)
from .cache import (
    get_cached_response,
    set_cached_response,
    response_cache_key,
    get_cached_retrieval,
    set_cached_retrieval,
    retrieval_cache_key,
    invalidate_document_caches,
)
from .perf import StageTimer, compute_dynamic_top_k
from .context_compression import build_retrieved_context, CompressedContext
from .faithfulness import compute_chat_faithfulness, compute_retrieval_confidence
from .rag_config import RAG_ADAPTIVE_CONTEXT
from .repository import RAGRepository
from .retriever import HybridRetriever
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


def _log_latency(label: str, latency_details: dict[str, Any]) -> None:
    """Ghi latency — chỉ INFO khi RAG_VERBOSE_LOG=1 để giảm overhead hot path."""
    payload = json.dumps(latency_details, indent=2)
    if RAG_VERBOSE_LOG:
        logger.info("RAG Latency Details (%s):\n%s", label, payload)
    else:
        logger.debug("RAG Latency Details (%s):\n%s", label, payload)


def clean_generated_title(title: str) -> str:
    title = title.strip().replace('"', '').replace("'", "")
    # Keep letters, numbers, spaces, and Vietnamese diacritics
    title = re.sub(
        r"[^\w\s\dàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"
        r"ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸÝ]",
        "",
        title
    )
    title = " ".join(title.split())  # clean extra spaces
    return title


def generate_title_for_conversation(user_query: str) -> str:
    # Build prompt
    prompt = (
        "Bạn là trợ lý ảo AI. Hãy đặt một tiêu đề ngắn gọn (từ 5 đến 10 từ) cho cuộc trò chuyện bắt đầu bằng câu hỏi sau đây.\n"
        "Yêu cầu: Không chứa ký tự đặc biệt, không vượt quá 80 ký tự, tập trung vào chủ đề chính. Chỉ trả về tiêu đề duy nhất, không thêm lời dẫn giải.\n\n"
        f"Câu hỏi: \"{user_query}\"\n\n"
        "Tiêu đề tiếng Việt:"
    )
    
    title = ""
    try:
        from .rag_config import RAG_GENERATOR_TYPE
        if RAG_GENERATOR_TYPE in {"gemini", "openai", "ollama"}:
            from .summarizer import _run_llm_api
            title = _run_llm_api(prompt, RAG_GENERATOR_TYPE)
        
        if not title:
            from .summarizer import _pick_available_model, _run_transformer_generate
            from .rag_config import resolve_generation_profile
            model_key = _pick_available_model()
            if model_key:
                profile = resolve_generation_profile(model_key)
                title = _run_transformer_generate(model_key, prompt, profile)
    except Exception as e:
        logger.error(f"Error in auto title generation: {e}")
        
    cleaned_title = clean_generated_title(title)
    word_count = len(cleaned_title.split())
    if cleaned_title and 3 <= word_count <= 15 and len(cleaned_title) <= 80:
        return cleaned_title
    else:
        # Fallback: Lấy 50 ký tự đầu tiên
        fallback = user_query.strip()
        fallback = re.sub(r"\s+", " ", fallback)
        if len(fallback) > 50:
            return fallback[:50] + "..."
        return fallback[:50]


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

    def _resolve_embedding_model(self, document_ids: list[str] | None) -> str:
        """Pick embedding model from the first selected document's metadata."""
        if not document_ids:
            return EMBEDDING_MODEL
        model = self.repository.get_document_embedding_model(document_ids[0])
        return model or EMBEDDING_MODEL

    def _resolve_selected_filenames(self, document_ids: list[str] | None) -> list[str] | None:
        """Map document_ids đã chọn sang tên file (cho multi-doc prompt)."""
        if not document_ids or len(document_ids) <= 1:
            return None
        doc_map = {d["id"]: d["filename"] for d in self.repository.list_documents()}
        return [doc_map.get(doc_id, doc_id) for doc_id in document_ids]

    def _warn_missing_doc_coverage(
        self,
        document_ids: list[str] | None,
        retrieved: list[dict[str, Any]],
    ) -> None:
        if not document_ids or len(document_ids) <= 1:
            return
        covered = {str(c.get("document_id", "")) for c in retrieved}
        missing = [doc_id for doc_id in document_ids if doc_id not in covered]
        if missing:
            logger.warning(
                "⚠️ Multi-doc: %d tài liệu đã chọn không có chunk sau backfill: %s",
                len(missing),
                missing,
            )

    def _gather_retrieval_candidates(
        self,
        *,
        query_vector: list[float],
        document_ids: list[str] | None,
        intent: str,
        embedding_model: str,
        current_query: str,
        retries: int,
        use_expansion: bool,
    ) -> list[dict[str, Any]]:
        """Thu thập candidates — per-doc khi chọn nhiều tài liệu."""
        if intent == "DOCUMENT_QA":
            if document_ids and len(document_ids) > 1:
                per_doc_k = max(5, RETRIEVAL_INITIAL_TOP_K // len(document_ids))
                candidates: list[dict[str, Any]] = []
                seen_ids: set[str] = set()
                for doc_id in document_ids:
                    doc_hits = self.vector_store.query(
                        query_vector=query_vector,
                        top_k=per_doc_k,
                        document_ids=[doc_id],
                    )
                    for hit in doc_hits:
                        if hit["id"] not in seen_ids:
                            candidates.append(hit)
                            seen_ids.add(hit["id"])
            else:
                candidates = self.vector_store.query(
                    query_vector=query_vector,
                    top_k=RETRIEVAL_INITIAL_TOP_K,
                    document_ids=document_ids or None,
                )

            if retries == 0 and use_expansion:
                from .agent import expand_query

                expanded = expand_query(current_query)
                if expanded:
                    seen_ids = {c["id"] for c in candidates}
                    eq_vectors = self.embedding_service.embed_queries_batch(
                        expanded, embedding_model
                    )
                    for eq, eq_vector in zip(expanded, eq_vectors):
                        if document_ids and len(document_ids) > 1:
                            for doc_id in document_ids:
                                eq_candidates = self.vector_store.query(
                                    query_vector=eq_vector,
                                    top_k=3,
                                    document_ids=[doc_id],
                                )
                                for eq_c in eq_candidates:
                                    if eq_c["id"] not in seen_ids:
                                        candidates.append(eq_c)
                                        seen_ids.add(eq_c["id"])
                        else:
                            eq_candidates = self.vector_store.query(
                                query_vector=eq_vector,
                                top_k=5,
                                document_ids=document_ids or None,
                            )
                            for eq_c in eq_candidates:
                                if eq_c["id"] not in seen_ids:
                                    candidates.append(eq_c)
                                    seen_ids.add(eq_c["id"])
            return candidates

        if not document_ids:
            return self.vector_store.query(
                query_vector=query_vector,
                top_k=RETRIEVAL_INITIAL_TOP_K,
                document_ids=None,
            )

        if len(document_ids) > 1:
            per_doc_k = max(5, RETRIEVAL_INITIAL_TOP_K // len(document_ids))
            merged: list[dict[str, Any]] = []
            seen: set[str] = set()
            for doc_id in document_ids:
                for hit in self.vector_store.query(
                    query_vector=query_vector,
                    top_k=per_doc_k,
                    document_ids=[doc_id],
                ):
                    if hit["id"] not in seen:
                        merged.append(hit)
                        seen.add(hit["id"])
            candidates = merged
        else:
            candidates = self.vector_store.query(
                query_vector=query_vector,
                top_k=RETRIEVAL_INITIAL_TOP_K,
                document_ids=document_ids,
            )

        summary_candidates = [
            c for c in candidates if c.get("metadata", {}).get("chunk_type") == "summary"
        ]
        if summary_candidates:
            return summary_candidates
        return candidates

    def _compute_answer_metrics(
        self,
        *,
        answer: str,
        retrieved: list[dict[str, Any]],
        compressed: CompressedContext | None,
    ) -> tuple[float, float, dict[str, Any] | None]:
        retrieval_confidence = compute_retrieval_confidence(retrieved)
        eval_metrics = None
        if retrieved and answer:
            source_text = (
                compressed.effective_context_text()
                if compressed and compressed.enabled
                else "\n\n".join(c["text"] for c in retrieved)
            )
            eval_metrics = compute_chat_faithfulness(answer, source_text, retrieved)
        faithfulness = float(eval_metrics.get("faithfulness", 0.0)) if eval_metrics else 0.0
        return faithfulness, retrieval_confidence, eval_metrics

    def _trigger_auto_title(self, conv_id: str) -> None:
        def run():
            try:
                messages = self.repository.list_messages(conv_id)
                if 2 <= len(messages) <= 4:
                    first_user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                    if first_user_msg:
                        new_title = generate_title_for_conversation(first_user_msg)
                        self.repository.rename_conversation(conv_id, new_title)
                        logger.info(f"Auto title generated asynchronously for conversation {conv_id}: {new_title}")
            except Exception as e:
                logger.error(f"Failed to auto-update conversation title asynchronously: {e}")

        import threading
        threading.Thread(target=run, daemon=True).start()

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
            "chunking_mode": "dynamic_semantic"
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
            embedding_service=self.embedding_service,
            embedding_model=embedding_model,
            chunking_mode="dynamic"
        )
        vectors = self.embedding_service.embed_documents(
            [c["text"] for c in chunks], embedding_model
        )
        self.repository.save_chunks(chunks, vectors, embedding_model)
        self.vector_store.upsert_chunks(chunks, vectors)

        raptor_task_id: str | None = None
        raptor_status = "skipped"
        if RAG_USE_RAPTOR:
            if RAG_RAPTOR_BACKGROUND:
                raptor_task_id = self._schedule_raptor_build(document_id, embedding_model)
                if raptor_task_id:
                    raptor_status = "pending"
                else:
                    self._build_raptor_sync(document_id, chunks, vectors, embedding_model)
                    raptor_status = "ready"
            else:
                self._build_raptor_sync(document_id, chunks, vectors, embedding_model)
                raptor_status = "ready"
        else:
            logger.info("🌲 Bỏ qua dựng cây RAPTOR theo cấu hình RAG_USE_RAPTOR=0 để nạp file nhanh.")

        logger.info("✅ Upload xong: %s — %d chunks (raptor=%s)", filename, len(chunks), raptor_status)
        invalidate_document_caches(document_id)
        return {
            "document_id": document_id,
            "filename": filename,
            "status": "ready",
            "chunk_count": len(chunks),
            "raptor_status": raptor_status,
            "raptor_task_id": raptor_task_id,
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

    def _build_raptor_sync(
        self,
        document_id: str,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
        embedding_model: str,
    ) -> None:
        try:
            from .raptor import RaptorIndexer
            indexer = RaptorIndexer(
                repository=self.repository,
                vector_store=self.vector_store,
                embedding_service=self.embedding_service,
                generator=self.generator,
            )
            indexer.build_tree(document_id, chunks, vectors, embedding_model, max_levels=3)
        except Exception as exc:
            logger.error(
                "❌ Failed to build RAPTOR tree for document %s: %s",
                document_id, exc, exc_info=True,
            )

    def _schedule_raptor_build(self, document_id: str, embedding_model: str) -> str | None:
        """Đẩy RAPTOR sang Celery — trả task_id hoặc None nếu fallback sync."""
        try:
            from workers.tasks import build_raptor_task
            result = build_raptor_task.delay(document_id, embedding_model)
            logger.info("🌲 RAPTOR background task queued: %s (doc=%s)", result.id, document_id)
            return result.id
        except Exception as exc:
            logger.warning("Celery RAPTOR unavailable (%s) — fallback sync build", exc)
            return None

    def build_raptor_from_db(self, document_id: str, embedding_model: str) -> None:
        """Tải base chunks từ DB và dựng cây RAPTOR (dùng cho Celery worker)."""
        all_chunks = self.repository.list_chunks([document_id])
        base_chunks = [
            c for c in all_chunks
            if c.get("chunk_index", 0) >= 0
            and c.get("metadata", {}).get("chunk_type") != "summary"
        ]
        if not base_chunks:
            logger.info("Không có base chunks cho RAPTOR doc=%s", document_id)
            return
        vectors = [c["vector"] for c in base_chunks]
        chunk_payload = [
            {k: c[k] for k in ("id", "document_id", "filename", "page", "chunk_index", "text", "metadata")}
            for c in base_chunks
        ]
        self._build_raptor_sync(document_id, chunk_payload, vectors, embedding_model)
        invalidate_document_caches(document_id)

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
        invalidate_document_caches(document_id)

    def delete_all_documents(self) -> None:
        """Xóa toàn bộ RAG documents, chunks, embeddings và vectors."""
        self.vector_store.delete_all_documents()
        self.repository.delete_all_documents()

    def delete_all_conversations(self) -> None:
        """Xóa toàn bộ cuộc trò chuyện và messages."""
        self.repository.delete_all_conversations()

    # ─────────────────────────────────────────────────────────────────────────
    # Chat Q&A — hardcode retrieval params
    # ─────────────────────────────────────────────────────────────────────────

    def chat(
        self,
        *,
        query: str,
        conversation_id: str | None,
        document_ids: list[str] | None,
        # Các tham số dưới đây được hỗ trợ cấu hình động truyền vào
        top_k: int = RETRIEVAL_FINAL_TOP_K,
        threshold: float = RETRIEVAL_THRESHOLD,
        retrieval_mode: str = "hybrid",
        use_reranking: bool = True,
        embedding_model: str = EMBEDDING_MODEL,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        t_total_start = time.perf_counter()
        timer = StageTimer()

        # Resolve embedding model from document metadata (single query, not full list scan)
        embedding_model = self._resolve_embedding_model(document_ids)

        conv_id = self.repository.ensure_conversation(
            conversation_id, title=query[:60] or "New chat"
        )

        # Lấy lịch sử tin nhắn trước khi append tin nhắn hiện tại
        chat_history = self.repository.list_messages(conv_id)

        cache_key = response_cache_key(
            query, document_ids, conv_id, embedding_model,
            retrieval_mode, use_reranking, top_k, threshold, temperature,
        )
        if RAG_RESPONSE_CACHE and len(chat_history) <= 1:
            cached = get_cached_response(cache_key)
            if cached:
                cached["conversation_id"] = conv_id
                self.repository.append_message(conv_id, "user", query)
                self.repository.append_message(
                    conv_id, "assistant", cached["answer"],
                    citations=cached.get("retrieved_context", []),
                    confidence=cached.get("confidence"),
                    retrieval_threshold=threshold,
                    model_used=cached.get("model_used"),
                    evaluation=cached.get("evaluation"),
                )
                cached["latency_details"] = cached.get("latency_details", {})
                cached["latency_details"]["cache_hit"] = True
                logger.info("⚡ RAG response cache hit cho query: %s", query[:60])
                return cached

        self.repository.append_message(conv_id, "user", query)

        # Classify intent
        from .agent import classify_intent
        timer.start("intent")
        intent = classify_intent(query, document_ids)
        if intent == "SUMMARIZE" and not document_ids:
            intent = "DOCUMENT_QA"
        timer.stop("intent")

        top_k = compute_dynamic_top_k(
            base_top_k=top_k,
            query=query,
            document_ids=document_ids,
            intent=intent,
        )

        t_embed_start = time.perf_counter()

        # 1. GENERAL INTENT (No retrieval needed)
        if intent == "GENERAL":
            embedding_time = 0.0
            t_gen_start = time.perf_counter()
            generation = self.generator.build_answer(
                query, [], chat_history=chat_history, temperature=temperature, general_chat=True
            )
            generation_time = time.perf_counter() - t_gen_start
            total_time = time.perf_counter() - t_total_start

            latency_details = {
                "embedding": f"{embedding_time:.4f}s",
                "retrieval": "0.0000s",
                "reranking": "0.0000s",
                "generation": f"{generation_time:.4f}s",
                "total": f"{total_time:.4f}s",
                "stage_breakdown": timer.report(),
            }
            _log_latency("GENERAL", latency_details)

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
                "latency_details": latency_details,
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
            self._trigger_auto_title(conv_id)
            return response

        # 2. DOCUMENT_QA / SUMMARIZE INTENT (Vòng lặp Agentic RAG tự sửa lỗi)
        import os
        current_query = query
        retries = 0
        max_retries = int(os.getenv("RAG_AGENT_MAX_RETRIES", "2"))
        current_threshold = threshold
        current_top_k = top_k
        if document_ids and len(document_ids) > 1:
            current_threshold = min(threshold, 0.25)
        selected_filenames = self._resolve_selected_filenames(document_ids)

        ret_key = retrieval_cache_key(
            current_query, document_ids, embedding_model,
            retrieval_mode, use_reranking, current_top_k, current_threshold,
        )

        t_retrieval_start = time.perf_counter()
        timer.start("retrieval")

        while True:
            # Embed truy vấn hiện tại (cached for identical queries)
            timer.start("embedding")
            t_embed_loop = time.perf_counter()
            query_vector = self.embedding_service.embed_query(current_query, embedding_model)
            loop_embedding_time = time.perf_counter() - t_embed_loop
            timer.stop("embedding")
            if retries == 0:
                embedding_time = loop_embedding_time

            cached_retrieved = get_cached_retrieval(ret_key) if retries == 0 else None
            if cached_retrieved is not None:
                retrieved = cached_retrieved
                logger.debug("⚡ Retrieval cache hit")
            else:
                candidates = self._gather_retrieval_candidates(
                    query_vector=query_vector,
                    document_ids=document_ids,
                    intent=intent,
                    embedding_model=embedding_model,
                    current_query=current_query,
                    retries=retries,
                    use_expansion=RAG_USE_LLM_QUERY_EXPANSION,
                )

                if not candidates:
                    candidates = self.repository.list_chunks(document_ids=document_ids)
                    candidates = [
                        c for c in candidates
                        if c.get("embedding_model") == embedding_model
                    ]

                timer.start("reranking")
                retrieved = self.retriever.retrieve(
                    query=current_query,
                    query_vector=query_vector,
                    chunks=candidates,
                    top_k=current_top_k,
                    threshold=current_threshold,
                    retrieval_mode=retrieval_mode,
                    use_reranking=use_reranking,
                    document_ids=document_ids,
                )
                timer.stop("reranking")
                if retries == 0:
                    set_cached_retrieval(ret_key, retrieved)
                self._warn_missing_doc_coverage(document_ids, retrieved)

            # Sinh câu trả lời bằng Generator (sau Hybrid Context Compression)
            timer.start("context_compression")
            compressed = build_retrieved_context(
                retrieved, current_query, timer, document_ids=document_ids,
            )
            timer.stop("context_compression")

            timer.start("prompt_build")
            prompt = self.generator.compose_prompt(
                current_query,
                retrieved,
                chat_history=chat_history,
                compressed_context=compressed if compressed.enabled else None,
                selected_document_ids=document_ids,
                selected_filenames=selected_filenames,
            )
            timer.stop("prompt_build")

            timer.start("generation")
            t_gen_loop = time.perf_counter()
            generation = self.generator.build_answer(
                current_query,
                retrieved,
                chat_history=chat_history,
                temperature=temperature,
                compressed_context=compressed if compressed.enabled else None,
                selected_document_ids=document_ids,
                selected_filenames=selected_filenames,
            )
            generation_time = time.perf_counter() - t_gen_loop
            timer.stop("generation")

            top_rerank = None
            if retrieved:
                top_rerank = retrieved[0].get("rerank_score") or retrieved[0].get("combined_score")

            # Gọi Judge — early exit khi rerank score cao
            if not retrieved:
                judge_result = {
                    "faithfulness": "yes",
                    "relevance": "no",
                    "sufficiency": "no",
                    "feedback": "Không tìm thấy bất kỳ phân đoạn tài liệu phù hợp nào."
                }
            else:
                judge_context = (
                    compressed.effective_context_text()
                    if compressed.enabled
                    else "\n\n".join(c["text"] for c in retrieved)
                )
                from .agent import evaluate_answer
                judge_result = evaluate_answer(
                    query, judge_context, generation["answer"],
                    top_rerank_score=top_rerank,
                    retrieved_count=len(retrieved),
                )

            # Kiểm tra xem có cần thử lại hay không
            if judge_result["sufficiency"] == "no" and retries < max_retries:
                retries += 1
                logger.info(f"🔄 Agentic RAG [Lượt {retries}]: Phản hồi Judge báo thiếu thông tin: {judge_result['feedback']}")
                
                # Viết lại câu hỏi tập trung vào thông tin bị thiếu
                from .agent import rewrite_query
                current_query = rewrite_query(query, judge_result["feedback"])
                
                # Nới lỏng ngưỡng tương đồng và tăng số lượng chunk
                current_threshold = max(0.1, current_threshold - 0.08)
                current_top_k = current_top_k + 2
                continue
            else:
                # Đạt yêu cầu hoặc đạt giới hạn số lần retry -> Kết thúc
                break

        retrieval_time = time.perf_counter() - t_retrieval_start
        timer.stop("retrieval")

        # Lấy chi tiết thời gian từ retriever
        bm25_time = self.retriever.last_latency.get("bm25", 0.0)
        fusion_time = self.retriever.last_latency.get("vector_rrf", 0.0)
        rerank_time = self.retriever.last_latency.get("rerank", 0.0)
        actual_retrieval_only = max(0.0, retrieval_time - rerank_time)
        if "compressed" not in locals():
            compressed = build_retrieved_context(
                retrieved, current_query, timer, document_ids=document_ids,
            )
        prompt = self.generator.compose_prompt(
            current_query,
            retrieved,
            chat_history=chat_history,
            compressed_context=compressed if compressed.enabled else None,
            selected_document_ids=document_ids,
            selected_filenames=selected_filenames,
        )

        faithfulness, retrieval_confidence, eval_metrics = self._compute_answer_metrics(
            answer=generation["answer"],
            retrieved=retrieved,
            compressed=compressed if compressed.enabled else None,
        )

        total_time = time.perf_counter() - t_total_start

        latency_details = {
            "embedding": f"{embedding_time:.4f}s",
            "retrieval": f"{actual_retrieval_only:.4f}s",
            "reranking": f"{rerank_time:.4f}s",
            "hybrid_summary": f"{compressed.latency_s:.4f}s" if compressed.enabled else "0.0000s",
            "context_compression": f"{compressed.latency_s:.4f}s",
            "generation": f"{generation_time:.4f}s",
            "total": f"{total_time:.4f}s",
            "stage_breakdown": timer.report(),
            "compression_ratio": compressed.compression_ratio,
            "compression_enabled": compressed.enabled,
            "adaptive_mode": compressed.mode == "adaptive",
            "token_reduction": compressed.context_details.get("token_reduction"),
            "chunks_kept": compressed.dynamic_chunks_kept or len(compressed.top_original_chunks),
            "summary_tokens": compressed.summary_tokens,
            "latency_saving_estimate_s": compressed.latency_saving_estimate,
        }
        _log_latency("QA", latency_details)

        response: dict[str, Any] = {
            "conversation_id": conv_id,
            "answer": generation["answer"],
            "confidence": retrieval_confidence,
            "retrieval_confidence": retrieval_confidence,
            "faithfulness": faithfulness,
            "grounded": generation["grounded"],
            "model_used": generation.get("model_used"),
            "fallback_used": generation.get("fallback_used", False),
            "retrieved_context": retrieved,
            "retrieval_threshold": threshold,
            "prompt_template": prompt,
            "intent": intent.lower(),
            "evaluation": eval_metrics,
            "latency_details": latency_details,
            "context_compression": {
                "enabled": compressed.enabled,
                "skipped_reason": compressed.skipped_reason,
                "hybrid_summary": compressed.hybrid_summary if compressed.enabled else None,
                "top_original_count": len(compressed.top_original_chunks),
                "compression_ratio": compressed.compression_ratio,
                "hybrid_algo": compressed.hybrid_algo_key,
                "summary_model": compressed.model_used,
                "mode": compressed.mode,
                "compression_tier": compressed.compression_tier,
                "query_intent": compressed.query_intent,
                "query_focus": compressed.query_focus,
                "facts_preserved": compressed.facts_preserved_count,
                "citations_count": len(compressed.citations),
            },
            "context_details": compressed.context_details if compressed.mode == "adaptive" else None,
            "rag_config": {
                "embedding_model": embedding_model,
                "retrieval_mode": retrieval_mode,
                "vector_weight": 0.70,
                "bm25_weight": 0.30,
                "top_k": top_k,
                "threshold": threshold,
                "reranking": use_reranking,
                "agent_retries": retries,
                "judge_feedback": judge_result.get("feedback", "")
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
        self._trigger_auto_title(conv_id)

        if RAG_RESPONSE_CACHE and len(chat_history) <= 1:
            cache_payload = {k: v for k, v in response.items() if k != "conversation_id"}
            set_cached_response(cache_key, cache_payload)

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
        embedding_model = self._resolve_embedding_model([document_id])

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

    def _prepare_summarize_document(
        self,
        *,
        document_id: str,
        query: str = "Tóm tắt nội dung chính của tài liệu",
    ) -> dict[str, Any]:
        """Chỉ retrieval — dùng cho streaming tóm tắt (tránh generate 2 lần)."""
        embedding_model = self._resolve_embedding_model([document_id])
        query_vector = self.embedding_service.embed_query(query, embedding_model)
        candidates = self.vector_store.query(
            query_vector=query_vector,
            top_k=RETRIEVAL_INITIAL_TOP_K,
            document_ids=[document_id],
        )
        if not candidates:
            candidates = self.repository.list_chunks(document_ids=[document_id])

        retrieved = self.retriever.retrieve(
            query=query,
            query_vector=query_vector,
            chunks=candidates,
            top_k=RETRIEVAL_FINAL_TOP_K,
            threshold=max(0.2, RETRIEVAL_THRESHOLD - 0.1),
            retrieval_mode="hybrid",
            use_reranking=True,
        )
        if not retrieved and candidates:
            retrieved = sorted(candidates, key=lambda x: x.get("combined_score", 0), reverse=True)[:5]
        return {
            "document_id": document_id,
            "query": query,
            "retrieved_context": retrieved,
            "embedding_model": embedding_model,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Streaming
    # ─────────────────────────────────────────────────────────────────────────

    async def stream_chat(self, **kwargs: Any) -> AsyncIterator[str]:
        """Stream chat response với token streaming thật (TTFB thấp)."""
        loop = asyncio.get_running_loop()
        prepare = await loop.run_in_executor(None, lambda: self._prepare_stream_chat(**kwargs))

        if prepare.get("error"):
            yield f"data: {json.dumps({'event': 'error', 'detail': prepare['error']}, ensure_ascii=False)}\n\n"
            return

        conv_id = prepare["conversation_id"]
        yield f"data: {json.dumps({'event': 'start', 'conversation_id': conv_id, 'intent': prepare.get('intent')}, ensure_ascii=False)}\n\n"

        for stage_event in prepare.get("stage_events", []):
            yield f"data: {json.dumps(stage_event, ensure_ascii=False)}\n\n"

        assembled = ""
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _produce_tokens():
            try:
                for token in self.generator.stream_answer(
                    prepare["query"],
                    prepare.get("retrieved", []),
                    chat_history=prepare.get("chat_history"),
                    temperature=prepare.get("temperature", 0.2),
                    general_chat=prepare.get("general_chat", False),
                    compressed_context=prepare.get("compressed_context"),
                    selected_document_ids=prepare.get("selected_document_ids"),
                    selected_filenames=prepare.get("selected_filenames"),
                ):
                    loop.call_soon_threadsafe(token_queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)

        import threading
        threading.Thread(target=_produce_tokens, daemon=True).start()

        while True:
            token = await token_queue.get()
            if token is None:
                break
            assembled += token
            payload = {
                "event": "token",
                "content": assembled,
                "delta": token,
                "conversation_id": conv_id,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        response = await loop.run_in_executor(
            None, lambda: self._finalize_stream_chat(prepare, assembled)
        )
        yield f"data: {json.dumps({'event': 'done', 'response': response}, ensure_ascii=False)}\n\n"

    def _prepare_stream_chat(self, **kwargs: Any) -> dict[str, Any]:
        """Chạy retrieval + context compression trước, trả metadata cho streaming generation."""
        query = kwargs.get("query", "")
        conversation_id = kwargs.get("conversation_id")
        document_ids = kwargs.get("document_ids")
        top_k = kwargs.get("top_k", RETRIEVAL_FINAL_TOP_K)
        threshold = kwargs.get("threshold", RETRIEVAL_THRESHOLD)
        retrieval_mode = kwargs.get("retrieval_mode", "hybrid")
        use_reranking = kwargs.get("use_reranking", True)
        temperature = kwargs.get("temperature", 0.2)

        stage_events: list[dict[str, Any]] = []
        timer = StageTimer()

        def _emit_stage(stage: str, status: str = "active") -> None:
            stage_events.append({"event": "stage", "stage": stage, "status": status})

        embedding_model = self._resolve_embedding_model(document_ids)
        conv_id = self.repository.ensure_conversation(conversation_id, title=query[:60] or "New chat")
        chat_history = self.repository.list_messages(conv_id)
        self.repository.append_message(conv_id, "user", query)

        from .agent import classify_intent
        intent = classify_intent(query, document_ids)
        top_k = compute_dynamic_top_k(
            base_top_k=top_k, query=query, document_ids=document_ids, intent=intent,
        )

        if intent == "GENERAL":
            _emit_stage("question", "done")
            _emit_stage("generation", "active")
            return {
                "conversation_id": conv_id,
                "query": query,
                "chat_history": chat_history,
                "retrieved": [],
                "general_chat": True,
                "intent": "general",
                "temperature": temperature,
                "threshold": threshold,
                "stage_events": stage_events,
            }

        selected_filenames = self._resolve_selected_filenames(document_ids)
        stream_threshold = min(threshold, 0.25) if document_ids and len(document_ids) > 1 else threshold

        _emit_stage("question", "done")
        _emit_stage("embedding", "active")
        timer.start("embedding")
        query_vector = self.embedding_service.embed_query(query, embedding_model)
        timer.stop("embedding")
        _emit_stage("embedding", "done")

        _emit_stage("retrieval", "active")
        timer.start("retrieval")
        candidates = self._gather_retrieval_candidates(
            query_vector=query_vector,
            document_ids=document_ids,
            intent=intent,
            embedding_model=embedding_model,
            current_query=query,
            retries=0,
            use_expansion=False,
        )

        _emit_stage("retrieval", "done")
        _emit_stage("crossencoder", "active")
        timer.start("reranking")
        retrieved = self.retriever.retrieve(
            query=query,
            query_vector=query_vector,
            chunks=candidates,
            top_k=top_k,
            threshold=stream_threshold,
            retrieval_mode=retrieval_mode,
            use_reranking=use_reranking,
            document_ids=document_ids,
        )
        timer.stop("reranking")
        timer.stop("retrieval")
        _emit_stage("crossencoder", "done")
        _emit_stage("top_k", "done")
        self._warn_missing_doc_coverage(document_ids, retrieved)

        def _acb_stage(name: str, status: str) -> None:
            _emit_stage(name, status)

        if RAG_ADAPTIVE_CONTEXT:
            _emit_stage("acb_intent", "active")
            compressed = build_retrieved_context(
                retrieved, query, timer,
                document_ids=document_ids,
                stage_callback=_acb_stage,
            )
            for sub in ("acb_intent", "acb_chunks", "acb_summary", "acb_facts", "acb_compose"):
                if not any(e.get("stage") == sub for e in stage_events):
                    _emit_stage(sub, "done")
        else:
            _emit_stage("context_compression", "active")
            compressed = build_retrieved_context(retrieved, query, timer, document_ids=document_ids)
            _emit_stage("context_compression", "done")

        _emit_stage("prompt", "active")
        timer.start("prompt_build")
        self.generator.compose_prompt(
            query,
            retrieved,
            chat_history=chat_history,
            compressed_context=compressed if compressed.enabled else None,
            selected_document_ids=document_ids,
            selected_filenames=selected_filenames,
        )
        timer.stop("prompt_build")
        _emit_stage("prompt", "done")
        _emit_stage("generation", "active")

        return {
            "conversation_id": conv_id,
            "query": query,
            "chat_history": chat_history,
            "retrieved": retrieved,
            "compressed_context": compressed if compressed.enabled else None,
            "compression_meta": {
                "enabled": compressed.enabled,
                "skipped_reason": compressed.skipped_reason,
                "compression_ratio": compressed.compression_ratio,
                "hybrid_algo": compressed.hybrid_algo_key,
                "mode": compressed.mode,
                "compression_tier": compressed.compression_tier,
                "query_intent": compressed.query_intent,
                "chunks_kept": compressed.dynamic_chunks_kept or len(compressed.top_original_chunks),
                "summary_tokens": compressed.summary_tokens,
                "token_reduction": compressed.context_details.get("token_reduction"),
                "latency_saving_estimate_s": compressed.latency_saving_estimate,
            },
            "context_details": compressed.context_details if compressed.mode == "adaptive" else None,
            "general_chat": False,
            "intent": intent.lower(),
            "temperature": temperature,
            "threshold": threshold,
            "embedding_model": embedding_model,
            "retrieval_mode": retrieval_mode,
            "use_reranking": use_reranking,
            "top_k": top_k,
            "stage_events": stage_events,
            "latency_timer": timer,
            "selected_document_ids": document_ids,
            "selected_filenames": selected_filenames,
        }

    def _finalize_stream_chat(self, prepare: dict[str, Any], answer: str) -> dict[str, Any]:
        """Lưu tin nhắn assistant sau khi stream xong."""
        conv_id = prepare["conversation_id"]
        retrieved = prepare.get("retrieved", [])
        threshold = prepare.get("threshold", RETRIEVAL_THRESHOLD)
        compression_meta = prepare.get("compression_meta", {})
        compressed_ctx = prepare.get("compressed_context")

        faithfulness, retrieval_confidence, eval_metrics = self._compute_answer_metrics(
            answer=answer,
            retrieved=retrieved,
            compressed=compressed_ctx,
        )

        timer: StageTimer | None = prepare.get("latency_timer")
        latency_details = None
        if timer:
            latency_details = {
                "stage_breakdown": timer.report(),
                "compression_enabled": compression_meta.get("enabled", False),
                "compression_ratio": compression_meta.get("compression_ratio", 1.0),
                "adaptive_mode": compression_meta.get("mode") == "adaptive",
                "token_reduction": compression_meta.get("token_reduction"),
                "chunks_kept": compression_meta.get("chunks_kept"),
                "summary_tokens": compression_meta.get("summary_tokens"),
                "latency_saving_estimate_s": compression_meta.get("latency_saving_estimate_s"),
            }

        response = {
            "conversation_id": conv_id,
            "answer": answer,
            "confidence": retrieval_confidence if not prepare.get("general_chat") else 1.0,
            "retrieval_confidence": retrieval_confidence if not prepare.get("general_chat") else 1.0,
            "faithfulness": faithfulness if not prepare.get("general_chat") else 1.0,
            "grounded": not prepare.get("general_chat"),
            "retrieved_context": retrieved,
            "intent": prepare.get("intent", "document_qa"),
            "evaluation": eval_metrics,
            "context_compression": compression_meta,
            "context_details": prepare.get("context_details"),
            "latency_details": latency_details,
        }

        self.repository.append_message(
            conv_id,
            "assistant",
            answer,
            citations=retrieved,
            confidence=response["confidence"],
            retrieval_threshold=threshold,
            evaluation=eval_metrics,
        )
        self._trigger_auto_title(conv_id)
        return response

    async def stream_summarize(
        self,
        *,
        document_id: str,
        query: str = "Tóm tắt nội dung chính của tài liệu",
    ) -> AsyncIterator[str]:
        """Stream tóm tắt tài liệu với token streaming thật."""
        loop = asyncio.get_running_loop()
        prepare = await loop.run_in_executor(
            None,
            lambda: self._prepare_summarize_document(document_id=document_id, query=query),
        )

        yield f"data: {json.dumps({'event': 'start', 'document_id': document_id}, ensure_ascii=False)}\n\n"

        contexts = prepare.get("retrieved_context", [])
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _produce():
            try:
                for token in self._summarizer_stream_tokens(contexts):
                    loop.call_soon_threadsafe(token_queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)

        import threading
        threading.Thread(target=_produce, daemon=True).start()

        assembled = ""
        while True:
            token = await token_queue.get()
            if token is None:
                break
            assembled += token
            payload = {"event": "token", "content": assembled, "delta": token}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        result = {
            "document_id": document_id,
            "summary": assembled,
            "word_count": len(assembled.split()),
            "retrieved_context": contexts,
        }
        yield f"data: {json.dumps({'event': 'done', 'result': result}, ensure_ascii=False)}\n\n"

    def _summarizer_stream_tokens(self, contexts: list[dict[str, Any]]):
        """Stream token tóm tắt từ contexts đã retrieve."""
        from .summarizer import RAGTransformerSummarizer
        from .rag_config import SUMMARIZE_PROMPT_TEMPLATE, RAG_GENERATOR_TYPE, resolve_generation_profile
        from .summarizer import _pick_available_model, _run_llm_api_stream, _run_transformer_generate_stream

        if not contexts:
            yield "Không tìm thấy nội dung phù hợp trong tài liệu."
            return

        context_parts = []
        for i, chunk in enumerate(contexts, start=1):
            filename = chunk.get("filename", "?")
            context_parts.append(f"[Nguồn {i} — {filename}]\n{chunk['text']}")
        full_context = "\n\n".join(context_parts)[:4000]
        prompt = SUMMARIZE_PROMPT_TEMPLATE.format(context=full_context)

        if RAG_GENERATOR_TYPE in {"gemini", "openai", "ollama"}:
            yielded = False
            for token in _run_llm_api_stream(prompt, RAG_GENERATOR_TYPE):
                yielded = True
                yield token
            if yielded:
                return

        model_key = _pick_available_model()
        if model_key:
            profile = resolve_generation_profile(model_key)
            for token in _run_transformer_generate_stream(model_key, full_context, profile):
                yield token
            return

        summarizer = RAGTransformerSummarizer()
        yield summarizer._extractive_fallback(contexts, max_sentences=8)

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
            from .summarizer import (
                _pick_available_model,
                _run_transformer_generate_batch,
                _run_llm_api_stream,
                _run_transformer_generate_stream,
            )
            from .rag_config import resolve_generation_profile, RAG_SUMMARIZE_BATCH_SIZE
            model_key = _pick_available_model()
            if model_key:
                profile = resolve_generation_profile(model_key)
                
                # Trích xuất văn bản tóm tắt thô của từng tài liệu
                current_summaries = []
                for s, doc_id in zip(individual_summaries, document_ids):
                    cleaned_s = s.replace(f"--- Tóm tắt tài liệu (ID: {doc_id}): ---\n", "").strip()
                    if cleaned_s:
                        current_summaries.append(cleaned_s)
                
                if current_summaries:
                    logger.info("🌳 Bắt đầu tóm tắt phân cấp đệ quy cho %d tài liệu bằng model local: %s", len(current_summaries), model_key)
                    step = 1
                    while len(current_summaries) > 1:
                        next_summaries = []
                        logger.info("🌳 Vòng tóm tắt phân cấp %d: Số lượng bản tóm tắt hiện tại là %d", step, len(current_summaries))
                        
                        pair_prompts: list[str] = []
                        pair_indices: list[tuple[int, int]] = []
                        for i in range(0, len(current_summaries), 2):
                            if i + 1 < len(current_summaries):
                                pair_text = (
                                    f"Bản tóm tắt 1:\n{current_summaries[i]}\n\n"
                                    f"Bản tóm tắt 2:\n{current_summaries[i+1]}"
                                )
                                pair_prompts.append(
                                    "Hãy viết một bản tóm tắt kết hợp nội dung của hai văn bản dưới đây "
                                    "một cách logic và ngắn gọn:\n\n"
                                    f"{pair_text}\n\n"
                                    "Bản tóm tắt kết hợp tiếng Việt:"
                                )
                                pair_indices.append((i, i + 1))
                            else:
                                next_summaries.append(current_summaries[i])

                        batch_size = max(1, RAG_SUMMARIZE_BATCH_SIZE)
                        for b_start in range(0, len(pair_prompts), batch_size):
                            batch = pair_prompts[b_start : b_start + batch_size]
                            batch_results = _run_transformer_generate_batch(
                                model_key, batch, profile
                            )
                            for (i, j), pair_summary in zip(
                                pair_indices[b_start : b_start + batch_size],
                                batch_results,
                            ):
                                if not pair_summary or len(pair_summary.split()) < 10:
                                    logger.warning(
                                        "⚠️ Lỗi sinh tóm tắt cặp ở vòng %d, dùng phương án ghép thô", step
                                    )
                                    pair_summary = f"{current_summaries[i]}\n{current_summaries[j]}"
                                next_summaries.append(pair_summary)
                        
                        current_summaries = next_summaries
                        step += 1
                    
                    summary = current_summaries[0]
                    model_used = model_key
                else:
                    summary = ""

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
