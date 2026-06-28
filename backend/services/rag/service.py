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
)
from .repository import RAGRepository
from .retriever import HybridRetriever
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


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
            from .summarizer import _pick_available_model, _run_transformer_generate, GENERATION_PROFILES
            model_key = _pick_available_model()
            if model_key:
                profile = GENERATION_PROFILES[model_key]
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

    def _trigger_auto_title(self, conv_id: str) -> None:
        try:
            messages = self.repository.list_messages(conv_id)
            if 2 <= len(messages) <= 4:
                first_user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                if first_user_msg:
                    new_title = generate_title_for_conversation(first_user_msg)
                    self.repository.rename_conversation(conv_id, new_title)
                    logger.info(f"Auto title generated for conversation {conv_id}: {new_title}")
        except Exception as e:
            logger.error(f"Failed to auto-update conversation title: {e}")

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

        # RAPTOR-lite hierarchical indexing (Recursive GMM Tree)
        try:
            from .raptor import RaptorIndexer
            indexer = RaptorIndexer(
                repository=self.repository,
                vector_store=self.vector_store,
                embedding_service=self.embedding_service,
                generator=self.generator
            )
            indexer.build_tree(document_id, chunks, vectors, embedding_model, max_levels=3)
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

        # Resolve embedding model from document metadata if available to avoid dimension mismatch (e.g. 768 vs 1024)
        if document_ids:
            doc_record = self.repository.list_documents()
            for doc in doc_record:
                if doc["id"] in document_ids:
                    embedding_model = doc.get("metadata", {}).get("embedding_model", EMBEDDING_MODEL)
                    break
        else:
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
        t_embed_start = time.perf_counter()
        query_vector = self.embedding_service.embed_query(query, embedding_model)
        embedding_time = time.perf_counter() - t_embed_start

        # 1. GENERAL INTENT (No retrieval needed)
        if intent == "GENERAL":
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
                "total": f"{total_time:.4f}s"
            }
            logger.info("RAG Latency Details (GENERAL):\n%s", json.dumps(latency_details, indent=2))

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
        current_query = query
        retries = 0
        max_retries = 2  # Cho phép thử lại tối đa 2 lần
        current_threshold = threshold
        current_top_k = top_k
        
        t_retrieval_start = time.perf_counter()
        
        while True:
            # Embed truy vấn hiện tại
            t_embed_loop = time.perf_counter()
            query_vector = self.embedding_service.embed_query(current_query, embedding_model)
            loop_embedding_time = time.perf_counter() - t_embed_loop
            
            # Lấy candidates dựa trên intent
            if intent == "DOCUMENT_QA":
                candidates = self.vector_store.query(
                    query_vector=query_vector,
                    top_k=RETRIEVAL_INITIAL_TOP_K,
                    document_ids=document_ids or None,
                )
                # Query expansion (chỉ chạy ở lần thử đầu tiên)
                if retries == 0:
                    expanded = expand_query(current_query)
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
            else:  # intent == "SUMMARIZE"
                # Nếu document_ids rỗng nhưng intent là SUMMARIZE, chuyển về DOCUMENT_QA
                if not document_ids:
                    intent = "DOCUMENT_QA"
                    candidates = self.vector_store.query(
                        query_vector=query_vector,
                        top_k=RETRIEVAL_INITIAL_TOP_K,
                        document_ids=None,
                    )
                else:
                    candidates = self.vector_store.query(
                        query_vector=query_vector,
                        top_k=RETRIEVAL_INITIAL_TOP_K,
                        document_ids=document_ids,
                    )
                    summary_candidates = [c for c in candidates if c.get("metadata", {}).get("chunk_type") == "summary"]
                    if summary_candidates:
                        candidates = summary_candidates
                    # Nếu không có summary chunks, giữ nguyên ALL candidates (base chunks)
                    # để retriever vẫn có dữ liệu để rerank thay vì trả rỗng

            # Hybrid retrieval + Cross-Encoder reranking
            retrieved = self.retriever.retrieve(
                query=current_query,
                query_vector=query_vector,
                chunks=candidates,
                top_k=current_top_k,
                threshold=current_threshold,
                retrieval_mode=retrieval_mode,
                use_reranking=use_reranking,
            )

            # Sinh câu trả lời bằng Generator
            t_gen_loop = time.perf_counter()
            generation = self.generator.build_answer(
                current_query, retrieved, chat_history=chat_history, temperature=temperature
            )
            generation_time = time.perf_counter() - t_gen_loop

            # Gọi LLM Judge đánh giá chất lượng câu trả lời (Self-Reflection)
            if not retrieved:
                judge_result = {
                    "faithfulness": "yes",
                    "relevance": "no",
                    "sufficiency": "no",
                    "feedback": "Không tìm thấy bất kỳ phân đoạn tài liệu phù hợp nào."
                }
            else:
                context_str = "\n\n".join(c["text"] for c in retrieved)
                from .agent import evaluate_answer
                judge_result = evaluate_answer(query, context_str, generation["answer"])

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

        # Lấy chi tiết thời gian từ retriever
        bm25_time = self.retriever.last_latency.get("bm25", 0.0)
        fusion_time = self.retriever.last_latency.get("vector_rrf", 0.0)
        rerank_time = self.retriever.last_latency.get("rerank", 0.0)
        actual_retrieval_only = max(0.0, retrieval_time - rerank_time)
        prompt = self.generator.prompt_template(retrieved, current_query)

        # Đánh giá chất lượng câu trả lời (Fact-checking/Hallucination risk) bằng NLI audit
        eval_metrics = None
        if retrieved:
            try:
                from evaluation.hallucination import audit_summary
                source_text = "\n\n".join(c["text"] for c in retrieved)
                formatted_chunks = [{"chunk_id": c.get("chunk_id") or c.get("id"), "text": c["text"]} for c in retrieved]
                audit_res = audit_summary(generation["answer"], source_text, chunks=formatted_chunks, mode="fast")
                eval_metrics = {
                    "consistency_score": float(audit_res.get("consistency_score", 0.0)),
                    "grounding_coverage": float(audit_res.get("grounding_coverage", 0.0)),
                    "semantic_coverage": float(audit_res.get("semantic_coverage", 0.0)),
                    "hallucination_risk": str(audit_res.get("hallucination_risk", "low")),
                }
            except Exception as exc:
                logger.error("Failed to run hallucination audit on chat response: %s", exc)

        total_time = time.perf_counter() - t_total_start

        latency_details = {
            "embedding": f"{embedding_time:.4f}s",
            "retrieval": f"{actual_retrieval_only:.4f}s",
            "reranking": f"{rerank_time:.4f}s",
            "generation": f"{generation_time:.4f}s",
            "total": f"{total_time:.4f}s"
        }
        logger.info("RAG Latency Details:\n%s", json.dumps(latency_details, indent=2))

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
            "latency_details": latency_details,
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
                        
                        for i in range(0, len(current_summaries), 2):
                            if i + 1 < len(current_summaries):
                                pair_text = f"Bản tóm tắt 1:\n{current_summaries[i]}\n\nBản tóm tắt 2:\n{current_summaries[i+1]}"
                                pair_prompt = (
                                    "Hãy viết một bản tóm tắt kết hợp nội dung của hai văn bản dưới đây một cách logic và ngắn gọn:\n\n"
                                    f"{pair_text}\n\n"
                                    "Bản tóm tắt kết hợp tiếng Việt:"
                                )
                                pair_summary = _run_transformer_generate(model_key, pair_prompt, profile)
                                if not pair_summary or len(pair_summary.split()) < 10:
                                    logger.warning("⚠️ Lỗi sinh tóm tắt cặp ở vòng %d, dùng phương án ghép thô", step)
                                    pair_summary = f"{current_summaries[i]}\n{current_summaries[i+1]}"
                                next_summaries.append(pair_summary)
                            else:
                                next_summaries.append(current_summaries[i])
                        
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
