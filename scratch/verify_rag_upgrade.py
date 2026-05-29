"""
verify_rag_upgrade.py — Kiểm tra toàn bộ RAG upgrade imports và logic.
"""
import sys
sys.path.insert(0, ".")

print("=" * 60)
print("KIỂM TRA RAG UPGRADE")
print("=" * 60)

# ── 1. rag_config ─────────────────────────────────────────────
print("\n[1] rag_config.py...")
from backend.services.rag.rag_config import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    VECTOR_WEIGHT, BM25_WEIGHT,
    RETRIEVAL_INITIAL_TOP_K, RETRIEVAL_PRE_RERANK_TOP_K, RETRIEVAL_FINAL_TOP_K,
    RETRIEVAL_THRESHOLD, RERANKER_MODEL, USE_RERANKING,
    BARTPHO_GENERATION, VIT5_GENERATION, MT5_GENERATION,
    GENERATION_PROFILES, PREFERRED_SUMMARIZER_ORDER,
    SUMMARIZE_PROMPT_TEMPLATE, QA_PROMPT_TEMPLATE,
)
print(f"  Embedding model: {EMBEDDING_MODEL}")
print(f"  Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
print(f"  Retrieval weights: {VECTOR_WEIGHT}V + {BM25_WEIGHT}BM25")
print(f"  Top-K pipeline: {RETRIEVAL_INITIAL_TOP_K} → {RETRIEVAL_PRE_RERANK_TOP_K} → {RETRIEVAL_FINAL_TOP_K}")
print(f"  Threshold: {RETRIEVAL_THRESHOLD}, Reranker: {RERANKER_MODEL}")
print(f"  Reranking: {USE_RERANKING}")
print(f"  BARTPho: beams={BARTPHO_GENERATION.num_beams}, ngram={BARTPHO_GENERATION.no_repeat_ngram_size}, penalty={BARTPHO_GENERATION.repetition_penalty}")
print(f"  mT5:     do_sample={MT5_GENERATION.do_sample}, temp={MT5_GENERATION.temperature}, top_p={MT5_GENERATION.top_p}")
print(f"  ViT5:    penalty={VIT5_GENERATION.repetition_penalty}, beams={VIT5_GENERATION.num_beams}")
assert VECTOR_WEIGHT + BM25_WEIGHT == 1.0, "Weights phải cộng thành 1.0!"
print("  ✅ OK")

# ── 2. reranker ───────────────────────────────────────────────
print("\n[2] reranker.py...")
from backend.services.rag.reranker import CrossEncoderReranker
reranker = CrossEncoderReranker()

# Test fallback reranking (không cần load model thực)
mock_chunks = [
    {"text": "Trí tuệ nhân tạo đang phát triển mạnh mẽ.", "combined_score": 0.85,
     "document_id": "d1", "filename": "test.pdf", "page": 1, "chunk_id": "c1", "embedding_score": 0.9, "bm25_score": 0.7},
    {"text": "AI hỗ trợ giáo dục và y tế hiệu quả.", "combined_score": 0.72,
     "document_id": "d1", "filename": "test.pdf", "page": 2, "chunk_id": "c2", "embedding_score": 0.8, "bm25_score": 0.6},
    {"text": "Các thuật toán học máy cần dữ liệu lớn.", "combined_score": 0.60,
     "document_id": "d1", "filename": "test.pdf", "page": 3, "chunk_id": "c3", "embedding_score": 0.65, "bm25_score": 0.5},
]
result = reranker.rerank("AI là gì?", mock_chunks, top_k=2, threshold=0.1)
assert isinstance(result, list), "Kết quả phải là list"
assert all("rerank_score" in r for r in result), "Thiếu rerank_score"
assert all("rank" in r for r in result), "Thiếu rank"
print(f"  Reranked {len(mock_chunks)} → {len(result)} chunks")
print(f"  Reranker available: {CrossEncoderReranker.is_available()}")
print("  ✅ OK")

# ── 3. summarizer ─────────────────────────────────────────────
print("\n[3] summarizer.py...")
from backend.services.rag.summarizer import RAGTransformerSummarizer, _build_gen_kwargs
summarizer = RAGTransformerSummarizer()

# Test _build_gen_kwargs
bartpho_kwargs = _build_gen_kwargs(BARTPHO_GENERATION)
assert "early_stopping" in bartpho_kwargs, "BARTPho cần early_stopping"
assert bartpho_kwargs["do_sample"] == False

mt5_kwargs = _build_gen_kwargs(MT5_GENERATION)
assert "early_stopping" not in mt5_kwargs, "mT5 không được có early_stopping khi do_sample=True"
assert mt5_kwargs.get("temperature") == 0.2
assert mt5_kwargs.get("top_p") == 0.90
print(f"  BARTPho gen_kwargs: {bartpho_kwargs}")
print(f"  mT5 gen_kwargs: {mt5_kwargs}")

# Test fallback (không có model loaded)
result = summarizer.summarize_context(mock_chunks)
assert "summary" in result
assert result["word_count"] >= 0
print(f"  summarize_context fallback: '{result['summary'][:80]}...'")
print(f"  model_used: {result['model_used']}, fallback: {result['fallback_used']}")
print("  ✅ OK")

# ── 4. retriever ──────────────────────────────────────────────
print("\n[4] retriever.py...")
from backend.services.rag.retriever import HybridRetriever
retriever = HybridRetriever()

mock_query_vector = [0.1] * 768  # fake 768-dim vector
mock_chunks_with_vector = [
    {**c, "id": c["chunk_id"], "vector": [0.1 + i*0.01] * 768}
    for i, c in enumerate(mock_chunks)
]
retrieved = retriever.retrieve(
    query="AI là gì?",
    query_vector=mock_query_vector,
    chunks=mock_chunks_with_vector,
    threshold=0.0,  # threshold thấp để test luôn có kết quả
)
assert isinstance(retrieved, list)
print(f"  Hybrid + Rerank: {len(mock_chunks_with_vector)} → {len(retrieved)} chunks")
if retrieved:
    print(f"  Top chunk score: {retrieved[0].get('rerank_score', retrieved[0].get('combined_score', '?'))}")
print("  ✅ OK")

# ── 5. generator ──────────────────────────────────────────────
print("\n[5] generator.py...")
from backend.services.rag.generator import GroundedGenerator
generator = GroundedGenerator()

answer_result = generator.build_answer("AI là gì?", retrieved if retrieved else mock_chunks)
assert "answer" in answer_result
assert "confidence" in answer_result
assert "grounded" in answer_result
print(f"  Answer: '{answer_result['answer'][:80]}...'")
print(f"  Confidence: {answer_result['confidence']}, model: {answer_result.get('model_used')}")

prompt = generator.prompt_template(mock_chunks, "Test câu hỏi?")
assert "Ngữ cảnh" in prompt or "NGỮCẢNH" in prompt or "TRẢ LỜI" in prompt
print(f"  Prompt preview: '{prompt[:120]}...'")
print("  ✅ OK")

# ── 6. service ────────────────────────────────────────────────
print("\n[6] service.py...")
from backend.services.rag.service import RAGChatService
# Chỉ test instantiation (không chạy DB thực tế)
print("  RAGChatService import: OK")
print("  ✅ OK")

print("\n" + "=" * 60)
print("✅ TẤT CẢ KIỂM TRA THÀNH CÔNG!")
print("=" * 60)
