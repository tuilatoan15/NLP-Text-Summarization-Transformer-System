"""Kiểm tra nhanh không cần DB/model registry."""
import sys
sys.path.insert(0, ".")

# Test 1: rag_config (không dependency nặng)
from backend.services.rag.rag_config import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    VECTOR_WEIGHT, BM25_WEIGHT,
    RETRIEVAL_FINAL_TOP_K, RETRIEVAL_THRESHOLD,
    BARTPHO_GENERATION, MT5_GENERATION, VIT5_GENERATION,
    GENERATION_PROFILES, SUMMARIZE_PROMPT_TEMPLATE, QA_PROMPT_TEMPLATE,
)
assert VECTOR_WEIGHT + BM25_WEIGHT == 1.0
assert EMBEDDING_MODEL == "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
assert CHUNK_SIZE == 512
assert BARTPHO_GENERATION.no_repeat_ngram_size == 4
assert BARTPHO_GENERATION.num_beams == 5
assert MT5_GENERATION.do_sample == True
assert MT5_GENERATION.temperature == 0.2
assert VIT5_GENERATION.repetition_penalty == 1.3
assert "chuyen gia" in SUMMARIZE_PROMPT_TEMPLATE or "chuyên gia" in SUMMARIZE_PROMPT_TEMPLATE
print("[1] rag_config.py OK")

# Test 2: _build_gen_kwargs
from backend.services.rag.summarizer import _build_gen_kwargs
bartpho_kw = _build_gen_kwargs(BARTPHO_GENERATION)
mt5_kw = _build_gen_kwargs(MT5_GENERATION)
assert "temperature" in bartpho_kw and bartpho_kw["temperature"] == 0.15
assert "early_stopping" not in mt5_kw, "mT5 do_sample=True phai bo early_stopping"
assert mt5_kw.get("temperature") == 0.2
assert mt5_kw.get("top_p") == 0.90
assert mt5_kw.get("num_beams") == 1
print("[2] _build_gen_kwargs OK - BARTPho:", {k:v for k,v in bartpho_kw.items() if k not in ("min_new_tokens","max_new_tokens")})
print("    mT5:", {k:v for k,v in mt5_kw.items() if k not in ("min_new_tokens","max_new_tokens")})

# Test 3: Reranker fallback (khong load model)
from backend.services.rag.reranker import CrossEncoderReranker
reranker = CrossEncoderReranker()
chunks = [
    {"text": "Tri tue nhan tao la gi?", "combined_score": 0.8, "embedding_score": 0.85, "bm25_score": 0.7,
     "document_id": "d1", "filename": "test.pdf", "page": 1, "chunk_id": "c1", "id": "c1"},
    {"text": "AI giup con nguoi lam viec hieu qua.", "combined_score": 0.65, "embedding_score": 0.7, "bm25_score": 0.55,
     "document_id": "d1", "filename": "test.pdf", "page": 2, "chunk_id": "c2", "id": "c2"},
]
result = reranker.rerank("AI la gi?", chunks, top_k=2, threshold=0.0)
assert all("rerank_score" in r for r in result)
assert all("rank" in r for r in result)
assert result[0]["rank"] == 1
print(f"[3] CrossEncoderReranker fallback OK -> {len(result)} chunks, scores={[r['rerank_score'] for r in result]}")

# Test 4: HybridRetriever
from backend.services.rag.retriever import HybridRetriever
retriever = HybridRetriever()
chunks_with_vec = [
    {**c, "vector": [0.1 + i*0.05]*128}
    for i, c in enumerate(chunks)
]
retrieved = retriever.retrieve(
    query="AI la gi?",
    query_vector=[0.1]*128,
    chunks=chunks_with_vec,
    threshold=0.0,
)
assert isinstance(retrieved, list)
print(f"[4] HybridRetriever OK -> {len(retrieved)} chunks")

print("\nTAT CA KIEM TRA THANH CONG!")
