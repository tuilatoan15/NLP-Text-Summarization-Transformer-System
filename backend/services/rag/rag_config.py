"""
rag_config.py — Cấu hình RAG cứng (hardcode) theo chuẩn công nghiệp.

Toàn bộ tham số ở đây đã được tinh chỉnh tối ưu.
Người dùng KHÔNG cần và KHÔNG được phép thay đổi từ frontend.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from typing import Any
from dotenv import load_dotenv

# Load env variables (if not loaded yet)
load_dotenv()

RAG_GENERATOR_TYPE: str = os.getenv("RAG_GENERATOR_TYPE", "local").lower()
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_API_URL: str = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE — tối ưu latency (có thể tắt qua env nếu cần chất lượng tối đa)
# ─────────────────────────────────────────────────────────────────────────────
RAG_USE_LLM_INTENT: bool = os.getenv("RAG_USE_LLM_INTENT", "0").lower() in ("1", "true", "yes")
RAG_USE_LLM_JUDGE: bool = os.getenv("RAG_USE_LLM_JUDGE", "1").lower() in ("1", "true", "yes")
RAG_USE_LLM_QUERY_EXPANSION: bool = os.getenv("RAG_USE_QUERY_EXPANSION", "1").lower() in ("1", "true", "yes")
RAG_SKIP_JUDGE_MIN_RERANK: float = float(os.getenv("RAG_SKIP_JUDGE_MIN_RERANK", "0.72"))
RAG_RESPONSE_CACHE: bool = os.getenv("RAG_RESPONSE_CACHE", "1").lower() in ("1", "true", "yes")
RAG_RETRIEVAL_CACHE: bool = os.getenv("RAG_RETRIEVAL_CACHE", "1").lower() in ("1", "true", "yes")
RAG_RESPONSE_CACHE_TTL: int = int(os.getenv("RAG_RESPONSE_CACHE_TTL", "300"))
RAG_RETRIEVAL_CACHE_TTL: int = int(os.getenv("RAG_RETRIEVAL_CACHE_TTL", "600"))
RAG_EMBEDDING_FP16: bool = os.getenv("RAG_EMBEDDING_FP16", "1").lower() in ("1", "true", "yes")
RAG_EMBEDDING_BATCH_SIZE: int = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "32"))
RAG_RERANKER_BATCH_SIZE: int = int(os.getenv("RAG_RERANKER_BATCH_SIZE", "16"))
RAG_RERANKER_FP16: bool = os.getenv("RAG_RERANKER_FP16", "1").lower() in ("1", "true", "yes")
RAG_PARALLEL_RETRIEVAL: bool = os.getenv("RAG_PARALLEL_RETRIEVAL", "1").lower() in ("1", "true", "yes")
RAG_EVALUATE_HALLUCINATION: bool = os.getenv("RAG_EVALUATE_HALLUCINATION", "0").lower() in ("1", "true", "yes")
RAG_RRF_K: float = float(os.getenv("RAG_RRF_K", "60"))
RAG_USE_RAPTOR: bool = os.getenv("RAG_USE_RAPTOR", "1").lower() in ("1", "true", "yes")
RAG_RAPTOR_BACKGROUND: bool = os.getenv("RAG_RAPTOR_BACKGROUND", "1").lower() in ("1", "true", "yes")
RAG_VERBOSE_LOG: bool = os.getenv("RAG_VERBOSE_LOG", "0").lower() in ("1", "true", "yes")
RAG_SENTENCE_EMBED_CACHE: bool = os.getenv("RAG_SENTENCE_EMBED_CACHE", "1").lower() in ("1", "true", "yes")
PRELOAD_RAG_MODELS: bool = os.getenv("PRELOAD_RAG_MODELS", "1").lower() in ("1", "true", "yes")
RAG_TORCH_COMPILE: bool = os.getenv("RAG_TORCH_COMPILE", "0").lower() in ("1", "true", "yes")
RAG_EXPANSION_CACHE: bool = os.getenv("RAG_EXPANSION_CACHE", "1").lower() in ("1", "true", "yes")
RAG_EXPANSION_CACHE_TTL: int = int(os.getenv("RAG_EXPANSION_CACHE_TTL", "1800"))
RAG_EXPANSION_MIN_WORDS: int = int(os.getenv("RAG_EXPANSION_MIN_WORDS", "3"))
RAG_SUMMARIZE_BATCH_SIZE: int = int(os.getenv("RAG_SUMMARIZE_BATCH_SIZE", "4"))

# ─────────────────────────────────────────────────────────────────────────────
# HYBRID CONTEXT COMPRESSION — nén ngữ cảnh trước khi đưa vào LLM
# ─────────────────────────────────────────────────────────────────────────────
RAG_CONTEXT_COMPRESSION: bool = os.getenv("RAG_CONTEXT_COMPRESSION", "1").lower() in ("1", "true", "yes")
RAG_CONTEXT_COMPRESSION_THRESHOLD: int = int(os.getenv("RAG_CONTEXT_COMPRESSION_THRESHOLD", "2500"))
RAG_TOP_ORIGINAL_CHUNKS: int = int(os.getenv("RAG_TOP_ORIGINAL_CHUNKS", "3"))
RAG_SUMMARY_FOR_LONG_CONTEXT_ONLY: bool = os.getenv("RAG_SUMMARY_FOR_LONG_CONTEXT_ONLY", "1").lower() in ("1", "true", "yes")

# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE CONTEXT BUILDER — nâng cấp Context Compression
# ─────────────────────────────────────────────────────────────────────────────
RAG_ADAPTIVE_CONTEXT: bool = os.getenv("RAG_ADAPTIVE_CONTEXT", "1").lower() in ("1", "true", "yes")
RAG_DYNAMIC_CHUNK_RATIO: float = float(os.getenv("RAG_DYNAMIC_CHUNK_RATIO", "0.9"))
RAG_MIN_RERANK_SCORE: float = float(os.getenv("RAG_MIN_RERANK_SCORE", "0.85"))
RAG_LIGHT_COMPRESSION: int = int(os.getenv("RAG_LIGHT_COMPRESSION", "1500"))
RAG_MEDIUM_COMPRESSION: int = int(os.getenv("RAG_MEDIUM_COMPRESSION", "3000"))
RAG_HEAVY_COMPRESSION: int = int(os.getenv("RAG_HEAVY_COMPRESSION", "6000"))
RAG_ADAPTIVE_CONTEXT_CACHE: bool = os.getenv("RAG_ADAPTIVE_CONTEXT_CACHE", "1").lower() in ("1", "true", "yes")
RAG_ADAPTIVE_CONTEXT_CACHE_TTL: int = int(os.getenv("RAG_ADAPTIVE_CONTEXT_CACHE_TTL", "600"))

# Override generation profile (để trống = giữ mặc định chất lượng)
def _optional_float(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return float(raw)


def _optional_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return int(raw)


RAG_GEN_NUM_BEAMS: int | None = _optional_int("RAG_GEN_NUM_BEAMS")
RAG_GEN_LENGTH_PENALTY: float | None = _optional_float("RAG_GEN_LENGTH_PENALTY")
RAG_GEN_REPETITION_PENALTY: float | None = _optional_float("RAG_GEN_REPETITION_PENALTY")


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────
# VoVanPhuc/sup-SimCSE-VietNamese-phobert-base: mô hình chuyên tiếng Việt siêu nhẹ (~540MB), hiệu suất cao.
# Kích thước vector: 768 chiều — override via RAG_EMBEDDING_MODEL or DEFAULT_EMBEDDING_MODEL
EMBEDDING_MODEL: str = os.getenv(
    "RAG_EMBEDDING_MODEL",
    os.getenv("DEFAULT_EMBEDDING_MODEL", "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"),
)

# Fallback nếu model chính chưa được tải
EMBEDDING_MODEL_FALLBACK: str = "keepitreal/vietnamese-sbert"


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────
# 512 tokens ~ 350-400 từ tiếng Việt — vừa đủ ngữ cảnh, không vượt max_length model
CHUNK_SIZE: int = 512
# 80 tokens overlap: đảm bảo không mất ngữ cảnh tại ranh giới chunk
CHUNK_OVERLAP: int = 80


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL — HYBRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────
# Trọng số Hybrid: 70% Vector (ngữ nghĩa) + 30% BM25 (từ khóa chuyên ngành)
VECTOR_WEIGHT: float = 0.70
BM25_WEIGHT: float = 0.30

# Lấy top 30 candidates từ vector store trước khi rerank
RETRIEVAL_INITIAL_TOP_K: int = 30

# Sau khi BM25 + Vector kết hợp, lấy top 8 trước khi đưa vào Cross-Encoder (tối ưu hóa tiếng Việt dài)
RETRIEVAL_PRE_RERANK_TOP_K: int = 8

# Sau Cross-Encoder reranker, giữ lại top-K chunks (mặc định 10 cho context compression)
RETRIEVAL_FINAL_TOP_K: int = int(os.getenv("RAG_RETRIEVAL_FINAL_TOP_K", "10"))

# Ngưỡng similarity tối thiểu sau reranking (0.35: lý tưởng để giữ các phân đoạn tiếng Việt quan trọng)
RETRIEVAL_THRESHOLD: float = 0.35

# Luôn bật hybrid mode
RETRIEVAL_MODE: str = "hybrid"


# ─────────────────────────────────────────────────────────────────────────────
# RERANKING — CROSS-ENCODER
# ─────────────────────────────────────────────────────────────────────────────
# BAAI/bge-reranker-v2-m3: Cross-Encoder đa ngôn ngữ mạnh nhất hiện nay
# Chấm điểm lại từng cặp (query, chunk) để loại bỏ false positives từ ANN search
RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

# Fallback Cross-Encoder nhẹ hơn nếu bge-reranker-v2-m3 chưa tải
RERANKER_MODEL_FALLBACK: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Luôn bật reranking
USE_RERANKING: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION CONFIG — HARDCODE chống lặp từ và collapse ngữ nghĩa
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GenerationProfile:
    """Bộ tham số generation cố định cho từng mô hình."""
    num_beams: int
    no_repeat_ngram_size: int
    repetition_penalty: float
    length_penalty: float
    min_new_tokens: int
    max_new_tokens: int
    early_stopping: bool
    do_sample: bool
    temperature: float | None = None
    top_p: float | None = None


# BARTPho — syllable-level BART, ổn định nhất cho tiếng Việt
BARTPHO_GENERATION = GenerationProfile(
    num_beams=5,                # Beam search với 5 beams cho độ mạch lạc cao nhất
    no_repeat_ngram_size=4,     # Triệt tiêu lặp 4-gram tiếng Việt
    repetition_penalty=1.3,     # Phạt lặp từ để cấu trúc câu linh hoạt
    length_penalty=1.2,
    min_new_tokens=80,
    max_new_tokens=400,
    early_stopping=True,
    do_sample=True,             # Bật sample ở nhiệt độ thấp
    temperature=0.15,           # Nhiệt độ 0.15 cực thấp giúp AI bám sát văn bản gốc, tránh bịa đặt
    top_p=0.95,
)

# ViT5 — T5 fine-tuned tiếng Việt
VIT5_GENERATION = GenerationProfile(
    num_beams=5,                # Tăng lên 5 beams cho đồng bộ và chất lượng tối đa
    no_repeat_ngram_size=4,     # Triệt tiêu lặp từ
    repetition_penalty=1.3,     # Phạt lặp từ chuẩn
    length_penalty=1.1,
    min_new_tokens=60,
    max_new_tokens=350,
    early_stopping=True,
    do_sample=True,
    temperature=0.15,           # Ép AI tóm tắt chính xác theo tài liệu gốc
    top_p=0.95,
)

# mT5 — multilingual T5, dùng sampling để tránh beam instability
MT5_GENERATION = GenerationProfile(
    num_beams=1,                # Beam search gây lặp từ với mT5 → dùng greedy + sampling
    no_repeat_ngram_size=4,
    repetition_penalty=1.8,
    length_penalty=1.0,
    min_new_tokens=80,
    max_new_tokens=400,
    early_stopping=False,       # Không dùng early_stopping khi do_sample=True
    do_sample=True,
    temperature=0.2,            # Thấp → tập trung, ít hallucination
    top_p=0.90,                 # Nucleus sampling — giữ 90% xác suất top tokens
)

# Mapping key → profile
GENERATION_PROFILES: dict[str, GenerationProfile] = {
    "bartpho": BARTPHO_GENERATION,
    "vit5": VIT5_GENERATION,
    "mt5": MT5_GENERATION,
}


def resolve_generation_profile(model_key: str) -> GenerationProfile:
    """
    Trả về GenerationProfile với override tùy chọn từ env.
    Mặc định giữ nguyên profile cứng — không ảnh hưởng chất lượng khi env trống.
    """
    base = GENERATION_PROFILES[model_key]
    overrides: dict[str, Any] = {}
    if RAG_GEN_NUM_BEAMS is not None:
        overrides["num_beams"] = max(1, RAG_GEN_NUM_BEAMS)
    if RAG_GEN_LENGTH_PENALTY is not None:
        overrides["length_penalty"] = RAG_GEN_LENGTH_PENALTY
    if RAG_GEN_REPETITION_PENALTY is not None:
        overrides["repetition_penalty"] = RAG_GEN_REPETITION_PENALTY
    if not overrides:
        return base
    return replace(base, **overrides)

# Mô hình ưu tiên cho tóm tắt trong RAG pipeline
# BARTPho được ưu tiên vì ổn định nhất với tiếng Việt
PREFERRED_SUMMARIZER_ORDER: list[str] = ["bartpho", "vit5", "mt5"]


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATE — Tiếng Việt chuẩn
# ─────────────────────────────────────────────────────────────────────────────
SUMMARIZE_PROMPT_TEMPLATE: str = """\
Bạn là một chuyên gia phân tích tài liệu cao cấp.
Dựa trên các đoạn ngữ cảnh được cung cấp dưới đây, hãy viết một bản tóm tắt mạch lạc, \
chính xác tuyệt đối, bao gồm các ý chính cốt lõi nhất.
Không bịa đặt thông tin nằm ngoài ngữ cảnh.

Ngữ cảnh:
{context}

Bản tóm tắt tiếng Việt chuẩn xác:"""

MULTI_DOC_QA_HINT: str = """\
LƯU Ý ĐA TÀI LIỆU: Người dùng đã chọn {doc_count} tài liệu: {filenames}.
- Bạn PHẢI trả lời dựa trên TẤT CẢ {doc_count} tài liệu có trong ngữ cảnh.

BẮT BUỘC định dạng trả lời:
### Tài liệu: [tên file 1]
(nội dung từ file 1)
### Tài liệu: [tên file 2]
...
- PHẢI có đúng {doc_count} mục, mỗi file một mục
- Nếu file không liên quan câu hỏi, ghi: "Không có thông tin liên quan trong [tên file]"
- KHÔNG gộp nội dung nhiều file vào một đoạn chung"""

QA_PROMPT_TEMPLATE: str = """\
Bạn là trợ lý phân tích tài liệu chuyên nghiệp. Chỉ trả lời dựa trên NGỮ CẢNH bên dưới.
Hãy tham khảo LỊCH SỬ HỘI THOẠI để hiểu ngữ cảnh các câu hỏi tiếp theo của người dùng (nếu có).
Nếu không tìm thấy thông tin, hãy trả lời: "Không tìm thấy thông tin trong tài liệu."
Trả lời bằng tiếng Việt, súc tích và chính xác.
{multi_doc_hint}

NGỮ CẢNH:
{context}

LỊCH SỬ HỘI THOẠI:
{chat_history}

CÂU HỎI HIỆN TẠI: {question}

TRẢ LỜI:"""

COMPRESSED_QA_PROMPT_TEMPLATE: str = """\
Bạn là trợ lý phân tích tài liệu chuyên nghiệp. Chỉ trả lời dựa trên TÀI LIỆU bên dưới.
Hãy tham khảo LỊCH SỬ HỘI THOẠI để hiểu ngữ cảnh các câu hỏi tiếp theo (nếu có).
Nếu không tìm thấy thông tin, hãy trả lời: "Không tìm thấy thông tin trong tài liệu."
Trả lời bằng tiếng Việt, súc tích và chính xác.
{multi_doc_hint}

═══ DOCUMENT SUMMARY ═══
{document_summary}

═══ IMPORTANT ORIGINAL PASSAGES ═══
{original_passages}

═══ INSTRUCTIONS ═══
- Trả lời CHỈ dựa trên tài liệu trên.
- Khi có mâu thuẫn giữa tóm tắt và đoạn gốc, ƯU TIÊN IMPORTANT ORIGINAL PASSAGES.
- Bảo toàn số liệu, tên riêng, công thức, trích dẫn và mốc thời gian từ đoạn gốc.
- Không bịa đặt thông tin ngoài tài liệu.

LỊCH SỬ HỘI THOẠI:
{chat_history}

CÂU HỎI HIỆN TẠI: {question}

TRẢ LỜI:"""

ADAPTIVE_QA_PROMPT_TEMPLATE: str = """\
Bạn là trợ lý phân tích tài liệu chuyên nghiệp. Chỉ trả lời dựa trên TÀI LIỆU bên dưới.
Hãy tham khảo LỊCH SỬ HỘI THOẠI để hiểu ngữ cảnh các câu hỏi tiếp theo (nếu có).
Nếu không tìm thấy thông tin, hãy trả lời: "Không tìm thấy thông tin trong tài liệu."
Trả lời bằng tiếng Việt, súc tích và chính xác.
{multi_doc_hint}

═══ QUERY FOCUS ═══
{query_focus}

═══ DOCUMENT SUMMARY (query-aware) ═══
{document_summary}

═══ VERIFIED ORIGINAL PASSAGES ═══
{original_passages}

═══ RULES ═══
- Trả lời CHỈ dựa trên tài liệu trên.
- Khi có mâu thuẫn giữa tóm tắt và đoạn gốc, ƯU TIÊN VERIFIED ORIGINAL PASSAGES.
- Bảo toàn số liệu, tên riêng, công thức, trích dẫn và mốc thời gian từ đoạn gốc.
- Không bịa đặt thông tin ngoài tài liệu.
- Trích dẫn nguồn [chunk:ID, p.X] khi có trong summary.

LỊCH SỬ HỘI THOẠI:
{chat_history}

CÂU HỎI HIỆN TẠI: {question}

TRẢ LỜI:"""
