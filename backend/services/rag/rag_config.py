"""
rag_config.py — Cấu hình RAG cứng (hardcode) theo chuẩn công nghiệp.

Toàn bộ tham số ở đây đã được tinh chỉnh tối ưu.
Người dùng KHÔNG cần và KHÔNG được phép thay đổi từ frontend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

# Load env variables (if not loaded yet)
load_dotenv()

RAG_GENERATOR_TYPE: str = os.getenv("RAG_GENERATOR_TYPE", "local").lower()
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_API_URL: str = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────
# VoVanPhuc/sup-SimCSE-VietNamese-phobert-base: mô hình chuyên tiếng Việt siêu nhẹ (~540MB), hiệu suất cao.
# Kích thước vector: 768 chiều
EMBEDDING_MODEL: str = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"

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

# Sau Cross-Encoder reranker, giữ lại top 4 chunks cô đọng, chất lượng nhất để đưa vào Prompt
RETRIEVAL_FINAL_TOP_K: int = 4

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

QA_PROMPT_TEMPLATE: str = """\
Bạn là trợ lý phân tích tài liệu chuyên nghiệp. Chỉ trả lời dựa trên NGỮ CẢNH bên dưới.
Hãy tham khảo LỊCH SỬ HỘI THOẠI để hiểu ngữ cảnh các câu hỏi tiếp theo của người dùng (nếu có).
Nếu không tìm thấy thông tin, hãy trả lời: "Không tìm thấy thông tin trong tài liệu."
Trả lời bằng tiếng Việt, súc tích và chính xác.

NGỮ CẢNH:
{context}

LỊCH SỬ HỘI THOẠI:
{chat_history}

CÂU HỎI HIỆN TẠI: {question}

TRẢ LỜI:"""
