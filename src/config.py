"""Central configuration for the research summarization system."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is in requirements.txt
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")
if os.getenv("CUDA_VISIBLE_DEVICES") == "":
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
MODEL_DIR = PROJECT_ROOT / "models"
STORAGE_DIR = PROJECT_ROOT / "storage"
RESULTS_DIR = STORAGE_DIR / "results"
UPLOAD_DIR = STORAGE_DIR / "uploads"
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "configs"

for _path in (DATA_DIR, CACHE_DIR, MODEL_DIR, RESULTS_DIR, UPLOAD_DIR, LOG_DIR, CONFIG_DIR):
    _path.mkdir(parents=True, exist_ok=True)


# ─────────────────────────── API ───────────────────────────
API_TITLE = "Vietnamese Summarization Research API"
API_VERSION = "3.2.0"
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ─────────────────────────── DATASET ───────────────────────
DATASET_NAME = os.getenv("DATASET_NAME", "nam194/vietnews")
DATASET_CACHE_DIR = DATA_DIR / "cache"
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "5000"))
VALIDATION_RATIO = float(os.getenv("VALIDATION_RATIO", "0.1"))

# ─────────────────────────── MODEL ─────────────────────────
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL", "VietAI/vit5-base")
LOCAL_VIT5_DIR = Path(os.getenv("LOCAL_VIT5_DIR", str(MODEL_DIR / "vit5-finetuned")))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "1024"))  # raised for BARTPho syllable tokenizer
MAX_TARGET_TOKENS = int(os.getenv("MAX_TARGET_TOKENS", "256"))
MAX_OUTPUT_LENGTH = int(os.getenv("MAX_OUTPUT_LENGTH", "200"))
MIN_OUTPUT_LENGTH = int(os.getenv("MIN_OUTPUT_LENGTH", "20"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "4"))
NO_REPEAT_NGRAM_SIZE = int(os.getenv("NO_REPEAT_NGRAM_SIZE", "3"))
# Parallel chunk inference workers (abstractive long-text processing)
ABSTRACTIVE_CHUNK_WORKERS = int(os.getenv("ABSTRACTIVE_CHUNK_WORKERS", "2"))
# Max chunks per document (None = unlimited, bounded by memory)
ABSTRACTIVE_MAX_CHUNKS = int(os.getenv("ABSTRACTIVE_MAX_CHUNKS", "16"))

# ─────────────────────────── PER-MODEL GENERATION CONFIGS ──
# Tuned individually to maximise output quality per architecture.
# Keys must match ABSTRACTIVE_ALGORITHMS keys in model_registry.py
GENERATION_CONFIGS: dict[str, dict] = {
    # ViT5: T5 tiếng Việt fine-tuned.
    # Su dung repetition_penalty=1.2 va no_repeat_ngram_size=3 de dam bao cau logic tu nhien.
    "vit5": dict(
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.35,
        length_penalty=1.05,
        early_stopping=True,
        do_sample=False,
    ),
    # mT5: da ngon ngu. Ha repetition_penalty de tranh lam meo mo tu vung sinh ra rac.
    "mt5": dict(
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        length_penalty=1.0,
        early_stopping=True,
        do_sample=False,
    ),
    # BARTPho: BART syllable-level. Su dung repetition_penalty va length_penalty hop ly.
    "bartpho": dict(
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.35,
        length_penalty=1.15,
        early_stopping=True,
        do_sample=False,
        forced_bos_token_id=None,
    ),
}

# Default fallback generation config (used if a key is missing above)
DEFAULT_GENERATION_CONFIG: dict = dict(
    max_new_tokens=MAX_OUTPUT_LENGTH,
    min_new_tokens=MIN_OUTPUT_LENGTH,
    num_beams=NUM_BEAMS,
    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
    repetition_penalty=1.15,
    length_penalty=1.0,
    early_stopping=True,
    do_sample=False,
)

# ─────────────────────────── mT5 EXPERIMENTAL ──────────────
# mT5 uses a multilingual tokenizer; vocab mismatch after resize can produce
# non-Vietnamese artifacts. If ratio of ASCII/latin chars exceeds this threshold
# in the generated output, the model is flagged as "experimental" in the response.
MT5_EXPERIMENTAL = os.getenv("MT5_EXPERIMENTAL", "1") == "1"
MT5_LATIN_RATIO_THRESHOLD = float(os.getenv("MT5_LATIN_RATIO_THRESHOLD", "0.35"))

# ─────────────────────────────────── GPU / DEVICE ──────────────────
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "1") == "1"
GPU_VRAM_LIMIT_GB = float(os.getenv("GPU_VRAM_LIMIT_GB", "4.0"))
USE_FP16 = os.getenv("USE_FP16", "auto")   # "auto" | "1" | "0"
USE_8BIT = os.getenv("USE_8BIT", "0") == "1"
USE_4BIT = os.getenv("USE_4BIT", "0") == "1"
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
# Tăng workers cho extractive (chúng nhẹ, chạy song song hoàn toàn được)
EXTRACTIVE_WORKERS = int(os.getenv("EXTRACTIVE_WORKERS", "4"))
MAX_GPU_CONCURRENT = int(os.getenv("MAX_GPU_CONCURRENT", "1"))

# ─────────────────────────── EVALUATION FAIRNESS ───────────
# When True, ROUGE/BLEU are computed even when source text is used as reference.
# Setting to False disables overlap metrics in that case (recommended for research).
ALLOW_SOURCE_AS_REFERENCE = os.getenv("ALLOW_SOURCE_AS_REFERENCE", "0") == "1"

# When no real reference is available, only these metrics are used for ranking.
NO_REFERENCE_RANKING_WEIGHTS: dict[str, float] = {
    "bertscore_f1": 0.45,
    "semantic_similarity": 0.35,
    "compression_score": 0.20,
}

# Weights used when a REAL reference summary is available.
WITH_REFERENCE_RANKING_WEIGHTS: dict[str, float] = {
    "rougeL": 0.25,
    "rouge2": 0.15,
    "bertscore_f1": 0.30,
    "semantic_similarity": 0.20,
    "compression_score": 0.10,
}

# ─────────────────────────── COMPOSITE SCORE ───────────────
# Trọng số điểm tổng hợp (Final Composite Score) cho bảng xếp hạng.
# Điều chỉnh theo kết quả nghiên cứu để cân bằng giữa overlap metrics
# và semantic quality metrics.
COMPOSITE_SCORE_WEIGHTS: dict[str, float] = {
    "rougeL": float(os.getenv("COMPOSITE_W_ROUGEL", "0.25")),
    "bertscore": float(os.getenv("COMPOSITE_W_BERTSCORE", "0.25")),
    "semantic_similarity": float(os.getenv("COMPOSITE_W_SEMANTIC", "0.20")),
    "faithfulness": float(os.getenv("COMPOSITE_W_FAITHFULNESS", "0.15")),
    "coverage": float(os.getenv("COMPOSITE_W_COVERAGE", "0.10")),
    "fluency": float(os.getenv("COMPOSITE_W_FLUENCY", "0.05")),
}

# ─────────────────────────── BENCHMARK CATEGORIES ──────────
BENCHMARK_DOCUMENT_CATEGORIES: dict[str, tuple[int, int]] = {
    "Short": (100, 500),
    "Medium": (500, 2000),
    "Long": (2000, 10000),
    "Very Long": (10000, 100000),
}

# ─────────────────────────── METRICS ───────────────────────
# xlm-roberta-base cho BERTScore: tốt hơn bert-base-multilingual-cased
# cho ngôn ngữ có dấu như tiếng Việt (tokenizer xử lý tốt syllable/diacritics).
BERTSCORE_MODEL = os.getenv("BERTSCORE_MODEL", "xlm-roberta-base")
BERTSCORE_LANG = os.getenv("BERTSCORE_LANG", "vi")
SBERT_MODEL = os.getenv("SBERT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
HEAVY_METRICS_TIMEOUT = float(os.getenv("HEAVY_METRICS_TIMEOUT", "30.0"))

# ─────────────────────────── TRAINING ──────────────────────
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "2"))
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "4"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-5"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "100"))

# ─────────────────────────── INFRA ─────────────────────────
DOCUMENT_INTELLIGENCE_DIR = Path(
    os.getenv("DOCUMENT_INTELLIGENCE_DIR", str(STORAGE_DIR / "document_intelligence"))
)
DOCUMENT_INTELLIGENCE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", "")
REDIS_URL = os.getenv("REDIS_URL", "")
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "qdrant").lower() # Mặc định chuyển sang Qdrant
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "agentic")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "agentic-secret")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "documents")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "BAAI/bge-m3")
ENABLE_DB_PERSISTENCE = os.getenv("ENABLE_DB_PERSISTENCE", "0") == "1"

# ─────────────────────────── LOGGING ───────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
