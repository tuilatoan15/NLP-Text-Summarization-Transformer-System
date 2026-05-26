"""Central configuration for the research summarization system."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
DATASET_NAME = os.getenv("DATASET_NAME", "thanhnew2001/vnexpress")
DATASET_CACHE_DIR = DATA_DIR / "cache"
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "5000"))
VALIDATION_RATIO = float(os.getenv("VALIDATION_RATIO", "0.1"))

# ─────────────────────────── MODEL ─────────────────────────
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL", "VietAI/vit5-base")
LOCAL_VIT5_DIR = Path(os.getenv("LOCAL_VIT5_DIR", str(MODEL_DIR / "vit5-finetuned")))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "512"))
MAX_TARGET_TOKENS = int(os.getenv("MAX_TARGET_TOKENS", "128"))
MAX_OUTPUT_LENGTH = int(os.getenv("MAX_OUTPUT_LENGTH", "150"))
MIN_OUTPUT_LENGTH = int(os.getenv("MIN_OUTPUT_LENGTH", "20"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "4"))
NO_REPEAT_NGRAM_SIZE = int(os.getenv("NO_REPEAT_NGRAM_SIZE", "3"))

# ─────────────────────────── PER-MODEL GENERATION CONFIGS ──
# Tuned individually to maximise output quality per architecture.
# Keys must match ABSTRACTIVE_ALGORITHMS keys in model_registry.py
GENERATION_CONFIGS: dict[str, dict] = {
    # ViT5: fine-tuned Vietnamese T5 — prone to repetition loops, use aggressive
    # deduplication (ngram_size=4, penalty=2.0) with conservative beam count.
    "vit5": dict(
        max_new_tokens=80,
        min_new_tokens=15,
        num_beams=2,
        no_repeat_ngram_size=4,
        repetition_penalty=2.0,
        length_penalty=1.0,
        early_stopping=True,
        do_sample=False,
    ),
    # mT5: multilingual T5 — vocab mismatch makes beam search unstable;
    # sampling is more robust for this checkpoint.
    "mt5": dict(
        max_new_tokens=80,
        min_new_tokens=10,
        num_beams=2,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
        length_penalty=1.0,
        early_stopping=True,
        do_sample=False,
    ),
    # BARTPho: syllable-level Vietnamese BART — most stable, allow longer output.
    "bartpho": dict(
        max_new_tokens=120,
        min_new_tokens=20,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        length_penalty=1.0,
        early_stopping=True,
        do_sample=False,
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

# ─────────────────────────── GPU / DEVICE ──────────────────
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "1") == "1"
GPU_VRAM_LIMIT_GB = float(os.getenv("GPU_VRAM_LIMIT_GB", "4.0"))
USE_FP16 = os.getenv("USE_FP16", "auto")   # "auto" | "1" | "0"
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
EXTRACTIVE_WORKERS = int(os.getenv("EXTRACTIVE_WORKERS", "3"))
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

# ─────────────────────────── METRICS ───────────────────────
BERTSCORE_MODEL = os.getenv("BERTSCORE_MODEL", "bert-base-multilingual-cased")
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
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "local").lower()
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "agentic")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "agentic-secret")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "documents")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "BAAI/bge-m3")
ENABLE_DB_PERSISTENCE = os.getenv("ENABLE_DB_PERSISTENCE", "0") == "1"

# ─────────────────────────── LOGGING ───────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

