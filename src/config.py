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
API_VERSION = "3.1.0"
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

# ─────────────────────────── GPU / DEVICE ──────────────────
# Preload all models on startup to avoid per-request cold-start latency
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "1") == "1"

# Maximum VRAM (GB) allowed before switching to sequential GPU scheduling
# Set lower on machines with < 8 GB VRAM
GPU_VRAM_LIMIT_GB = float(os.getenv("GPU_VRAM_LIMIT_GB", "4.0"))

# Whether to use fp16 inference (requires GPU)
USE_FP16 = os.getenv("USE_FP16", "auto")  # "auto" | "1" | "0"

# torch.compile — enabled on PyTorch >= 2 for ~20-40% speedup after warm-up
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"

# Number of worker threads for extractive parallel inference
EXTRACTIVE_WORKERS = int(os.getenv("EXTRACTIVE_WORKERS", "3"))

# Maximum concurrent Transformer models on GPU (to avoid OOM)
MAX_GPU_CONCURRENT = int(os.getenv("MAX_GPU_CONCURRENT", "1"))

# ─────────────────────────── METRICS ───────────────────────
BERTSCORE_MODEL = os.getenv("BERTSCORE_MODEL", "bert-base-multilingual-cased")
BERTSCORE_LANG = os.getenv("BERTSCORE_LANG", "vi")
SBERT_MODEL = os.getenv("SBERT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Run heavy metrics (BERTScore, Semantic Sim) synchronously or skip if too slow
HEAVY_METRICS_TIMEOUT = float(os.getenv("HEAVY_METRICS_TIMEOUT", "30.0"))

# ─────────────────────────── TRAINING ──────────────────────
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "2"))
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "4"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-5"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "100"))

# ─────────────────────────── LOGGING ───────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
