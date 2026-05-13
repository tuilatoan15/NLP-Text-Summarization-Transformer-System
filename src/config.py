"""
src/config.py — Cấu hình tập trung cho toàn bộ hệ thống.

Đọc từ biến môi trường (environment variables) hoặc dùng giá trị mặc định.
Tất cả các module khác nên import từ đây thay vì hardcode.
"""
import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
MODEL_DIR      = PROJECT_ROOT / "models" / "vit5-finetuned"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
STORAGE_DIR    = PROJECT_ROOT / "storage" / "results"
UPLOAD_DIR     = PROJECT_ROOT / "storage" / "uploads"
CACHE_DIR      = PROJECT_ROOT / "cache"
LOG_DIR        = PROJECT_ROOT / "logs"
DATA_DIR       = PROJECT_ROOT / "data"

# ─── Model ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME   = os.getenv("DEFAULT_MODEL", "VietAI/vit5-base")
MAX_INPUT_TOKENS     = int(os.getenv("MAX_INPUT_TOKENS", "512"))
MAX_OUTPUT_LENGTH    = int(os.getenv("MAX_OUTPUT_LENGTH", "150"))
MIN_OUTPUT_LENGTH    = int(os.getenv("MIN_OUTPUT_LENGTH", "20"))
NUM_BEAMS            = int(os.getenv("NUM_BEAMS", "2"))

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST  = os.getenv("API_HOST", "0.0.0.0")
API_PORT  = int(os.getenv("API_PORT", "8000"))
API_TITLE = "🇻🇳 Vietnamese Text Summarization API"
API_VERSION = "2.0.0"

# ─── MongoDB (tùy chọn) ───────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "")  # Để trống nếu không dùng MongoDB
MONGO_DB  = os.getenv("MONGO_DB", "nlp_summarization")

# ─── Training ─────────────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE  = int(os.getenv("TRAIN_BATCH_SIZE", "2"))
EVAL_BATCH_SIZE   = int(os.getenv("EVAL_BATCH_SIZE", "4"))
LEARNING_RATE     = float(os.getenv("LEARNING_RATE", "5e-5"))
NUM_EPOCHS        = int(os.getenv("NUM_EPOCHS", "3"))
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "5000"))
DATASET_NAME      = os.getenv("DATASET_NAME", "thanhnew2001/vnexpress")

# ─── Evaluation ───────────────────────────────────────────────────────────────
BERTSCORE_MODEL = os.getenv("BERTSCORE_MODEL", "bert-base-multilingual-cased")
BERTSCORE_LANG  = os.getenv("BERTSCORE_LANG", "vi")
SBERT_MODEL     = os.getenv("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
