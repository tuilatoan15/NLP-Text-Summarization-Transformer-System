"""
utils.py — Các hàm tiện ích dùng chung trong toàn hệ thống.
Bao gồm: logging, đọc/ghi file, định dạng output.
"""

import logging
import os
import json
from datetime import datetime
from pathlib import Path


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logger(name: str = "nlp_summarizer", level: int = logging.INFO) -> logging.Logger:
    """
    Tạo và cấu hình logger có định dạng rõ ràng cho toàn hệ thống.
    
    Args:
        name: Tên logger (mặc định: 'nlp_summarizer')
        level: Mức độ log (mặc định: INFO)
    
    Returns:
        Logger đã được cấu hình
    """
    logger = logging.getLogger(name)
    
    # Tránh thêm handler trùng lặp khi gọi lại hàm
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Định dạng log: thời gian | tên | cấp độ | nội dung
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler ra console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler ra file log (lưu vào thư mục logs/)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"summarizer_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Tạo logger mặc định để import ở các module khác
logger = setup_logger()


# ==============================================================================
# FILE I/O
# ==============================================================================

def read_text_file(filepath: str, encoding: str = "utf-8") -> str:
    """Đọc nội dung một file văn bản."""
    with open(filepath, "r", encoding=encoding) as f:
        return f.read()


def write_text_file(filepath: str, content: str, encoding: str = "utf-8") -> None:
    """Ghi nội dung ra file văn bản, tạo thư mục nếu chưa có."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding=encoding) as f:
        f.write(content)
    logger.info(f"Đã ghi file: {filepath}")


def load_json(filepath: str) -> dict:
    """Đọc file JSON và trả về dict."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, filepath: str, indent: int = 2) -> None:
    """Lưu dict ra file JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    logger.info(f"Đã lưu JSON: {filepath}")


# ==============================================================================
# TEXT UTILITIES
# ==============================================================================

def count_words(text: str) -> int:
    """Đếm số từ trong đoạn văn (tách bằng khoảng trắng)."""
    return len(text.split()) if text else 0


def count_sentences(text: str) -> int:
    """Đếm sơ bộ số câu dựa trên dấu câu kết thúc (.!?)."""
    import re
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def truncate_text(text: str, max_words: int = 512) -> str:
    """
    Cắt ngắn văn bản theo số từ tối đa.
    Dùng để đảm bảo đầu vào không vượt quá giới hạn token của model.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    logger.warning(f"Văn bản đã bị cắt ngắn từ {len(words)} xuống {max_words} từ.")
    return truncated


def format_scores(scores: dict) -> str:
    """Định dạng dict điểm ROUGE để hiển thị ra console rõ ràng."""
    lines = ["=" * 40, "📊 ROUGE SCORES:", "=" * 40]
    for key, value in scores.items():
        lines.append(f"  {key:12s}: {value:.4f}")
    lines.append("=" * 40)
    return "\n".join(lines)


def ensure_dir(path: str) -> str:
    """Tạo thư mục nếu chưa tồn tại, trả về path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
