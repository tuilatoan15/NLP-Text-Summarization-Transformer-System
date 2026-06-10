"""Shared utilities: logging, file I/O, text statistics, GPU diagnostics."""

from __future__ import annotations

import json
import logging
import sys
import time
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from src import config

# Global reentrant lock to prevent PyTorch/Transformers race conditions during model loading across threads
MODEL_LOAD_LOCK = threading.RLock()


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


def setup_logger(name: str = "nlp_summarizer", level: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, (level or config.LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOG_DIR / f"summarizer_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


logger = setup_logger()


def get_device_info() -> dict:
    info: dict[str, Any] = {"device": "cpu", "cuda_available": False}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            total_mb = props.total_memory / 1024 ** 2
            reserved_mb = torch.cuda.memory_reserved(idx) / 1024 ** 2
            allocated_mb = torch.cuda.memory_allocated(idx) / 1024 ** 2
            info.update(
                device=f"cuda:{idx}",
                gpu_name=props.name,
                total_vram_mb=round(total_mb, 1),
                reserved_vram_mb=round(reserved_mb, 1),
                allocated_vram_mb=round(allocated_mb, 1),
                free_vram_mb=round(total_mb - reserved_mb, 1),
                torch_version=torch.__version__,
            )
    except Exception as exc:
        info["error"] = str(exc)
    return info


def log_device_info() -> None:
    try:
        import torch
    except ImportError:
        logger.info("⚠️  PyTorch not importable")
        return

    if not torch.cuda.is_available():
        logger.info("⚠️  No GPU found — running on CPU (inference will be slower)")
        logger.info("    torch=%s  cuda_build=%s", torch.__version__, torch.version.cuda or "None (CPU-only build)")
        return

    try:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
    except Exception as exc:
        logger.warning("CUDA diagnostics failed; continuing without GPU startup details: %s", exc)
        return

    total_mb = props.total_memory / 1024 ** 2
    free_mb = (props.total_memory - torch.cuda.memory_reserved(idx)) / 1024 ** 2
    fp16_ok = props.major >= 7

    logger.info("=" * 56)
    logger.info("  🚀 GPU DETECTED — Running on CUDA")
    logger.info("  GPU Name    : %s", props.name)
    logger.info("  VRAM Total  : %.0f MB  (%.1f GB)", total_mb, total_mb / 1024)
    logger.info("  VRAM Free   : %.0f MB", free_mb)
    logger.info("  Compute Cap : %d.%d  (sm_%d%d)", props.major, props.minor, props.major, props.minor)
    logger.info("  CUDA Build  : %s  (Driver CUDA: %s)", torch.version.cuda, torch.version.cuda)
    logger.info("  fp16 OK     : %s  (requires compute >= 7.0)", fp16_ok)
    logger.info("  torch ver   : %s", torch.__version__)
    logger.info("=" * 56)


def log_vram_usage(tag: str = "") -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            return
        idx = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(idx) / 1024 ** 2
        reserved = torch.cuda.memory_reserved(idx) / 1024 ** 2
        logger.debug(
            "VRAM [%s]: allocated=%.0f MB  reserved=%.0f MB",
            tag or "checkpoint",
            alloc,
            reserved,
        )
    except Exception:
        pass


def clear_gpu_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


@contextmanager
def timer(label: str = "operation") -> Generator[None, None, None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("⏱  %s completed in %.3f s", label, elapsed)


def ensure_dir(path: str | Path) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def read_text_file(filepath: str | Path, encoding: str = "utf-8") -> str:
    return Path(filepath).read_text(encoding=encoding)


def write_text_file(filepath: str | Path, content: str, encoding: str = "utf-8") -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)
    logger.info("Wrote text file: %s", path)


def load_json(filepath: str | Path) -> dict:
    return json.loads(Path(filepath).read_text(encoding="utf-8"))


def save_json(data: Any, filepath: str | Path, indent: int = 2) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")
    logger.info("Saved JSON: %s", path)


def count_words(text: str) -> int:
    return len(text.split()) if text else 0


def count_sentences(text: str) -> int:
    from src.preprocess import split_sentences
    return len(split_sentences(text))


def truncate_text(text: str, max_words: int = 512) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    logger.warning("Truncated text from %s to %s words", len(words), max_words)
    return " ".join(words[:max_words])


def compression_ratio(summary: str, source: str) -> float:
    return round(count_words(summary) / max(1, count_words(source)), 4)


def format_scores(scores: dict) -> str:
    rows = ["ROUGE scores"]
    rows.extend(f"{key}: {float(value):.4f}" for key, value in scores.items())
    return "\n".join(rows)


def clear_cache() -> int:
    count = 0
    cache_path = config.CACHE_DIR
    if not cache_path.exists():
        return count
    for path in cache_path.rglob("*"):
        if path.is_file():
            path.unlink()
            count += 1
    logger.info("Cleared %s cache files", count)
    return count
