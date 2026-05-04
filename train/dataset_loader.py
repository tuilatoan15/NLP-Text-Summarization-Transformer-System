"""
dataset_loader.py — Tải và chuẩn bị dataset huấn luyện cho ViT5.

Hỗ trợ 2 nguồn dữ liệu:
  1. File CSV/JSON nội bộ (nếu user chuẩn bị sẵn dữ liệu)
  2. Dataset từ Hugging Face Hub (VnExpress hoặc các dataset tiếng Việt có sẵn)

Format chuẩn:
  - Cột đầu vào: 'article' (văn bản đầy đủ)
  - Cột đầu ra : 'title'   (tiêu đề/tóm tắt mẫu)
"""

import os
from pathlib import Path
from typing import Optional

from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
    load_from_disk,
)

from src.utils import logger


# ==============================================================================
# CẤU HÌNH MẶC ĐỊNH
# ==============================================================================

# Số lượng sample tối đa để tránh dùng quá nhiều RAM/thời gian trên máy yếu
DEFAULT_MAX_SAMPLES = 5000
DEFAULT_TEST_SPLIT  = 0.1       # 10% dùng cho validation

# Thư mục lưu dataset đã xử lý để tái sử dụng
CACHE_DIR = Path("./data/cache")


# ==============================================================================
# TẢI TỪ FILE CSV / JSON NỘI BỘ
# ==============================================================================

def load_from_csv(
    filepath: str,
    article_col: str = "article",
    title_col:   str = "title",
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> DatasetDict:
    """
    Tải dataset từ file CSV nội bộ.

    File CSV cần có ít nhất 2 cột:
      - article_col: văn bản đầy đủ (input)
      - title_col:   tiêu đề hoặc tóm tắt ngắn (target)

    Args:
        filepath:    Đường dẫn file CSV
        article_col: Tên cột chứa văn bản
        title_col:   Tên cột chứa tiêu đề/tóm tắt
        max_samples: Số sample tối đa dùng để train

    Returns:
        DatasetDict với keys 'train' và 'validation'
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    logger.info(f"Đang tải dataset từ file: {filepath}")

    ext = Path(filepath).suffix.lower()
    if ext == ".csv":
        raw_ds = load_dataset("csv", data_files=filepath, split="train")
    elif ext in (".json", ".jsonl"):
        raw_ds = load_dataset("json", data_files=filepath, split="train")
    else:
        raise ValueError(f"Định dạng file không hỗ trợ: {ext}. Dùng .csv hoặc .json")

    # Đổi tên cột về chuẩn 'article' và 'title' nếu cần
    if article_col != "article" and article_col in raw_ds.column_names:
        raw_ds = raw_ds.rename_column(article_col, "article")
    if title_col != "title" and title_col in raw_ds.column_names:
        raw_ds = raw_ds.rename_column(title_col, "title")

    # Kiểm tra cột tồn tại
    for col in ["article", "title"]:
        if col not in raw_ds.column_names:
            raise ValueError(
                f"Không tìm thấy cột '{col}' trong dataset. "
                f"Các cột hiện có: {raw_ds.column_names}"
            )

    # Chỉ giữ 2 cột cần thiết và lọc dòng rỗng
    raw_ds = raw_ds.select_columns(["article", "title"])
    raw_ds = raw_ds.filter(
        lambda x: x["article"] and x["title"] and
                  len(x["article"].split()) > 10 and
                  len(x["title"].split()) > 0
    )

    # Giới hạn số sample
    if len(raw_ds) > max_samples:
        raw_ds = raw_ds.select(range(max_samples))
        logger.info(f"Đã giới hạn dataset: {max_samples} samples.")

    # Chia train/validation
    split = raw_ds.train_test_split(test_size=DEFAULT_TEST_SPLIT, seed=42)
    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    logger.info(
        f"Dataset từ file: train={len(dataset['train'])}, "
        f"validation={len(dataset['validation'])}"
    )
    return dataset


# ==============================================================================
# TẢI TỪ HUGGING FACE HUB
# ==============================================================================

def load_from_huggingface(
    dataset_name: str = "vietgpt/binhvq-news-vi",
    article_col: str = "text",
    title_col:   str = "title",
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> DatasetDict:
    """
    Tải dataset tiếng Việt từ Hugging Face Hub.

    Dataset mặc định: vietgpt/binhvq-news-vi (tin tức tiếng Việt)
    Bạn có thể thay thế bằng bất kỳ dataset nào có cấu trúc article/title.

    Args:
        dataset_name: Tên dataset trên Hugging Face Hub
        article_col:  Tên cột chứa văn bản gốc
        title_col:    Tên cột chứa tiêu đề/tóm tắt
        max_samples:  Số sample tối đa

    Returns:
        DatasetDict với keys 'train' và 'validation'
    """
    logger.info(f"Đang tải dataset từ Hugging Face: {dataset_name}")

    try:
        raw_ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
    except Exception as e:
        logger.error(f"Lỗi tải dataset {dataset_name}: {e}")
        raise

    logger.info(f"Dataset gốc: {len(raw_ds)} samples, cột: {raw_ds.column_names}")

    # Kiểm tra và chuẩn hóa tên cột
    if article_col not in raw_ds.column_names or title_col not in raw_ds.column_names:
        logger.warning(
            f"Không tìm thấy cột '{article_col}' hoặc '{title_col}'. "
            f"Các cột hiện có: {raw_ds.column_names}"
        )
        # Thử detect tự động
        text_candidates = ["text", "article", "content", "body"]
        title_candidates = ["title", "headline", "summary", "label"]

        for col in text_candidates:
            if col in raw_ds.column_names:
                article_col = col
                break

        for col in title_candidates:
            if col in raw_ds.column_names:
                title_col = col
                break

        logger.info(f"Auto-detect cột: article='{article_col}', title='{title_col}'")

    # Đổi tên cột về chuẩn
    rename_map = {}
    if article_col != "article":
        rename_map[article_col] = "article"
    if title_col != "title":
        rename_map[title_col] = "title"
    if rename_map:
        raw_ds = raw_ds.rename_columns(rename_map)

    # Chỉ giữ 2 cột cần thiết
    raw_ds = raw_ds.select_columns(["article", "title"])

    # Lọc mẫu rỗng hoặc quá ngắn
    raw_ds = raw_ds.filter(
        lambda x: x["article"] and x["title"] and
                  len(str(x["article"]).split()) > 20 and
                  len(str(x["title"]).split()) >= 3
    )

    # Giới hạn số sample
    if len(raw_ds) > max_samples:
        raw_ds = raw_ds.select(range(max_samples))
        logger.info(f"Đã giới hạn: {max_samples} samples.")

    # Chia train/validation
    split = raw_ds.train_test_split(test_size=DEFAULT_TEST_SPLIT, seed=42)
    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    logger.info(
        f"Dataset HF: train={len(dataset['train'])}, "
        f"validation={len(dataset['validation'])}"
    )
    return dataset


# ==============================================================================
# HÀM LOAD TỔNG HỢP (ƯU TIÊN FILE LOCAL)
# ==============================================================================

def load_vnexpress_dataset(
    local_csv_path: Optional[str] = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    use_cache: bool = True,
) -> DatasetDict:
    """
    Hàm tải dataset chính cho pipeline huấn luyện.

    Thứ tự ưu tiên:
      1. Cache đã xử lý từ lần chạy trước (nếu use_cache=True)
      2. File CSV/JSON nội bộ (nếu local_csv_path được cung cấp)
      3. Tải từ Hugging Face Hub (mặc định)

    Args:
        local_csv_path: Đường dẫn file CSV nội bộ (None = dùng HF Hub)
        max_samples:    Số sample tối đa
        use_cache:      Dùng cache từ lần chạy trước

    Returns:
        DatasetDict với 'train' và 'validation'
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"dataset_{max_samples}"

    # 1. Thử load từ cache
    if use_cache and cache_path.exists():
        logger.info(f"Đang load dataset từ cache: {cache_path}")
        try:
            dataset = load_from_disk(str(cache_path))
            logger.info(
                f"✅ Load cache thành công: "
                f"train={len(dataset['train'])}, "
                f"validation={len(dataset['validation'])}"
            )
            return dataset
        except Exception as e:
            logger.warning(f"Cache bị lỗi, bỏ qua: {e}")

    # 2. Load từ file CSV nội bộ
    if local_csv_path:
        dataset = load_from_csv(local_csv_path, max_samples=max_samples)
    else:
        # 3. Load từ Hugging Face Hub
        dataset = load_from_huggingface(max_samples=max_samples)

    # Lưu cache để lần sau load nhanh hơn
    if use_cache:
        dataset.save_to_disk(str(cache_path))
        logger.info(f"Đã lưu dataset cache: {cache_path}")

    return dataset


# ==============================================================================
# CHẠY THỬ TRỰC TIẾP
# ==============================================================================

if __name__ == "__main__":
    import sys

    # Thử load 100 sample để test nhanh
    ds = load_vnexpress_dataset(max_samples=100, use_cache=False)

    print("\n=== THÔNG TIN DATASET ===")
    print(f"Train:      {len(ds['train'])} samples")
    print(f"Validation: {len(ds['validation'])} samples")
    print(f"\nVí dụ mẫu đầu tiên:")
    print(f"  article: {ds['train'][0]['article'][:200]}...")
    print(f"  title:   {ds['train'][0]['title']}")
