"""
train_vit5.py — Fine-tune mô hình VietAI/vit5-base cho bài toán tóm tắt tiếng Việt.

Pipeline huấn luyện:
  1. Tải dataset (từ file CSV hoặc Hugging Face Hub)
  2. Tokenize input/output với ViT5Tokenizer
  3. Cấu hình Seq2SeqTrainer với tham số nhỏ phù hợp máy yếu
  4. Huấn luyện với logging và checkpoint định kỳ
  5. Lưu model đã fine-tune vào ./models/vit5-finetuned/

Chạy: python -m train.train_vit5 [--local_data path/to/file.csv] [--max_samples 5000]
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

# Thêm project root vào Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate as hf_evaluate

from train.dataset_loader import load_vnexpress_dataset
from src.utils import logger, ensure_dir, save_json


# ==============================================================================
# CẤU HÌNH
# ==============================================================================

MODEL_NAME     = "VietAI/vit5-base"
OUTPUT_DIR     = "./models/vit5-finetuned"
LOG_DIR        = "./logs/train"
CHECKPOINT_DIR = "./models/checkpoints"

# Hyperparameters
MAX_INPUT_LENGTH  = 512   # Token tối đa cho input (article)
MAX_TARGET_LENGTH = 128   # Token tối đa cho output (title/summary)
TRAIN_BATCH_SIZE  = 2     # Nhỏ để chạy được trên máy yếu/CPU
EVAL_BATCH_SIZE   = 4
LEARNING_RATE     = 5e-5
NUM_EPOCHS        = 3
WARMUP_STEPS      = 100
WEIGHT_DECAY      = 0.01
SAVE_STEPS        = 500
EVAL_STEPS        = 500
LOGGING_STEPS     = 50
GRADIENT_ACCUM    = 4     # Tích lũy gradient để mô phỏng batch size lớn hơn


# ==============================================================================
# TOKENIZATION
# ==============================================================================

def tokenize_function(examples, tokenizer, max_input_len, max_target_len):
    """
    Tokenize batch dữ liệu cho Seq2Seq model.

    Prefix "summarize: " được thêm vào đầu mỗi input theo chuẩn T5/ViT5.
    Labels là token IDs của target (title), với -100 ở vị trí padding
    để loss function bỏ qua padding token.
    """
    # Thêm prefix cho input (chuẩn T5)
    inputs = ["summarize: " + str(article) for article in examples["article"]]
    targets = [str(title) for title in examples["title"]]

    # Tokenize input
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_len,
        truncation=True,
        padding=False,  # Padding sẽ được xử lý bởi DataCollator
    )

    # Tokenize target (labels)
    # Support both tokenizers that provide `as_target_tokenizer()` and ones that don't.
    try:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_len,
                truncation=True,
                padding=False,
            )
    except AttributeError:
        # Fallback: call tokenizer directly on targets
        labels = tokenizer(
            targets,
            max_length=max_target_len,
            truncation=True,
            padding=False,
        )

    # Thay padding token ID bằng -100 để loss không tính padding
    label_ids = []
    for label in labels["input_ids"]:
        label_ids.append(
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
        )

    model_inputs["labels"] = label_ids
    return model_inputs


# ==============================================================================
# COMPUTE METRICS
# ==============================================================================

def build_compute_metrics(tokenizer):
    """
    Tạo hàm tính ROUGE metric cho Trainer.
    
    Hàm được trả về sẽ decode token IDs thành chuỗi văn bản
    rồi tính ROUGE-1, ROUGE-2, ROUGE-L.
    """
    rouge_metric = hf_evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Xử lý trường hợp predictions là tuple (do beam search)
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # Decode labels (thay -100 về pad_token_id trước)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # Làm sạch: strip whitespace
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # Tính ROUGE
        result = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=False,
        )

        # Làm tròn
        result = {k: round(v * 100, 2) for k, v in result.items()}  # Đổi sang %
        return result

    return compute_metrics


# ==============================================================================
# PIPELINE HUẤN LUYỆN CHÍNH
# ==============================================================================

def train(
    local_data: str = None,
    dataset_name: str = "thanhnew2001/vnexpress",
    max_samples: int = 5000,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    output_dir: str = OUTPUT_DIR,
):
    """
    Pipeline fine-tune ViT5.

    Args:
        local_data:    Đường dẫn file CSV nội bộ (None = dùng HF Hub)
        max_samples:   Số sample tối đa để dùng
        num_epochs:    Số epoch huấn luyện
        batch_size:    Batch size (khuyến nghị 2-4 trên CPU)
        learning_rate: Learning rate
        output_dir:    Thư mục lưu model đã fine-tune
    """
    # --- Chuẩn bị thư mục ---
    ensure_dir(output_dir)
    ensure_dir(LOG_DIR)
    ensure_dir(CHECKPOINT_DIR)

    # --- Phát hiện thiết bị ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Thiết bị huấn luyện: {device.upper()}")
    if device == "cpu":
        logger.warning(
            "Đang chạy trên CPU. Quá trình huấn luyện sẽ chậm hơn GPU khoảng 10-50x. "
            "Khuyến nghị giảm max_samples và num_epochs."
        )

    # --- Bước 1: Tải tokenizer & model ---
    # Nếu có bản model local (đã tải/saved trước đó), ưu tiên load local để tránh vấn đề tokenizer HF
    local_model_dir = Path("./models/vit5-finetuned")
    if local_model_dir.exists() and any(local_model_dir.iterdir()):
        logger.info(f"Tìm thấy model local: {local_model_dir}. Load tokenizer và model từ local.")
        # use_fast=False để tránh các incompatibility với tokenizer.json
        tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir), use_fast=False)
        logger.info(f"Đang load model từ local: {local_model_dir}")
        model = AutoModelForSeq2SeqLM.from_pretrained(str(local_model_dir))
    else:
        logger.info(f"Đang load tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        logger.info(f"Đang load model: {MODEL_NAME}")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model có {model_params:.1f}M tham số.")

    # --- Bước 2: Tải dataset ---
    logger.info("Đang tải dataset...")
    dataset = load_vnexpress_dataset(
        local_csv_path=local_data,
        max_samples=max_samples,
        dataset_name=dataset_name,
    )
    logger.info(f"Train: {len(dataset['train'])} | Validation: {len(dataset['validation'])}")

    # --- Bước 3: Tokenize dataset ---
    logger.info("Đang tokenize dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(
            examples, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
        ),
        batched=True,
        batch_size=32,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    logger.info("Tokenize xong!")

    # --- Bước 4: Data Collator ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,  # Tối ưu cho mixed precision training
    )

    # --- Bước 5: Cấu hình Training Arguments ---
    # fp16 chỉ dùng được trên GPU CUDA
    use_fp16 = (device == "cuda")

    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM,
        learning_rate=learning_rate,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,

        # Evaluation & Saving
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        # Trên CPU nhỏ (smoke-test) có thể gây lỗi khi lưu optimizer lớn -> tắt auto-save
        save_strategy=("no" if device == "cpu" else "steps"),
        save_steps=SAVE_STEPS,
        save_total_limit=2,           # Chỉ giữ 2 checkpoint gần nhất
        # Nếu tắt save (ví dụ trên CPU smoke-test), không thể load best model at end
        load_best_model_at_end=(False if device == "cpu" else True),
        metric_for_best_model="rougeL",
        greater_is_better=True,

        # Logging
        logging_dir=LOG_DIR,
        logging_steps=LOGGING_STEPS,
        report_to="none",              # Không dùng WandB hay TensorBoard

        # Generation config (dùng cho evaluation)
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,

        # Hiệu suất
        fp16=use_fp16,
        dataloader_num_workers=0,      # 0 để tránh lỗi multiprocessing trên Windows
        
    )

    # --- Bước 6: Khởi tạo Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3)
        ],
    )

    # --- Bước 7: Huấn luyện ---
    logger.info("=" * 60)
    logger.info("🚀 BẮT ĐẦU HUẤN LUYỆN...")
    logger.info("=" * 60)

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Huấn luyện bị dừng bởi người dùng (Ctrl+C).")

    # --- Bước 8: Lưu model tốt nhất ---
    logger.info(f"Đang lưu model vào: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("=" * 60)
    logger.info(f"✅ HUẤN LUYỆN HOÀN TẤT! Model lưu tại: {output_dir}")
    logger.info("=" * 60)

    # --- Bước 9: Đánh giá cuối cùng ---
    logger.info("Đang đánh giá model trên tập validation...")
    eval_results = trainer.evaluate()
    logger.info(f"Kết quả đánh giá: {eval_results}")

    metrics_path = Path(output_dir) / "eval_results.json"
    save_json(
        {
            "model_name": MODEL_NAME,
            "dataset_name": dataset_name,
            "output_dir": output_dir,
            "max_samples": max_samples,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "metrics": eval_results,
        },
        str(metrics_path),
    )
    logger.info(f"Đã lưu kết quả đánh giá tại: {metrics_path}")

    return eval_results


# ==============================================================================
# ENTRY POINT (CLI)
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune VietAI/vit5-base cho bài toán tóm tắt tiếng Việt"
    )
    parser.add_argument(
        "--local_data",
        type=str,
        default=None,
        help="Đường dẫn file CSV/JSON nội bộ (nếu không set sẽ dùng Hugging Face Hub)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="thanhnew2001/vnexpress",
        help="Dataset Hugging Face dùng để train nếu không truyền --local_data",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Số sample tối đa dùng để train (mặc định: 5000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Số epoch huấn luyện (mặc định: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (mặc định: 2, phù hợp máy yếu)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (mặc định: 5e-5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Thư mục lưu model (mặc định: {OUTPUT_DIR})",
    )

    args = parser.parse_args()

    train(
        local_data=args.local_data,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )
