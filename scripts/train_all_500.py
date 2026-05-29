"""
train_all_500.py — Training tuần tự 3 mô hình tóm tắt tiếng Việt
Dataset: 8Opt/vietnamese-summarization-dataset-0001
Mỗi model: 500 mẫu train, 100 mẫu eval, 3 epochs

Cách chạy:
    python scripts/train_all_500.py
"""

from __future__ import annotations

import sys
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
from argparse import Namespace

# Thêm project root vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import logger

DATASET = "8Opt/vietnamese-summarization-dataset-0001"
TRAIN_SAMPLES = 500
EVAL_SAMPLES  = 100
EPOCHS        = 3

# Cấu hình từng model
MODEL_CONFIGS = [
    {
        "key":              "vit5",
        "output_dir":       "models/vit5-finetuned",
        "batch_size":       4,
        "grad_accum":       4,
        "lr":               5e-5,
        "warmup_steps":     50,
        "skip_rouge_eval":  False,
    },
    {
        "key":              "mt5",
        "output_dir":       "models/mt5-finetuned",
        "batch_size":       4,
        "grad_accum":       4,
        "lr":               5e-5,
        "warmup_steps":     50,
        "skip_rouge_eval":  False,
    },
    {
        "key":              "bartpho",
        "output_dir":       "models/bartpho-finetuned",
        "batch_size":       2,
        "grad_accum":       8,
        "lr":               3e-5,
        "warmup_steps":     50,
        "skip_rouge_eval":  False,
    },
]


def make_args(cfg: dict) -> Namespace:
    """Tạo Namespace args tương thích với scripts/train.py::train_model()."""
    return Namespace(
        model=cfg["key"],
        dataset_name=DATASET,
        local_data=None,
        max_samples=TRAIN_SAMPLES,
        epochs=EPOCHS,
        batch_size=cfg["batch_size"],
        eval_batch_size=4,
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        weight_decay=0.01,
        warmup_steps=cfg["warmup_steps"],
        max_input_tokens=512,
        max_target_tokens=128,
        eval_steps=100,
        save_steps=100,
        logging_steps=20,
        early_stopping_patience=3,
        auto_schedule=True,
        no_cache=True,          # Không dùng cache để luôn dùng dataset mới
        no_save=False,          # Lưu model sau training
        skip_rouge_eval=cfg["skip_rouge_eval"],
        output_dir=cfg["output_dir"],
    )


def train_one(cfg: dict) -> dict:
    key = cfg["key"]
    report_file = Path(cfg["output_dir"]) / "training_report.json"
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            logger.info("⏭️ [%s] Đã có kết quả huấn luyện cũ. Bỏ qua huấn luyện.", key.upper())
            return {
                "status": "success",
                "key": key,
                "elapsed_minutes": 0.0,
                "metrics": saved.get("metrics", {}),
            }
        except Exception as exc:
            logger.warning("⚠️ Lỗi khi đọc training report của %s: %s. Tiến hành training lại.", key, exc)

    logger.info("=" * 60)
    logger.info("🚀 Bắt đầu training: %s", key.upper())
    logger.info("   Dataset : %s", DATASET)
    logger.info("   Samples : %d train / %d eval", TRAIN_SAMPLES, EVAL_SAMPLES)
    logger.info("   Epochs  : %d", EPOCHS)
    logger.info("   LR      : %s", cfg["lr"])
    logger.info("   Batch   : %d × %d = %d effective",
                cfg["batch_size"], cfg["grad_accum"],
                cfg["batch_size"] * cfg["grad_accum"])
    logger.info("=" * 60)

    t0 = time.perf_counter()
    try:
        from scripts.train import train_model
        args = make_args(cfg)
        metrics = train_model(args)
        elapsed = time.perf_counter() - t0
        logger.info("✅ [%s] Hoàn tất sau %.1f phút", key, elapsed / 60)
        return {
            "status": "success",
            "key": key,
            "elapsed_minutes": round(elapsed / 60, 2),
            "metrics": metrics,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("❌ [%s] Lỗi sau %.1f phút: %s", key, elapsed / 60, exc)
        traceback.print_exc()
        return {
            "status": "error",
            "key": key,
            "elapsed_minutes": round(elapsed / 60, 2),
            "error": str(exc),
        }


def main():
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║   TRAINING 3 MÔ HÌNH TÓM TẮT TIẾNG VIỆT — 500 MẪU     ║")
    logger.info("║   Dataset: 8Opt/vietnamese-summarization-dataset-0001   ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    overall_start = time.perf_counter()
    results = []

    for cfg in MODEL_CONFIGS:
        result = train_one(cfg)
        results.append(result)

        # Clear GPU cache giữa các model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared after [%s]", cfg["key"])
        except Exception:
            pass

    total_elapsed = time.perf_counter() - overall_start
    logger.info("=" * 60)
    logger.info("🏁 Tất cả model đã hoàn tất sau %.1f phút", total_elapsed / 60)

    # Lưu kết quả tổng hợp
    report = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": DATASET,
        "train_samples": TRAIN_SAMPLES,
        "eval_samples": EVAL_SAMPLES,
        "epochs": EPOCHS,
        "total_elapsed_minutes": round(total_elapsed / 60, 2),
        "results": results,
    }
    out_path = Path("storage/results/training_summary_500.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("📄 Kết quả training đã lưu: %s", out_path)

    # In tóm tắt
    print("\n" + "=" * 60)
    print("KẾT QUẢ TRAINING:")
    for r in results:
        status = "✅" if r["status"] == "success" else "❌"
        print(f"  {status} [{r['key'].upper():8}] {r['elapsed_minutes']:.1f} phút  ", end="")
        if r["status"] == "success" and r.get("metrics"):
            m = r["metrics"]
            rouge = m.get("eval_rougeL", m.get("eval_loss", "N/A"))
            print(f"eval_loss={m.get('eval_loss', 'N/A'):.4f}" if isinstance(m.get('eval_loss'), float) else f"{rouge}")
        else:
            print(r.get("error", ""))
    print("=" * 60)
    return report


if __name__ == "__main__":
    main()
