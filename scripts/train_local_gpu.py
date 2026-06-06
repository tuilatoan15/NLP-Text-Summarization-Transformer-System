"""
scripts/train_local_gpu.py — Fine-tuning tối ưu cho GPU 4GB VRAM (RTX 3050 Ti / RTX 3060 6GB).

Chiến lược tiết kiệm VRAM:
  - batch_size=1 + gradient_accumulation=16 → effective batch=16
  - gradient_checkpointing=True (giảm ~30% VRAM)
  - fp16=True (giảm ~50% VRAM)
  - LoRA nếu vẫn OOM
  - max_input_tokens=256 (giảm từ 512)

Usage:
    # ViT5 (khả thi nhất trên 4GB)
    python scripts/train_local_gpu.py --model vit5

    # Với LoRA (nếu OOM, hoặc muốn nhanh hơn)
    python scripts/train_local_gpu.py --model vit5 --use_lora

    # BARTPho (cần ít nhất 6GB, dùng LoRA)
    python scripts/train_local_gpu.py --model bartpho --use_lora

    # mT5
    python scripts/train_local_gpu.py --model mt5 --use_lora
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Vietnamese summarization models on local 4GB GPU."
    )
    parser.add_argument("--model", default="vit5", choices=["vit5", "mt5", "bartpho"])
    parser.add_argument("--dataset_name", default="nam194/vietnews")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Số mẫu train (10000 mẫu = ~1h trên RTX 3050)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--use_lora", action="store_true",
                        help="Dùng LoRA để tiết kiệm VRAM (khuyên dùng cho 4GB)")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (8=ít VRAM, 16=chất lượng tốt hơn)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    import argparse as _ap
    from scripts.train import train_model

    # Cấu hình tối ưu cho 4GB VRAM
    train_args = _ap.Namespace(
        model=args.model,
        dataset_name=args.dataset_name,
        local_data=None,
        max_samples=args.max_samples,
        output_dir=args.output_dir,

        # ─── VRAM tiết kiệm ───────────────────────────────
        batch_size=1,                       # Nhỏ nhất có thể
        gradient_accumulation_steps=16,     # Effective batch = 16
        eval_batch_size=2,
        gradient_checkpointing=True,        # -30% VRAM, +20% thời gian

        # ─── Learning rate ───────────────────────────────
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=200,
        lr_scheduler_type="cosine",

        # ─── Token limits (giảm để tiết kiệm VRAM) ───────
        max_input_tokens=256,               # RTX 3050 không đủ cho 512
        max_target_tokens=64,

        # ─── Training schedule ───────────────────────────
        epochs=args.epochs,
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        early_stopping_patience=3,
        auto_schedule=True,

        # ─── LoRA ────────────────────────────────────────
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,         # alpha = 2 * rank (best practice)
        lora_dropout=0.05,

        # ─── Other ───────────────────────────────────────
        no_cache=args.no_cache,
        no_save=False,
        skip_rouge_eval=True,               # Dùng eval_loss (nhanh hơn, ít VRAM hơn)
        use_augmentation=False,
    )

    print(f"\n{'='*60}")
    print(f"  Training: {args.model.upper()} on nam194/vietnews")
    print(f"  VRAM mode: {'LoRA (param-efficient)' if args.use_lora else 'Full fine-tune'}")
    print(f"  Samples  : {args.max_samples:,}")
    print(f"  Eff. batch: {train_args.batch_size * train_args.gradient_accumulation_steps}")
    print(f"  Input tokens: {train_args.max_input_tokens}")
    print(f"{'='*60}\n")

    metrics = train_model(train_args)
    print(f"\nTraining complete! Eval metrics: {metrics}")
