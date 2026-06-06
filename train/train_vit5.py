"""Backward-compatible entry point for ViT5 fine-tuning.

Prefer:
    python scripts/train.py --model vit5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune VietAI/vit5-base for Vietnamese summarization.")
    parser.add_argument("--local_data", default=None)
    parser.add_argument("--dataset_name", default="nam194/vietnews")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", default="models/vit5-finetuned")
    parser.add_argument("--no_cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    legacy = parse_args()
    args = argparse.Namespace(
        model="vit5",
        local_data=legacy.local_data,
        dataset_name=legacy.dataset_name,
        max_samples=legacy.max_samples,
        epochs=legacy.epochs,
        batch_size=legacy.batch_size,
        eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=legacy.lr,
        weight_decay=0.01,
        warmup_steps=100,
        max_input_tokens=512,
        max_target_tokens=128,
        eval_steps=500,
        save_steps=500,
        logging_steps=50,
        early_stopping_patience=3,
        output_dir=legacy.output_dir,
        no_cache=legacy.no_cache,
    )
    train_model(args)
