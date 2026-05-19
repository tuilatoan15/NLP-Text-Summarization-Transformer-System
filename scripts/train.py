"""CLI: fine-tune ViT5, mT5, or BARTPho on cleaned VNExpress data."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src import config
from src.evaluate import compute_rouge_batch
from src.model_registry import ABSTRACTIVE_ALGORITHMS, resolve_algorithm
from src.preprocess import clean_generated_summary
from src.utils import logger, save_json
from train.dataset_loader import load_vnexpress_dataset


def _prefix(model_key: str, text: str) -> str:
    return f"summarize: {text}" if model_key in {"vit5", "mt5"} else text


def tokenize_batch(examples, tokenizer, model_key: str, max_input_len: int, max_target_len: int):
    inputs = [_prefix(model_key, str(article)) for article in examples["article"]]
    targets = [str(summary) for summary in examples["title"]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_len,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_len,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decoded_preds = [clean_generated_summary(text) for text in decoded_preds]
        decoded_labels = [clean_generated_summary(text) for text in decoded_labels]
        return compute_rouge_batch(decoded_preds, decoded_labels)

    return compute_metrics


def train_model(args) -> dict:
    algorithm = resolve_algorithm(args.model)
    if algorithm.key not in ABSTRACTIVE_ALGORITHMS:
        raise ValueError("--model must be one of: vit5, mt5, bartpho")

    output_dir = Path(args.output_dir or algorithm.local_dir or config.MODEL_DIR / f"{algorithm.key}-finetuned")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_source = algorithm.model_name
    logger.info("Training %s from %s", algorithm.name, model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_source)

    dataset = load_vnexpress_dataset(
        local_csv_path=args.local_data,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        use_cache=not args.no_cache,
    )

    tokenized = dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, algorithm.key, args.max_input_tokens, args.max_target_tokens),
        batched=True,
        batch_size=32,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        label_smoothing_factor=0.1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=64,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        logging_steps=args.logging_steps,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    save_json(
        {
            "algorithm": algorithm.key,
            "base_model": model_source,
            "dataset": args.dataset_name,
            "max_samples": args.max_samples,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "metrics": metrics,
        },
        output_dir / "training_report.json",
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Vietnamese summarization model.")
    parser.add_argument("--model", default="vit5", choices=["vit5", "mt5", "bartpho"])
    parser.add_argument("--dataset_name", default=config.DATASET_NAME)
    parser.add_argument("--local_data", default=None)
    parser.add_argument("--max_samples", type=int, default=config.MAX_TRAIN_SAMPLES)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--eval_batch_size", type=int, default=config.EVAL_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=config.WARMUP_STEPS)
    parser.add_argument("--max_input_tokens", type=int, default=config.MAX_INPUT_TOKENS)
    parser.add_argument("--max_target_tokens", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--no_cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
